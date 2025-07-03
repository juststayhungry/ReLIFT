# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from collections import defaultdict, Counter
from tensordict import TensorDict
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
import torch

from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer, 
    Role, 
    ResourcePoolManager, 
    WorkerType, 
    _timer, 
    # compute_data_metrics, 
    compute_timing_metrics, 
    dataprotoitem_to_dataproto, 
    # compute_advantage, 
    reduce_metrics
)
from verl.utils.torch_functional import masked_mean


# directly copied from verl/trainer/ppo/ray_trainer.py
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics

def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, grpo_use_std=True):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        use_std=grpo_use_std)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns

    elif adv_estimator == 'reinforce':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                             eos_mask=response_mask,
                                                                             index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data

class ReLIFTRayPPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        

        assert self.hybrid_engine, 'Currently, only support hybrid engine'
        # breakpoint()
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.dual_actor = Role.ActorRollout2 in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    # def _validate(self):
    #     reward_tensor_lst1, data_source_lst1 = [], []
    #     reward_tensor_lst2, data_source_lst2 = [], []

    #     for test_data in self.val_dataloader:
    #         test_batch = DataProto.from_single_dict(test_data)

    #         if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
    #             return {}

    #         n_val_samples = self.config.actor_rollout_ref.rollout.n_val
    #         test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
    #         test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
    #         test_gen_batch.meta_info = {
    #             'eos_token_id': self.tokenizer.eos_token_id,
    #             'pad_token_id': self.tokenizer.pad_token_id,
    #             'recompute_log_prob': False,
    #             'do_sample': False,
    #             'validate': True,
    #         }

    #         # pad to be divisible by dp size
    #         test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
    #         test_gen_batch_padded.meta_info['val_temperature'] = self.config.actor_rollout_ref.rollout.val_temperature

    #         # === Actor 1 ===
    #         test_output_gen_batch_padded1 = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
    #         test_output_gen_batch1 = unpad_dataproto(test_output_gen_batch_padded1, pad_size=pad_size)
    #         test_batch1 = test_batch.union(test_output_gen_batch1)

    #         reward_tensor1 = self.val_reward_fn(test_batch1)
    #         reward_tensor_lst1.append(reward_tensor1)
    #         data_source_lst1.append(test_batch1.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor1.shape[0]))

    #         # === Actor 2 ===
    #         if self.dual_actor:
    #             test_output_gen_batch_padded2 = self.actor_rollout_wg2.generate_sequences(test_gen_batch_padded)
    #             test_output_gen_batch2 = unpad_dataproto(test_output_gen_batch_padded2, pad_size=pad_size)
    #             # test_batch2 = test_batch.union(test_output_gen_batch2)
    #             test_batch2 = DataProto(
    #                 batch={**test_batch.batch, **test_output_gen_batch2.batch},
    #                 non_tensor_batch={**test_batch.non_tensor_batch, **test_output_gen_batch2.non_tensor_batch},
    #                 meta_info={**test_batch.meta_info, **test_output_gen_batch2.meta_info},
    #             )
    #             reward_tensor2 = self.val_reward_fn(test_batch2)
    #             reward_tensor_lst2.append(reward_tensor2)
    #             data_source_lst2.append(test_batch2.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor2.shape[0]))

    #         print('Validation: Generation end.')

    #     # ===== Actor 1 metrics =====
    #     metric_dict1 = {}
    #     rewards1 = torch.cat(reward_tensor_lst1, dim=0).sum(-1).cpu()
    #     data_sources1 = np.concatenate(data_source_lst1, axis=0)
    #     data_source_reward1 = {}
    #     for i in range(rewards1.shape[0]):
    #         ds = data_sources1[i]
    #         data_source_reward1.setdefault(ds, []).append(rewards1[i].item())
    #     for ds, rewards in data_source_reward1.items():
    #         metric_dict1[f'actor1/val/test_score/{ds}'] = np.mean(rewards)

    #     # ===== Actor 2 metrics (if dual actor) =====
    #     metric_dict2 = {}
    #     if self.dual_actor:
    #         rewards2 = torch.cat(reward_tensor_lst2, dim=0).sum(-1).cpu()
    #         data_sources2 = np.concatenate(data_source_lst2, axis=0)
    #         data_source_reward2 = {}
    #         for i in range(rewards2.shape[0]):
    #             ds = data_sources2[i]
    #             data_source_reward2.setdefault(ds, []).append(rewards2[i].item())
    #         for ds, rewards in data_source_reward2.items():
    #             metric_dict2[f'actor2/val/test_score/{ds}'] = np.mean(rewards)

    #     # Return merged metrics
    #     if self.dual_actor:
    #         return {**metric_dict1, **metric_dict2}
    #     else:
    #         return metric_dict1

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        from torch.utils.data import DataLoader, SequentialSampler
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        from .rl_dataset_with_target import RLHFDatasetWithTarget
        self.train_dataset = RLHFDatasetWithTarget(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True, return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error',
                                         max_target_length=self.config.data.max_target_len)

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            from verl.relift.rl_dataset_with_target import ResumableRandomSampler
            sampler = ResumableRandomSampler(data_source=self.train_dataset)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)
        
        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            if self.dual_actor:
                self.config.actor_rollout_ref2.actor.optim.total_training_steps = total_training_steps
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def replace_response_in_batch(self, batch):
        """
            Replace generated responses with ground truth tgt_input_ids in batch.
        """

        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        
        batch = batch.batch
        prompts = batch['prompts'] # b x p_l
        tgt_input_ids = batch['tgt_input_ids'].clone() # b x r_l
        
        # add eos token
        seq_len = tgt_input_ids.shape[1]
        tgt_lengths = (tgt_input_ids != pad_token_id).sum(dim=1)
        replace_positions = torch.where(tgt_lengths < seq_len, tgt_lengths, seq_len - 1)
        tgt_input_ids[torch.arange(len(tgt_input_ids)), replace_positions] = eos_token_id

        original_attention_mask = batch['attention_mask'][:, :prompts.size(1)] # b x p_l
        original_position_ids = batch['position_ids'][:, :prompts.size(1)] # b x p_l
        
        device = prompts.device
        batch_size = prompts.size(0)
        response_length = tgt_input_ids.size(1) # r_l
        
        # replace responses
        batch['responses'] = tgt_input_ids
        
        # replace input_ids (prompt + tgt)
        batch['input_ids'] = torch.cat([prompts, tgt_input_ids], dim=-1)
        
        # find eos_token of tgt_input_ids, and mask the tokens before eos_token and eos_token
        response_attention_mask = get_eos_mask(tgt_input_ids, eos_token_id, 
                                                dtype=original_attention_mask.dtype)
        
        batch['attention_mask'] = torch.cat(
            [original_attention_mask, response_attention_mask], dim=-1)
        
        delta_pos = torch.arange(1, response_length+1, device=device)\
                    .expand(batch_size, response_length)
        last_prompt_pos = original_position_ids[:, -1].unsqueeze(-1)
        response_position_ids = last_prompt_pos + delta_pos
        
        batch['position_ids'] = torch.cat(
            [original_position_ids, response_position_ids], dim=-1)

        return
        

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf
        

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            pprint(f'Initial validation metrics: {val_metrics}')
            if self.dual_actor:
                val_metrics, val_metrics2 = self._validate()
                # 示例：WandB 的嵌套记录
                logger.log({
                    "actor": val_metrics,
                    "actor2": val_metrics2
                }, step=self.global_steps)
            else:
                val_metrics = self._validate()
                pprint(f'Initial validation metrics: {val_metrics}')
                logger.log(val_metrics, step=self.global_steps)
            # logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1
        # breakpoint()
        n_samples = self.config.actor_rollout_ref.rollout.n
        sft_buffer_batch = None

        batch_size = self.config.data.train_batch_size
        n_samples = self.config.actor_rollout_ref.rollout.n

        #breakpoint()

        for _ in range(self.config.trainer.total_epochs):
            
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)   
                if self.dual_actor:   
                    batch_dict2 = batch_dict.copy()         
                    batch2: DataProto = DataProto.from_single_dict(batch_dict2)   
                
                #batch_dict['input_ids'].shape torch.Size([4, 512])
                #batch_dict['sample_id']
                #array([10897, 45416, 23865, 26862], dtype=object)
                metrics = {}
                if self.dual_actor:
                    metrics2 = {}
                    metrics_combined = {}
                timing_raw = {}

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                if self.dual_actor:
                    gen_batch2 = batch2.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                    gen_batch2.meta_info['global_steps'] = self.global_steps
                gen_batch.meta_info['global_steps'] = self.global_steps
                
                with _timer('step', timing_raw):#记录timing_raw这个dict中关于step这个key的耗时为value
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        if self.dual_actor:
                            gen_batch_output2 = self.actor_rollout_wg2.generate_sequences(gen_batch2)
                            combined_gen = DataProto.concat([gen_batch_output, gen_batch_output2])
                        #将gen_batch copy的rollout_n次
                    # This code matches a prompt ID with its N responses.
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    if self.dual_actor:
                        batch2.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch2.batch))],
                                                                dtype=object)
                        batch2 = batch2.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch2 = batch2.union(gen_batch_output2)


                        batch_combined = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n*2, interleave=True)
                        # batch_combined.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch_combined.batch))],
                        #                                         dtype=object)
                        batch_combined = batch_combined.union(combined_gen)
                        
                    # else:
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)#将question与answer合并(因为n次)
                        # batch = batch.union(gen_batch_output2)
                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                    with _timer('adv', timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            if self.dual_actor:
                                pass
                                reward_tensor2 = self.rm_wg.compute_rm_score(batch2)
                                batch2 = batch2.union(reward_tensor2)
                                # # reward_tensor_combined = self.rm_wg.compute_rm_score(batch_combined)
                                # batch_combined = batch_combined.union(reward_tensor_combined)
                            # else:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        if self.dual_actor:
                            reward_tensor2 = self.reward_fn(batch2) # [bsz, l], only the last valid token has reward
                            reward_tensor_combined = self.reward_fn(batch_combined)
                        reward_tensor = self.reward_fn(batch) # [bsz, l], only the last valid token has reward
                        
                        # batch.batch['token_level_scores'] = reward_tensor
                        if self.dual_actor:
                            batch2.batch['token_level_scores'] = reward_tensor2
                            batch_combined.batch['token_level_scores'] = reward_tensor_combined
                            # uids2 = batch2.non_tensor_batch['uid']
                            
                        # else:
                        batch.batch['token_level_scores'] = reward_tensor
                        uids = batch.non_tensor_batch['uid']
                        seen = set()
                        unique_uids = [uid for uid in uids if uid not in seen and not seen.add(uid)]
                        # if self.dual_actor:
                        #     unique_uids2 = [uid for uid in uids2 if uid not in seen and not seen.add(uid)]
                        
                        if self.config.data.reward_impl_version == 0:
                            fail_value = 0
                            success_value = 1
                            format_value = -1 # not defined.
                        elif self.config.data.reward_impl_version == 1:
                            fail_value = -0.5
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 2:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 3:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 4:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        else:
                            raise ValueError(f'Invalid reward implementation version: {self.config.data.reward_impl_version}')
                        
                        solve_none = 0
                        solve_all = 0
                        solve_none_format = 0

                        solve_none_uids = []
                        uid2solve_num = {}
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
                            
                            # Check if all rewards are 0 or all are 1 for this uid
                            if (uid_rewards == fail_value).all():
                                solve_none_uids.append(uid)
                                solve_none += 1
                            elif (uid_rewards == success_value).all():
                                solve_all += 1
                            elif (uid_rewards == format_value).all():
                                solve_none_format += 1

                            uid2solve_num[uid] = uid_rewards.sum().item()

                        # Log to metrics
                        metrics['batch/solve_none'] = solve_none
                        metrics['batch/solve_none_format'] = solve_none_format
                        metrics['batch/solve_all'] = solve_all

                        metrics['batch/solved'] = (reward_tensor.sum(-1) == success_value).sum().item() / len(uids)
                        metrics['batch/failed'] = (reward_tensor.sum(-1) == fail_value).sum().item() / len(uids)

                        # how to buffer samples for subsequent SFT
                        sft_buffer_uids = solve_none_uids
                        
                        # create buffer batch
                        buffer_indexes = []
                        for i, uid in enumerate(unique_uids):
                            if uid in sft_buffer_uids:
                                buffer_indexes.append(i * n_samples)

                        # update sft_buffer_batch
                        if buffer_indexes:
                            if self.dual_actor:
                                buffer_batch = batch_combined.slice(buffer_indexes)
                            else:
                                buffer_batch = batch.slice(buffer_indexes)
                            
                            if sft_buffer_batch is not None:
                                sft_buffer_batch = DataProto.concat([buffer_batch, sft_buffer_batch])
                            else:
                                sft_buffer_batch = buffer_batch
                          
                        # recompute old_log_probs
                        with _timer('old_log_prob', timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)
                            if self.dual_actor:
                                old_log_prob2 = self.actor_rollout_wg2.compute_log_prob(batch2)
                                batch2 = batch2.union(old_log_prob2)
                                # old_log_probs_combined = DataProto.concat([old_log_prob,old_log_prob2]) #.concat()
                                # batch_combined = batch_combined.union(old_log_probs_combined)
                                # 拼接 old_log_probs
                                old_log_probs_combined = torch.cat(
                                    [batch.batch['old_log_probs'], batch2.batch['old_log_probs']], dim=0
                                )
                                # 只传 batch，不传 non_tensor_batch 和 meta_info
                                temp_dp = DataProto(
                                    batch=TensorDict({"old_log_probs": old_log_probs_combined}, batch_size=[old_log_probs_combined.shape[0]]),
                                    # non_tensor_batch={},  # ✅ 避免 union 冲突
                                    # meta_info={},          # ✅ 避免 union 冲突
                                )

                                # 执行 union
                                batch_combined = batch_combined.union(temp_dp)


                        if self.use_reference_policy:#False
                            # breakpoint()
                            # compute reference log_prob
                            with _timer('ref', timing_raw):
                                if self.dual_actor:
                                    ref_log_prob2 = self.ref_policy_wg2.compute_ref_log_prob(gen_batch_output2)
                                    batch2 = batch.union(ref_log_prob2)
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                                # batch_combined['old_log_probs']
                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):#True
                            # breakpoint()
                            if self.dual_actor:
                                batch_combined, kl_metrics_combined = apply_kl_penalty(batch_combined,
                                                                    kl_ctrl=self.kl_ctrl,
                                                                    kl_penalty=self.config.algorithm.kl_penalty)
                                metrics_combined.update(kl_metrics_combined)
                                batch2, kl_metrics2 = apply_kl_penalty(batch2,
                                                                    kl_ctrl=self.kl_ctrl,
                                                                    kl_penalty=self.config.algorithm.kl_penalty)
                                metrics2.update(kl_metrics2)
                            # else:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                kl_ctrl=self.kl_ctrl,
                                                                kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            if self.dual_actor:
                                batch_combined.batch['token_level_rewards'] = batch_combined.batch['token_level_scores']
                            else:
                                batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # NOTE: the advantages are the same for all tokens in the response
                        # compute advantages, executed on the driver process
                        #tlx
                        if self.dual_actor:
                            batch_combined = compute_advantage(batch_combined,
                                                    adv_estimator=self.config.algorithm.adv_estimator,
                                                    gamma=self.config.algorithm.gamma,
                                                    lam=self.config.algorithm.lam,
                                                    grpo_use_std=self.config.algorithm.grpo_use_std)
                            
                        else:
                            batch = compute_advantage(batch,
                                                    adv_estimator=self.config.algorithm.adv_estimator,
                                                    gamma=self.config.algorithm.gamma,
                                                    lam=self.config.algorithm.lam,
                                                    grpo_use_std=self.config.algorithm.grpo_use_std)
                            
                        # compute alpha and beta for prefix reward weighting
                        if self.dual_actor:
                            advantages = batch_combined.batch['advantages']
                            batch_combined.batch['advantages'] = advantages
                        else:
                            advantages = batch.batch['advantages']
                            batch.batch['advantages'] = advantages
                    
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.dual_actor:
                        self._balance_batch(batch_combined, metrics=metrics_combined)
                        self._balance_batch(batch2, metrics=metrics2)
                    # else:
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    if self.dual_actor:
                        batch_combined.meta_info['global_token_num'] = torch.sum(batch_combined.batch['attention_mask'], dim=-1).tolist()
                        batch2.meta_info['global_token_num'] = torch.sum(batch2.batch['attention_mask'], dim=-1).tolist()
                    else:
                        batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # update critic
                    # if self.use_critic:
                    #     with _timer('update_critic', timing_raw):
                    #         critic_output = self.critic_wg.update_critic(batch)
                    #     critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                    #     metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if self.dual_actor:
                                # breakpoint()#
                                # batch2.batch.batch_size = torch.Size([32]) 
                                # batch.batch.batch_size = torch.Size([32]) 
                                # batch2.batch["advantages"] = batch_combined.batch["advantages"]
                                # batch.batch["advantages"] = batch_combined.batch["advantages"]
                                actor_output2 = self.actor_rollout_wg2.update_actor(batch_combined)
                                actor_output = self.actor_rollout_wg.update_actor(batch_combined)
                            else:
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                            # batch的key含有之前的return['tgt_input_ids', 'position_ids', 'attention_mask', 'responses', 'prompts', 'input_ids', 'token_level_scores', 'old_log_probs', 'token_level_rewards', 'advantages', 'returns']
                        if self.dual_actor:
                            actor_output_metrics2 = reduce_metrics(actor_output2.meta_info['metrics'])
                            metrics2.update(actor_output_metrics2)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                    # # SFT update using hard-batch
                    # sft_data_size = self.config.actor_rollout_ref.actor.sft.sft_data_size
                    # if len(sft_buffer_batch) >= sft_data_size:
                    #     with _timer('sft_update_actor', timing_raw):
                    #         sft_buffer_batch.to('cpu')
                    #         sft_buffer_batch.batch.to('cpu')
                            
                    #         sft_train_batch = sft_buffer_batch.slice(range(sft_data_size))
                            
                    #         # replace on-policy with off-policy
                    #         self.replace_response_in_batch(sft_train_batch)
                            
                    #         if len(sft_buffer_batch) == sft_data_size:
                    #             sft_buffer_batch = None
                    #         else:
                    #             sft_buffer_batch = sft_buffer_batch.slice(range(sft_data_size, len(sft_buffer_batch)))

                    #         self._balance_batch(sft_train_batch, metrics=metrics)
                    #         sft_output = self.actor_rollout_wg.sft_update_actor(sft_train_batch)
                    #         sft_output_metrics = reduce_metrics(sft_output.meta_info['metrics'])
                    #         metrics.update(sft_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            if self.dual_actor:
                                # 双 Actor 情况：处理两个指标字典
                                val_metrics, val_metrics2 = self._validate()
                                # 计算第一个 Actor 的平均分
                                if 'avg_score' not in val_metrics:
                                    val_metrics['avg_score'] = np.mean([
                                        val_metrics[key] for key in val_metrics 
                                        if isinstance(key, str) and key.startswith('val/test_score/')
                                    ])
                                
                                # 计算第二个 Actor 的平均分
                                if 'avg_score' not in val_metrics2:
                                    val_metrics2['avg_score'] = np.mean([
                                        val_metrics2[key] for key in val_metrics2 
                                        if isinstance(key, str) and key.startswith('val/test_score/')
                                    ])
                                metrics.update(val_metrics)
                                metrics2.update(val_metrics2)
                            else:
                                val_metrics: dict = self._validate()
                                if 'avg_score' not in val_metrics:
                                    val_metrics['avg_score'] = np.mean([val_metrics[key] for key in val_metrics if key.startswith('val/test_score/')])
                                metrics.update(val_metrics)
                            self.maybe_save_best_hf(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                    if self.config.trainer.get("track_or_not", False):
                        if self.global_steps % self.config.trainer.track_freq == 0:
                            path = os.path.join(self.config.trainer.default_local_dir, f'step_{self.global_steps}')
                            self.actor_rollout_wg.save_checkpoint_hf(path)

                if self.dual_actor:
                    # metrics_combined.update(compute_data_metrics_ours(batch=batch2, use_critic=self.use_critic))
                    # metrics_combined.update(compute_timing_metrics(batch=batch2, timing_raw=timing_raw))
                    # collect metrics
                    metrics_combined.update(compute_data_metrics_ours(batch=batch_combined, use_critic=self.use_critic))
                    metrics_combined.update(compute_timing_metrics(batch=batch_combined, timing_raw=timing_raw))
                    metrics_combined.update({f"actor/{k}": v for k, v in metrics.items()})
                    metrics_combined.update({f"actor2/{k}": v for k, v in metrics2.items()})
                    logger.log(data=metrics_combined, step=self.global_steps)
                else:
                    # collect metrics
                    metrics.update(compute_data_metrics_ours(batch=batch, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    # TODO: make a canonical logger that supports various backend
                    logger.log(data=metrics, step=self.global_steps)
                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        if self.dual_actor:
                            val_metrics, val_metrics2 = self._validate()
                            # 示例：WandB 的嵌套记录
                            logger.log({
                                "actor": val_metrics,
                                "actor2": val_metrics2
                            }, step=self.global_steps)
                        else:
                            val_metrics = self._validate()
                            pprint(f'Initial validation metrics: {val_metrics}')
                            logger.log(val_metrics, step=self.global_steps)
                    return

    def maybe_save_best_hf(self, val_metrics: dict):
        import json
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'best',
                                        f'actor')
        
        os.makedirs(actor_local_path, exist_ok=True)
        if os.path.exists(f'{actor_local_path}/metrics.json'):
            with open(f'{actor_local_path}/metrics.json', 'r') as f:
                if self.dual_actor:
                    metrics_combined = json.load(f)
                    best_score = metrics_combined['best_avg_score']
                else:
                    metrics = json.load(f)
                    best_score = metrics['best_avg_score']
        else:
            print('Find no current best saved. Best score is set to -inf')
            best_score = -float('inf')
        
        cur_score = val_metrics['avg_score']
        
        if cur_score > best_score:
            print(f'Saving best checkpoint with score {cur_score} at {actor_local_path}')
            best_score = cur_score
            self.actor_rollout_wg.save_checkpoint_hf(actor_local_path)
            with open(f'{actor_local_path}/metrics.json', 'w') as f:
                f.write(json.dumps({'best_avg_score': best_score, 'global_step': self.global_steps})+'\n')
        
def compute_data_metrics_ours(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    from verl.trainer.ppo.ray_trainer import _compute_response_info
    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    # compute on/off policy stats
    on_response_length = response_length
    on_sequence_score = sequence_score

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # on/off policy response length
        'on_off_metrics/on_response_length_mean':
            torch.mean(on_response_length).detach().item(),
        'on_off_metrics/on_score':
            torch.mean(on_sequence_score).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics
