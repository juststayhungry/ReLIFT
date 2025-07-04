set -x

# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS

ray stop 
ray start --head --num-cpus=100

export MODEL_PATH=$MODEL_ROOT_PATH/Qwen2.5-7B-think
export DATA_DIR=./dataset/

export EXP_NAME=7b_relift_test
export WANDB_PROJECT="ReLIFT"

# Train over a single node, 8 A100-80GB GPUs.
python3 -u -m verl.relift.main_ppo \
    actor_rollout_ref.actor.sft.sft_epochs=1 \
    actor_rollout_ref.actor.sft.sft_data_size=128 \
    actor_rollout_ref.actor.sft.sft_mini_batch_size=128 \
    actor_rollout_ref.actor.sft.sft_micro_batch_size=2 \
    actor_rollout_ref.actor.sft.entropy_coeff=0.001 \
    actor_rollout_ref.actor.optim.sft.lr=1e-6 \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/openr1.parquet \
    data.val_files=$DATA_DIR/valid.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.max_target_len=8192 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.grad_clip=0.7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.sft_param_offload=False \
    actor_rollout_ref.actor.fsdp_config.sft_grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.sft_optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.ref.use_ref=False \
    algorithm.grpo_use_std=False \
    actor_rollout_ref.actor.loss_remove_token_mean=True \
    actor_rollout_ref.actor.loss_remove_clip=True \
    data.reward_impl_version=3 \
    trainer.max_optim_to_keep=2 \
    data.shuffle=True \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=./train_results/${WANDB_PROJECT}/${EXP_NAME} \
    trainer.total_epochs=3
