cd /dlabscratch1/amani/LLM-RL/
source /dlabscratch1/amani/miniconda3/bin/activate
conda activate LLM-RL


# python3 src/train.py experiment=grpo data=openmathr1 \
#     model_path="Qwen/Qwen2-0.5B" \
#     data.train_dataset_type=adaptive data.sampler=null data.train_size=256 data.test_size=100 \
#     trainer.n_gpus_per_node=1 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=4 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 actor_rollout_ref.actor.optim.lr=1e-6 critic.optim.lr=1e-5 \
#     trainer.total_epochs=10 actor_rollout_ref.actor.ppo_epochs=1 data.train_batch_size=4 trainer.test_freq=1000

python3 src/train.py experiment=grpo data=gsm8k \
    model_path="meta-llama/Meta-Llama-3-8B" \
    data.train_dataset_type=base data.sampler=null \
    trainer.n_gpus_per_node=1 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.actor.optim.lr=1e-6 critic.optim.lr=1e-5 \
    trainer.total_epochs=10 actor_rollout_ref.actor.ppo_epochs=1 data.train_batch_size=32 trainer.test_freq=1000


# python3 src/train.py experiment=grpo data=gsm8k \
#     model_path="meta-llama/Llama-3.2-3B-instruct" \
#     data.train_dataset_type=adaptive data.sampler=null \
#     trainer.n_gpus_per_node=1 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=32 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.actor.optim.lr=1e-6 critic.optim.lr=1e-5 \
#     trainer.total_epochs=10 actor_rollout_ref.actor.ppo_epochs=1 data.train_batch_size=32 trainer.test_freq=1000
    
    # data.train_batch_size=2 \
    # actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    # actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    # actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2


# python3 src/train.py experiment=grpo data=cumulative_parity_r3_length_16_bitwidth_1_1024_512 \
#     model_path="meta-llama/Llama-3.2-3B-instruct" \
#     data.train_dataset_type=base data.sampler=null \
#     trainer.n_gpus_per_node=1 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=32 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.actor.optim.lr=1e-6 critic.optim.lr=1e-5 \
#     trainer.total_epochs=10 actor_rollout_ref.actor.ppo_epochs=1 data.train_batch_size=32 trainer.test_freq=1000
    
    # data.train_batch_size=2 \
    # actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    # actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    # actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2


# python3 src/train.py experiment=grpo data.sampler=null data=gsm8k_r3 data.train_dataset_type=base \
#     model_path="meta-llama/Llama-3.2-3B" \
#     trainer.n_gpus_per_node=1 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=32 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 actor_rollout_ref.actor.optim.lr=1e-6 critic.optim.lr=1e-5 \
#     trainer.total_epochs=10 actor_rollout_ref.actor.ppo_epochs=1 data.train_batch_size=32 trainer.test_freq=40


# python3 src/train.py experiment=grpo data.sampler=null data=cumulative_parity_length_16_bitwidth_1_2048_512 \
# model_path="masani/SFT_cumulative_parity_length_16_bitwidth_1_2048_512_Qwen2-1.5B_epoch_16_global_step_128" \
# data.train_dataset_type=adaptive \
# trainer.n_gpus_per_node=4 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=24 \
# actor_rollout_ref.actor.ppo_mini_batch_size=256 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=24 \
# actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=24 actor_rollout_ref.actor.optim.lr=1e-6 critic.optim.lr=1e-5 \
# trainer.total_epochs=10000 actor_rollout_ref.actor.ppo_epochs=1 data.train_batch_size=256 trainer.test_freq=40
    
# ### END OF COMMAND ###

# python3 src/train.py experiment=grpo data.sampler=null data=cumulative_parity_r3_length_16_bitwidth_1_1024_512 \
# model_path="meta-llama/Llama-3.2-3B-instruct" \
# data.train_dataset_type=base \
# trainer.n_gpus_per_node=1 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=24 \
# actor_rollout_ref.actor.ppo_mini_batch_size=256 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=24 \
# actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=24 actor_rollout_ref.actor.optim.lr=1e-6 critic.optim.lr=1e-5 \
# trainer.total_epochs=10000 actor_rollout_ref.actor.ppo_epochs=1 data.train_batch_size=256 trainer.test_freq=40

