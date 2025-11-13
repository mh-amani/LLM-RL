# cd /dlabscratch1/amani/LLM-RL/
# source /dlabscratch1/amani/miniconda3/bin/activate
# conda activate verl

# set -x

# # to train:
# torchrun --nproc_per_node=8 scripts/sft_on_dataset/train.py \
#     data.name=math \
#     data.max_length=3072 \
#     model.partial_pretrain=mistralai/Mistral-7B-v0.1 \
#     data.micro_batch_size_per_gpu=4 \
#     data.train_batch_size=256 \
#     trainer.total_epochs=5


# torchrun --nproc_per_node=8 scripts/sft_on_dataset/train.py \
#     data.name=gsm8k \
#     data.max_length=2048 \
#     model.partial_pretrain=mistralai/Mistral-7B-v0.1 \
#     data.micro_batch_size_per_gpu=16 \
#     data.train_batch_size=256 \
#     trainer.total_epochs=5


# torchrun --nproc_per_node=1 scripts/sft_on_dataset/train.py \
#     data.name=gsm8k \
#     data.max_length=2048 \
#     model.partial_pretrain=microsoft/rho-math-1b-v0.1 \
#     data.micro_batch_size_per_gpu=32 \
#     data.train_batch_size=256 \.
#     trainer.total_epochs=5
    

# torchrun --nproc_per_node=1 scripts/sft_on_dataset/train.py \
#     data.name=parity \
#     data.max_length=2048 \
#     model.partial_pretrain=Qwen/Qwen2-0.5B \
#     data.micro_batch_size_per_gpu=4 \
#     data.train_batch_size=256 \
#     trainer.total_epochs=20


torchrun --nproc_per_node=1 scripts/sft_on_dataset/train.py \
    data.name=modgsm8k \
    optim.warmup_steps_ratio=0.01 \
    data.max_length=2048 \
    model.partial_pretrain=meta-llama/Llama-3.2-3B \
    data.micro_batch_size_per_gpu=4 \
    data.train_batch_size=256 \
    trainer.total_epochs=5

torchrun --nproc_per_node=4 scripts/sft_on_dataset/train.py \
    data.name=gsm8k \
    optim.warmup_steps_ratio=0.01 \
    data.max_length=2048 \
    model.partial_pretrain=meta-llama/Llama-2-7b-hf \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=256 \
    trainer.total_epochs=1


# # to just upload models from the above training to hub:
# torchrun --nproc_per_node=1 scripts/sft_on_dataset/train.py \
#     +just_upload_models_to_hub=True \
#     +path_to_checkpoints_folder=/dlabscratch1/amani/prod/LLM-RL/logs/SFT_for_rl/gsm8k_Llama-2-7b-hf/2025-04-24_20-49-32/checkpoints/