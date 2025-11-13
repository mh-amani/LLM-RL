# LLM-RL

## Installation

```bash
conda create -n LLM-RL python=3.11
conda activate LLM-RL
source ./install.sh
``` 

downloading the data
https://github.com/eric-haibin-lin/verl-data/tree/main

verl git version to install
git checkout 550bbbbffe23bc5450db8ce02b256eb75fbf4129

pip install "triton==3.1.0"


#### to continue training a model (actor critic etc), find the logging directory, date and time, then write the command with configs you prefer, and append `day_time=${now:%Y-%m-%d}_${now:%H-%M-%S}` (or maybe also `log_dir=... `) e.g.
```bash
python ./src/train.py experiment=qwen_gsm8k_uniform day_time="2023-10-12_16-00-00" log_dir=./logs/progressive_rl_ppo_on_gsm8k_Qwen2.5-0.5B-Instruct
```

To continue training from last steps, you need to pass `day_time=2023-10-12_16-00-00` or anything for that matter, also, for your jobs if they get preempted, you can set the day_time in the script so it 
keeps going back to the last step and loading that checkpoint


## preparing the data.

for math:
```bash
python scripts/prepare_dataset/prepare_math_dataset.py --local_dir LLM-RL/data/verl-data/math
python scripts/prepare_dataset/prepare_gsm8k_dataset.py --local_dir LLM-RL/data/verl-data/gsm8k
```

for parity;
```bash
python scripts/prepare_dataset/prepare_parity_dataset.py --length 64 --train_size 999999 --test_size 512 --local_dir data/verl-data
```

for cumulative parity:
```bash
python scripts/prepare_dataset/prepare_cumulative_parity_dataset.py --length 16 --train_size 1024 --test_size 512 --local_dir data/verl-data
```



### to sft

torchrun --nproc_per_node=4 scripts/sft_on_dataset/train.py \
data.name=cumulative_parity_length_16_bitwidth_1_2048_512 \
optim.lr=4e-6 optim.num_cycles=2 \
data.max_length=256 \
model.partial_pretrain=Qwen/Qwen2-1.5B \
data.micro_batch_size_per_gpu=16 \
data.train_batch_size=256 \
trainer.total_epochs=16



torchrun --nproc_per_node=4 scripts/sft_on_dataset/train.py \
data.name=gsm8k \
optim.lr=9e-6 optim.num_cycles=0.5 \
data.max_length=2048 \
model.partial_pretrain=meta-llama/Llama-3.2-3B \
data.micro_batch_size_per_gpu=8 \
data.train_batch_size=256 \
trainer.total_epochs=1

torchrun --nproc_per_node=4 scripts/sft_on_dataset/train.py \
data.name=gsm8k \
optim.lr=9e-6 optim.num_cycles=0.5 \
data.max_length=2048 \
model.partial_pretrain=meta-llama/Llama-2-7b-hf \
data.micro_batch_size_per_gpu=4 \
data.train_batch_size=256 \
trainer.total_epochs=1

### END OF COMMAND ###



### to RL syntethically
python3 src/train.py experiment=grpo data.sampler=null data=cumulative_parity_length_16_bitwidth_1_2048_512 \
model_path="masani/SFT_cumulative_parity_length_16_bitwidth_1_2048_512_Qwen2-1.5B_epoch_16_global_step_128" \
data.train_dataset_type=base \
trainer.n_gpus_per_node=4 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=24 \
actor_rollout_ref.actor.ppo_mini_batch_size=256 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=24 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=24 actor_rollout_ref.actor.optim.lr=1e-6 critic.optim.lr=1e-5 \
trainer.total_epochs=10000 actor_rollout_ref.actor.ppo_epochs=1 data.train_batch_size=256 trainer.test_freq=40



## Running commands

first activate the conda environment
```bash
conda activate LLM-RL
```

### Multiple GPUs with accelerate <s>god</s>deepspeed
```bash

2. In `./config/train.yaml`, set 
    ```yaml
    num_gpus: 7
    num_nodes: 1
    use_vllm: true
    vllm_device: "cuda:7"
    ``` 
3. Run following command
```bash
accelerate launch --num_processes 7 --config_file ./config/accelerate_deepspeed_zero3_config.yaml ./src/train.py experiment=qwen_gsm8k_uniform
```

### Single GPU, without accelerate or vllm

1. in `./config/train.yaml`, comment out the `use_vllm`, `vllm_device`, and `vllm_gpu_memory_utilization` fields and set `num_gpus: 1` and `num_nodes: 1`

2. Run following command
```bash
python ./src/train.py experiment=qwen_gsm8k_uniform
```


### Single GPU, with accelerate

set 

```bash
accelerate launch --num_processes 1 --config_file ./config/accelerate_deepspeed_zero3.yaml ./src/train.py experiment=qwen_gsm8k_uniform
```

## Downloading data

You need to download JSON files for the datasets you want to use and put them under .data/ directory.
### MATH dataset

You can download the the data here (it's just the MATH dataset in a special format and gsm8k): 
`https://drive.google.com/file/d/1kRv4X3ZDlKj9-4Rf5E5MqzqQWJooX340/view?usp=sharing`

from googledrive it's trick
```bash
cd ./data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kRv4X3ZDlKj9-4Rf5E5MqzqQWJooX340' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kRv4X3ZDlKj9-4Rf5E5MqzqQWJooX340" -O data.zip && rm -rf /tmp/cookies.txt
unzip data.zip
mv ./data/MATH_json ./MATH
rm data.zip
rm -rf ./data
```

It's also publicly available here but needs to be processed. Each question is one single JSON file...
```bash
cd ./data
wget https://www.modelscope.cn/datasets/opencompass/competition_math/resolve/master/data/MATH.zip 
unzip MATH.zip
```
