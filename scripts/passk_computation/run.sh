

# python passk_compute.py \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --dataset_path LLM-RL/data/verl-data/modgsm8k \
#     --reward_fn None \
#     --n_max 20 \
#     --temperature 0.8 \
#     --top_p 0.95 \
#     --max_tokens 1024



# python passk_compute.py \
#     --model masani/SFT_cumulative_parity_length_16_bitwidth_1_2048_512_Qwen2-1.5B_epoch_8_global_step_64 \
#     --dataset_path LLM-RL/data/verl-data/cumulative_parity_length_16_bitwidth_1_1024_512 \
#     --reward_fn src.utils.rewards.CumulativeParityReward \
#     --n_max 20 \
#     --temperature 0.8 \
#     --top_p 0.95 \
#     --max_tokens 1024
#     --save_dir LLM-RL/scripts/passk_computation/outputs



# #  to plot
# python plot_passk.py \
#   --paths outputs/gsm8k--base--eval.jsonl outputs/gsm8k--rl--eval.jsonl outputs/gsm8k--curriculum--eval.jsonl \
#   --labels base RL curriculum \
#   --kmax 20 \
#   --out passk_gsm8k_llama3_1_8b_comparison.png



# #### TEMP

# python scripts/passk_computation/passk_computation.py  \
#     --model meta-llama/Llama-3.2-1B \
#     --dataset_path data/verl-data/gsm8k/test.parquet \
#     --n_max 20 \
#     --temperature 0.8 \
#     --top_p 0.95 \
#     --max_tokens 1024 \
#     --save_dir scripts/passk_computation/outputs


VLLM_WORKER_MULTIPROC_METHOD=spawn python scripts/passk_computation/passk_computation.py      --model meta-llama/Llama-3.2-1B      --dataset_path data/verl-data/gsm8k/test.parquet     --n_max 256     --temperature 0.8     --top_p 0.95     --max_tokens 1024     --save_dir  ouputs_1B_base > llama-1B-gsm8k-base.txt;
VLLM_WORKER_MULTIPROC_METHOD=spawn python scripts/passk_computation/passk_computation.py      --model tmp/     --dataset_path data/verl-data/gsm8k/test.parquet     --n_max 256     --temperature 0.8     --top_p 0.95     --max_tokens 1024     --save_dir  ouputs_1B_RL > llama-1B-gsm8k-RL.txt;
VLLM_WORKER_MULTIPROC_METHOD=spawn python scripts/passk_computation/passk_computation.py      --model tmpadapt/     --dataset_path data/verl-data/gsm8k/test.parquet     --n_max 256     --temperature 0.8     --top_p 0.95     --max_tokens 1024     --save_dir  ouputs_1B_ours > llama-1B-gsm8k-ours.txt;