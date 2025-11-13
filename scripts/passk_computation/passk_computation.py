import re
import os
import datasets
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
import math

from verl.utils.hdfs_io import copy, makedirs
import argparse
import hydra
# Import your reward function here
from verl.utils.reward_score import _default_compute_score
from transformers import AutoTokenizer

import json
import os
from pathlib import Path

def compute_passk(n, c, k):
    """
    Analytical pass@k computation as in OpenAI HumanEval.
    n: number of samples
    c: number of correct samples
    k: pass@k
    """
    if c == 0 or n < k:
        return 0.0
    if c == n:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))

def main(args):
    # Load dataset
    dataset = datasets.load_dataset("parquet", data_files=args.dataset_path)['train'] # .select(range(10))

    # load reward function
    if args.reward_fn is None:
        compute_reward = _default_compute_score
    else:
        compute_reward = hydra.utils.get_method(args.reward_fn)

    # Load model
    llm = LLM(model=args.model, dtype="auto", gpu_memory_utilization=0.9, tensor_parallel_size=8)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n_max,
    )
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.chat_template is None:
            print("No chat template found. Setting a custom one. it should take care of add_generation_prompt")
            tokenizer.chat_template = """{% for message in messages -%}
                                            {{ message['role'] }}: {{ message['content'] }}
                                            {% endfor -%}{% if add_generation_prompt -%}
                                            assistant: {% endif %}"""


    prompts = [tokenizer.apply_chat_template(example["prompt"], tokenize=False) for example in dataset]
    outputs = llm.generate(prompts, sampling_params)  # returns list of RequestOutput

    all_passk = {k: [] for k in range(1, args.n_max + 1)}
    for example, output in zip(dataset, outputs):
        completions = [o.text for o in output.outputs]
        rewards = [compute_reward(
            data_source=example["data_source"],
            solution_str=completion,
            ground_truth=example["reward_model"]["ground_truth"]
        ) for completion in completions]

        successes = [r > 0 for r in rewards]  # or whatever threshold means "correct"

        n = len(successes)
        c = sum(successes)
        for k in range(1, min(args.n_max, n) + 1):
            passk = compute_passk(n, c, k)
            all_passk[k].append(passk)

    # Aggregate and print results
    print("pass@k results:")
    for k in range(1, args.n_max + 1):
        avg_passk = np.mean(all_passk[k])
        print(f"pass@{k}: {avg_passk:.4f}")
    
    
    # Save all individual results
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"{Path(args.dataset_path.split('/')[-2])}--{args.model.split('/')[-1]}--eval.jsonl"

    with open(save_path, "w") as f:
        for example, output in zip(dataset, outputs):
            completions = [o.text for o in output.outputs]
            rewards = [compute_reward(
                data_source=example["data_source"],
                solution_str=completion,
                ground_truth=example["reward_model"]["ground_truth"]
            ) for completion in completions]

            successes = [r > 0 for r in rewards]
            n = len(successes)
            c = sum(successes)
            passk_dict = {
                str(k): compute_passk(n, c, k)
                for k in range(1, min(args.n_max, n) + 1)
            }

            f.write(json.dumps({
                "id": example['extra_info'].get('index', None),
                "prompt": example["prompt"],
                "completions": completions,
                "rewards": rewards,
                "successes": successes,
                "pass@k": passk_dict,
                "ground_truth": example["reward_model"]["ground_truth"]
            }) + "\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument("--dataset_path", default='LLM-RL/data/verl-data/gsm8k/test.parquet', type=str, help="Path to the dataset parquet file")
    parser.add_argument("--reward_fn", default=None, type=str, help="task reward function if it's not math and gsm8k")
    parser.add_argument("--n_max", type=int, default=20, help="Maximum number of generations per example")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--save_dir", type=str, default='LLM-RL/scripts/passk_computation/outputs')
    args = parser.parse_args()

    main(args)
