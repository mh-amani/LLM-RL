# The parity dataset.

import os
import argparse

from datasets import Dataset
import random
import pandas as pd


def generate_xor_sequence_dataset(num_samples=10, seq_length=500, bit_width=1, separator=';'):
    data = []

    for _ in range(num_samples):
        x_seq = [random.randint(0, 2**bit_width - 1) for _ in range(seq_length)]
        y_seq = [random.randint(0, 2**bit_width - 1) for _ in range(seq_length)]
        z_seq = [x ^ y for x, y in zip(x_seq, y_seq)]
        
        # Format
        x_str = " ".join(str(x) for x in x_seq)
        yz_str = " ".join(f"{y} {z}" for y, z in zip(y_seq, z_seq))
        # sample = {
        #     "input": f"{x_str} {separator} {yz_str}",
        #     "output": yz_str
        # }
        sample = {
            'prompt': x_str,
            'answer': yz_str,
        }

        data.append(sample)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/LLM-RL/data/verl-data/parity')
    parser.add_argument('--train_size', type=int, default=1024)
    parser.add_argument('--length', type=int, default=256)
    parser.add_argument('--bit_width', type=int, default=1)
    parser.add_argument('--test_size', type=int, default=512)

    args = parser.parse_args()
    
    # Create train/test datasets
    train_data = generate_xor_sequence_dataset(num_samples=args.train_size, seq_length=args.length, bit_width=args.bit_width)
    test_data = generate_xor_sequence_dataset(num_samples=args.test_size, seq_length=args.length, bit_width=args.bit_width)

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))

    data_source = f'xor/sequence_{args.length}_bitwidth_{args.bit_width}'

    # instruction_following = " Let's think step by step" # and output the final answer after ...
    instruction_following = ""
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('prompt')
            question = question_raw + instruction_following
            answer_raw = example.pop('answer')
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "answer": answer_raw,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer_raw
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir + f'/parity_length_{args.length}_bitwidth_{args.bit_width}_{args.train_size}_{args.test_size}'
    print(f"Saving to {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    