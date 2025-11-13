
# The parity dataset.

import os
import argparse

import datasets
import random
import pandas as pd


def generate_xor_sequence_dataset(num_samples=10, seq_length=500, bit_width=1, separator=';'):
    data = []

    for _ in range(num_samples):
        x_seq = [random.randint(0, 2**bit_width - 1) for _ in range(seq_length)]
        y_seq = [random.randint(0, 2**bit_width - 1) for _ in range(seq_length)]

        z_seq = []
        for i in range(len(x_seq)):
            if i == 0:
                z_seq.append(x_seq[i] ^ y_seq[i])
            else:
                z_seq.append(z_seq[i-1] ^ y_seq[i] ^ x_seq[i])
        
        # Format
        x_str = " ".join(str(x) for x in x_seq)
        yz_str = " ".join(f"{y} {z}" for y, z in zip(y_seq, z_seq))
        
        sample = {
            'prompt': x_str,
            'answer': yz_str,
        }
        data.append(sample)

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/LLM-RL/data/verl-data/')
    parser.add_argument('--train_size', type=int, default=1024)
    parser.add_argument('--length', type=int, default=16)
    parser.add_argument('--bit_width', type=int, default=1)
    parser.add_argument('--test_size', type=int, default=512)

    args = parser.parse_args()
    
    # Create train/test datasets
    train_data = generate_xor_sequence_dataset(num_samples=args.train_size, seq_length=args.length, bit_width=args.bit_width)
    test_data = generate_xor_sequence_dataset(num_samples=args.test_size, seq_length=args.length, bit_width=args.bit_width)

    # Convert to Hugging Face Dataset
    train_dataset = datasets.Dataset.from_pandas(pd.DataFrame(train_data))
    test_dataset = datasets.Dataset.from_pandas(pd.DataFrame(test_data))

    data_source = f'xor/r3_sequence_{args.length}_bitwidth_{args.bit_width}'

    # instruction_following = " Let's think step by step" # and output the final answer after ...
    instruction_following = ""
    # add a row to each data item that represents a unique id
    
    r3_train_dataset = []
    idxx = 0
    for example in train_dataset:
        question_raw = example.pop('prompt')
        question = question_raw + ' ' + instruction_following

        answer_raw = example.pop('answer')
        solution = answer_raw
        answer_steps = answer_raw.split('\n')

        r3_train_dataset.append({
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "answer": answer_raw,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': 'train',
                    'index': idxx,
                    'question': question_raw,
                }
            })
        idxx += 1

        for j in range(0, len(answer_steps)):  
            cut_answer = '\n'.join(answer_steps[:j])
            completion = '\n'.join(answer_steps[j:])
            portion = j / len(answer_steps)
            r3_train_dataset.append({
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "answer": answer_raw,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': 'train',
                    'index': idxx,
                    'question': question_raw,
                    'partial_answer': cut_answer,
                    'completion': completion,
                    'portion': portion,
                }
            })
            idxx += 1


    r3_train_dataset = datasets.Dataset.from_list(r3_train_dataset)

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('prompt')

            question = question_raw + ' ' + instruction_following

            answer_raw = example.pop('answer')
            solution = answer_raw
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
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    "question": question_raw,
                }
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)


    local_dir = args.local_dir + f'/cumulative_parity_r3_length_{args.length}_bitwidth_{args.bit_width}_{args.train_size}_{args.test_size}'
    print(f"Saving to {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    r3_train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
