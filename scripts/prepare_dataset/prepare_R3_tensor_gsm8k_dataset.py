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
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modgsm8k_dir', default='LLM-RL/data/verl-data/gsm8k-t2')
    parser.add_argument('--local_dir', default='LLM-RL/data/verl-data/gsm8k-t2_r3')

    args = parser.parse_args()

    data_source = 'openai/gsm8k'

    # parquet files called train.parquet and test.parquet should be in modgsm8k_dir
    train_dataset = datasets.load_dataset("parquet", data_files=os.path.join(args.modgsm8k_dir, 'train.parquet'))['train']
    test_dataset = datasets.load_dataset("parquet", data_files=os.path.join(args.modgsm8k_dir, 'test.parquet'))['train']

    r3_train_dataset = []
    idxx = 0

    for item in train_dataset:
        
        r3_train_dataset.append({
                "data_source": data_source,
                "prompt": item['prompt'],
                "answer": item['answer'],
                "ability": "math",
                "reward_model": item['reward_model'],
                "extra_info": item['extra_info'],
            })
        idxx += 1

        answer_raw = item['answer']
        # Split answer into steps
        answer_steps = item['answer'].split('\n')
        num_steps = len(answer_steps)

        # For each possible split, create a new datapoint
        for j in range(1, num_steps + 1):
            cut_answer = '\n'.join(answer_steps[:j])
            completion = '\n'.join(answer_steps[j:])
            portion = j / num_steps if num_steps > 0 else 0.0

            r3_train_dataset.append({
                "data_source": item['data_source'],
                "prompt": item['prompt'],
                "answer": item['answer'],
                "ability": "math",
                "reward_model": item['reward_model'],
                "extra_info": {
                    'split': 'train',
                    'index': idxx,
                    'question': item['extra_info']['question'],
                    'partial_answer': cut_answer,
                    'completion': completion,
                    'portion': portion,
                }
            })
            idxx += 1

    r3_train_dataset = datasets.Dataset.from_list(r3_train_dataset)

    local_dir = args.local_dir

    r3_train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
