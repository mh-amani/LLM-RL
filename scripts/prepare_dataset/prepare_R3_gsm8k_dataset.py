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


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/LLM-RL/data/verl-data/gsm8k_r3')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'openai/gsm8k'

    dataset = datasets.load_dataset(data_source, 'main')

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    instruction_following = "Let's think step by step and output the final answer after \"####\"."

    r3_train_dataset = []
    idxx = 0
    for example in train_dataset:
        question_raw = example.pop('question')
        question = question_raw + ' ' + instruction_following

        answer_raw = example.pop('answer')
        solution = extract_solution(answer_raw)
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
            question_raw = example.pop('question')

            question = question_raw + ' ' + instruction_following

            answer_raw = example.pop('answer')
            solution = extract_solution(answer_raw)
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

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    r3_train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
