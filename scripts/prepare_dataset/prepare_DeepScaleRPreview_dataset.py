
"""
Preprocess the MATH-lighteval dataset to parquet format
"""
import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/LLM-RL/data/verl-data/DeepScaleR")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "agentica-org/DeepScaleR-Preview-Dataset"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)

    train_dataset, test_dataset = dataset['train'].train_test_split(test_size=0.07).values()

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")

            question = question + " " + instruction_following

            solution = example.pop("solution")
            answer = example.pop("answer")
            boxed_solution = f"\\boxed{{{answer}}}"
            solution = solution + " " + boxed_solution

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "answer": solution,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split, "index": idx,},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    print(f"Number of train examples: {len(train_dataset)}")
    print(f"Number of test examples: {len(test_dataset)}")
    empty_answers = dataset["train"].filter(lambda x: x["answer"] == "")
    empty_solutions = dataset["train"].filter(lambda x: x["solution"] == "")
    print(f'number of empty answers: {len(empty_answers)}')
    print(f'number of empty solutions: {len(empty_solutions)}')

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)