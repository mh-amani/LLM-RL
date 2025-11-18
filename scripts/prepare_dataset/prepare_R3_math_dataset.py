
import argparse
import os
import math

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str: str) -> str:
    """Extract the final boxed answer from a LaTeX solution."""
    return remove_boxed(last_boxed_only_string(solution_str))


def build_r3_split(
    hf_split,
    split_name: str,
    data_source: str,
    instruction_following: str,
    k: int,
):
    """
    Turn a HF split (train/test) into an R3-style list of examples.

    For each original example i, we produce:
      - 1 "base" example (no partial fields)
      - k "partial" examples with increasing prefixes of the solution
    """
    r3_examples = []
    idx = 0

    for original_idx, example in enumerate(hf_split):
        # Original fields
        question_raw = example["problem"]
        full_answer = example["solution"]

        # Chat-style question with instruction
        question = question_raw + " " + instruction_following

        # Final boxed answer only
        ground_truth = extract_solution(full_answer)

        # Tokenize the full CoT answer at the word level
        words = full_answer.split()
        n_words = len(words)

        # --- 1) Base example (no partial info) ---
        r3_examples.append(
            {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                # Full CoT solution
                "answer": full_answer,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    # Only the final boxed answer as ground truth
                    "ground_truth": ground_truth,
                },
                "extra_info": {
                    "split": split_name,
                    "index": idx,
                    "orig_index": original_idx,
                    "question": question_raw,
                },
            }
        )
        idx += 1

        # --- 2) k partial-rationale curriculum variants ---
        # j = 1..k: reveal j/k of the words of the solution
        if n_words > 0:
            for j in range(1, k + 1):
                cutoff = math.ceil(j * n_words / k)
                partial_words = words[:cutoff]
                completion_words = words[cutoff:]

                partial_answer = " ".join(partial_words)
                completion = " ".join(completion_words)
                portion = cutoff / n_words

                r3_examples.append(
                    {
                        "data_source": data_source,
                        "prompt": [
                            {
                                "role": "user",
                                "content": question,
                            }
                        ],
                        # Still keep the *full* solution as the model target
                        "answer": full_answer,
                        "ability": "math",
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": ground_truth,
                        },
                        "extra_info": {
                            "split": split_name,
                            "index": idx,
                            "orig_index": original_idx,
                            "question": question_raw,
                            "partial_answer": partial_answer,
                            "completion": completion,
                            "portion": portion,  # fraction of words revealed
                        },
                    }
                )
                idx += 1

    return datasets.Dataset.from_list(r3_examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        default="~/LLM-RL/data/verl-data/math-r3",
        help="Local directory to save parquet files.",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory to copy the processed data.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of equal word-level chunks for partial solutions.",
    )
    args = parser.parse_args()

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir
    k = args.k

    # Data source (mirror for lighteval/MATH)
    data_source = "DigitalLearningGmbH/MATH-lighteval"

    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Same instruction as before (or tweak as you like)
    instruction_following = (
        "Let's think step by step and output the final answer within \\boxed{}."
    )

    print(f"Building R3-style train split with k={k}...", flush=True)
    r3_train_dataset = build_r3_split(
        hf_split=train_dataset,
        split_name="train",
        data_source=data_source,
        instruction_following=instruction_following,
        k=k,
    )

    print(f"Building R3-style test split with k={k}...", flush=True)
    r3_test_dataset = build_r3_split(
        hf_split=test_dataset,
        split_name="test",
        data_source=data_source,
        instruction_following=instruction_following,
        k=k,
    )

    os.makedirs(local_dir, exist_ok=True)
    train_path = os.path.join(local_dir, f"train_k{k}.parquet")
    test_path = os.path.join(local_dir, f"test_k{k}.parquet")

    print(f"Saving train to {train_path}")
    r3_train_dataset.to_parquet(train_path)

    print(f"Saving test to {test_path}")
    r3_test_dataset.to_parquet(test_path)

    if hdfs_dir is not None:
        print(f"Copying to HDFS: {hdfs_dir}")
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print("Copy to HDFS completed.")
