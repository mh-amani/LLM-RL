"""
Preprocess the OpenR1-Math-220k dataset into VERL-style parquet.

- Uses the 'answer' field as verifiable ground truth.
- Uses the main CoT 'solution' when available, otherwise falls back
  to a verified generation.
- Filters to examples whose solution length is between [min_words, max_words].
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


def pick_solution(example):
    """
    Choose a chain-of-thought solution text for an example.

    Priority:
    1. `solution` field if non-empty.
    2. First generation in `generations` whose `correctness_math_verify` is True.
    3. Fallback to the first generation in `generations` if any.
    4. Otherwise return empty string.
    """
    sol = example.get("solution", "") or ""
    sol = sol.strip()
    if sol:
        return sol

    gens = example.get("generations") or []
    math_flags = example.get("correctness_math_verify") or []
    print(f"Found {len(gens)} generations.")

    # Prefer a generation that passed math_verify
    for g, flg in zip(gens, math_flags):
        if flg and isinstance(g, str):
            gs = g.strip()
            if gs:
                return gs

    # Fallback: first non-empty generation
    for g in gens:
        if isinstance(g, str) and g.strip():
            return g.strip()

    return ""


def make_filter_fn(min_words, max_words):
    """
    Keep only examples whose chosen solution has word count in [min_words, max_words].
    """

    # def _filter_fn(example):
    #     if not sol:
    #         return False
    #     n_words = len(sol.split())
    #     return (n_words >= min_words) and (n_words <= max_words)
    
    def _filter_fn(example):
        answer = example['answer']
        if not answer:
            return False
        # answer should be an integer
        try :
            answer = int(answer.strip())
        except:
            return False
        
        sol = len(pick_solution(example).split())
        return (sol >= min_words) and (sol <= max_words)

    return _filter_fn


def make_map_fn(
    data_source: str,
    split_name: str,
    instruction_following: str,
):
    """
    Map raw OpenR1 examples to VERL-style schema.
    """

    def _map_fn(example, idx):
        problem = example["problem"].strip()
        solution_text = pick_solution(example)
        gt_answer = (example.get("answer") or "").strip()

        question = problem + " " + instruction_following

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            # Full chain-of-thought solution we train on
            "answer": solution_text,
            "ability": "math",
            # Reward model uses the canonical short answer from the dataset
            "reward_model": {
                "style": "rule",
                "ground_truth": gt_answer,
            },
            "extra_info": {
                "split": split_name,
                "index": idx,
                "problem": problem,
                "problem_type": example.get("problem_type"),
                "question_type": example.get("question_type"),
                "source": example.get("source"),
            },
        }
        return data

    return _map_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dir",
        default="./data/verl-data/openr1-math-220k",
        help="Local directory to save the parquet file.",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory to copy the processed data.",
    )
    parser.add_argument(
        "--config_name",
        default="default",
        choices=["default", "extended", "all"],
        help="OpenR1-Math-220k config to use.",
    )
    parser.add_argument(
        "--min_words",
        type=int,
        default=100,
        help="Minimum number of words in the solution (inclusive).",
    )
    parser.add_argument(
        "--max_words",
        type=int,
        default=1000,
        help="Maximum number of words in the solution (inclusive).",
    )

    args = parser.parse_args()

    import debugpy
    debugpy.listen(("0.0.0.0", 5678))  # Or another port
    print("Waiting for debugger to attach...")
    debugpy.wait_for_client()

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir
    config_name = args.config_name
    min_words = args.min_words
    max_words = args.max_words

    data_source = f"open-r1/OpenR1-Math-220k-{config_name}"

    print(f"Loading {data_source} from Hugging Face...", flush=True)
    # This returns a DatasetDict with a single 'train' split
    ds_dict = datasets.load_dataset("open-r1/OpenR1-Math-220k", config_name)
    raw_train = ds_dict["train"]

    print(
        f"Filtering examples with solution word count in [{min_words}, {max_words}]...",
        flush=True,
    )
    filtered = raw_train.filter(
        make_filter_fn(min_words=min_words, max_words=max_words)
    )

    print(f"Kept {len(filtered)} examples after filtering.", flush=True)

    # Instruction consistent with how OpenR1 was generated
    instruction_following = (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )

    print("Mapping to VERL-style schema...", flush=True)
    processed = filtered.map(
        function=make_map_fn(
            data_source='DigitalLearningGmbH/MATH-lighteval',
            split_name="train",
            instruction_following=instruction_following,
        ),
        with_indices=True,
    )

    split_dict = processed.train_test_split(
        test_size=0.1,
        seed=42,
    )
    train_split = split_dict["train"]
    test_split = split_dict["test"]

    os.makedirs(local_dir, exist_ok=True)

    train_path = os.path.join(
        local_dir,
        f"train.parquet",
    )
    test_path = os.path.join(
        local_dir,
        f"test.parquet",
    )

    print(f"Saving train to {train_path}", flush=True)
    train_split.to_parquet(train_path)

    print(f"Saving test to {test_path}", flush=True)
    test_split.to_parquet(test_path)

