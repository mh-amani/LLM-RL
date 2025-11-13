import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def extract_solution(solution_str):
    match = re.search(r"#### (-?[0-9\.,]+)", solution_str)
    assert match is not None
    final_solution = match.group(1).replace(',', '')
    try:
        return str(int(final_solution))  # final answer mod 7
    except ValueError:
        return final_solution  # fallback if malformed


def mod7_transform_text(text):
    # Replace integers with their mod-7 equivalents
    def repl(m):
        num_str = m.group()
        try:
            num = int(num_str.replace(',', ''))
            return str(num % 7)
        except ValueError:
            return num_str

    # Match numbers that are at least 1 digit, optional comma
    return re.sub(r'\b\d{1,6}\b', repl, text)


def to_base_7(n):
    if n == 0:
        return "0"
    digits = []
    neg = n < 0
    n = abs(n)
    while n:
        digits.append(str(n % 7))
        n //= 7
    if neg:
        digits.append('-')
    return ''.join(reversed(digits))


def base7_transform_text(text):
    # Replace integers with base-7 equivalents
    def repl(m):
        num_str = m.group()
        try:
            num = int(num_str.replace(',', ''))
            return to_base_7(num)
        except ValueError:
            return num_str

    return re.sub(r'\b\d{1,6}\b', repl, text)

def base7_transform_final_answer(text):
    # replace the final answer with its base-7 equivalent
    # find the final answer
    match = re.search(r"#### (-?[0-9\.,]+)", text)
    if match:
        final_answer = match.group(1)
        base7_answer = to_base_7(int(final_answer.replace(',', '')))
        end_index = match.end()
        length = len(final_answer)
        return text[:end_index-length] + base7_answer + text[end_index:]
    return text

def is_integer_only(example):
    # Match decimal numbers like 0.2, 12.5, etc.
    pattern = r'\d+\.\d+'
    return not (re.search(pattern, example['question']) or re.search(pattern, example['answer']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/LLM-RL/data/verl-data/modgsm8k')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'openai/gsm8k'

    dataset = datasets.load_dataset(data_source, 'main')

    # Filter train and test splits
    filtered_train = dataset["train"].filter(is_integer_only)
    filtered_test = dataset["test"].filter(is_integer_only)

    print(f"Original train size: {len(dataset['train'])}")
    print(f"Filtered train size: {len(filtered_train)}")

    print(f"Original test size: {len(dataset['test'])}")
    print(f"Filtered test size: {len(filtered_test)}")


    train_dataset = filtered_train
    test_dataset = filtered_test

    instruction_following = "Let's think step by step and output the final answer after \"####\" "

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop('question')
            answer_raw = example.pop('answer')

            # Apply mod7 to question and full answer rationale
            question_mod7 = base7_transform_text(question_raw)
            answer_mod7 = base7_transform_text(answer_raw)

            solution = extract_solution(answer_mod7)

            question = question_mod7 + ' ' + instruction_following

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "answer": answer_mod7,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    "question": question_mod7,
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)


