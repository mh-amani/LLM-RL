#!/usr/bin/env python3
"""
Remove duplicate prompts from a parquet dataset.

This script loads a parquet dataset, identifies duplicates based on prompt content
(specifically checking the first message's content in the chat format), and creates
a new dataset with duplicates removed.
"""

import argparse
import pathlib
from typing import Optional

import datasets


def remove_duplicates(
    input_path: pathlib.Path,
    output_path: Optional[pathlib.Path] = None,
    prompt_key: str = "prompt",
) -> None:
    """
    Remove duplicate prompts from a parquet dataset.

    Args:
        input_path: Path to the input parquet file
        output_path: Path to save the deduplicated parquet file (optional)
        prompt_key: Column name for prompts (default: "prompt")
    """
    # Load the dataset
    print(f"Loading dataset from: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {input_path}")

    dataset = datasets.load_dataset('parquet', data_files=str(input_path), split='train')
    original_size = len(dataset)
    print(f"Original dataset size: {original_size}")
    print(f"Features: {list(dataset.features.keys())}\n")

    # Track seen prompts and indices to keep
    seen_prompts = set()
    indices_to_keep = []
    duplicate_count = 0

    print("Scanning for duplicates...")
    for idx, row in enumerate(dataset):
        prompt = row[prompt_key]

        # Extract the actual prompt content
        # Check if prompt is a list (chat format)
        if isinstance(prompt, list) and len(prompt) > 0:
            # Get the first message's content
            if isinstance(prompt[0], dict) and 'content' in prompt[0]:
                prompt_content = prompt[0]['content']
            else:
                # Fallback: convert to string
                prompt_content = str(prompt)
        else:
            # If it's already a string or other format
            prompt_content = str(prompt)

        # Check if we've seen this prompt before
        if prompt_content not in seen_prompts:
            seen_prompts.add(prompt_content)
            indices_to_keep.append(idx)
        else:
            duplicate_count += 1

        # Progress update
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{original_size} rows... (found {duplicate_count} duplicates so far)")

    # Create deduplicated dataset
    print(f"\nRemoving {duplicate_count} duplicate entries...")
    deduplicated_dataset = dataset.select(indices_to_keep)
    new_size = len(deduplicated_dataset)

    print(f"\n{'='*80}")
    print("DEDUPLICATION SUMMARY")
    print(f"{'='*80}")
    print(f"Original dataset size:      {original_size:,}")
    print(f"Deduplicated dataset size:  {new_size:,}")
    print(f"Duplicates removed:         {duplicate_count:,}")
    print(f"Reduction:                  {(duplicate_count / original_size * 100):.2f}%")
    print(f"{'='*80}\n")

    # Save the deduplicated dataset
    if output_path:
        print(f"Saving deduplicated dataset to: {output_path}")

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as parquet
        deduplicated_dataset.to_parquet(str(output_path))
        print(f"✓ Saved successfully!")
    else:
        # Generate default output path
        default_output = input_path.parent / f"{input_path.stem}_dedup{input_path.suffix}"
        print(f"Saving deduplicated dataset to: {default_output}")
        deduplicated_dataset.to_parquet(str(default_output))
        print(f"✓ Saved successfully!")


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove duplicate prompts from a parquet dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Remove duplicates from GSM8K dataset (creates train_dedup.parquet)
  python scripts/remove_duplicate_prompts.py data/verl-data/gsm8k/train.parquet

  # Specify custom output path
  python scripts/remove_duplicate_prompts.py data/verl-data/gsm8k/train.parquet \\
      --output data/verl-data/gsm8k/train_clean.parquet

  # Use custom prompt column name
  python scripts/remove_duplicate_prompts.py dataset.parquet \\
      --prompt-key "question" --output cleaned.parquet
        """
    )

    parser.add_argument(
        "input_path",
        type=pathlib.Path,
        help="Path to the input parquet file",
    )

    parser.add_argument(
        "-o", "--output",
        type=pathlib.Path,
        default=None,
        help="Path to save the deduplicated parquet file (default: <input>_dedup.parquet)",
    )

    parser.add_argument(
        "--prompt-key",
        type=str,
        default="prompt",
        help="Column name for prompts (default: prompt)",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    remove_duplicates(
        input_path=args.input_path,
        output_path=args.output,
        prompt_key=args.prompt_key,
    )


if __name__ == "__main__":
    main()
