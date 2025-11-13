#!/usr/bin/env python3
"""Utility to preview the first few rows of a Parquet dataset."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

import pandas as pd


def display_head(parquet_path: pathlib.Path, num_rows: int = 5) -> pd.DataFrame:
    """Load a Parquet file and return the first `num_rows` rows as a DataFrame."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if df.empty:
        raise ValueError(f"The Parquet file at {parquet_path} is empty.")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 0)
    head = df.head(num_rows)
    print(head.to_string(index=False))
    return head


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview the first N rows of a Parquet file."
    )
    parser.add_argument(
        "parquet_path",
        type=pathlib.Path,
        help="Absolute or relative path to the Parquet file.",
    )
    parser.add_argument(
        "-n",
        "--num-rows",
        type=int,
        default=5,
        help="Number of rows to display (default: 5).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    display_head(args.parquet_path, args.num_rows)


# usage: python /dlabscratch1/amani/prod/LLM-RL/scripts/ispect_data.py /path/to/file.parquet -n 10

if __name__ == "__main__":
    main()