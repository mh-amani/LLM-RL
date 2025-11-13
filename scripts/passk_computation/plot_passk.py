import json
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def load_results(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def aggregate_passk(data, kmax=None):
    acc = defaultdict(list)
    for ex in data:
        for k_str, v in ex["pass@k"].items():
            k = int(k_str)
            if kmax is None or k <= kmax:
                acc[k].append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", required=True,
                        help="Paths to JSONL files with eval results")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Optional custom labels for each file (same order as --paths)")
    parser.add_argument("--out", default=None,
                        help="Optional path to save the plot instead of showing it")
    parser.add_argument("--kmax", type=int, default=None,
                        help="Maximum value of k to show on the x-axis")
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.paths):
        raise ValueError("Number of --labels must match number of --paths")

    for i, path in enumerate(args.paths):
        data = load_results(path)
        passk = aggregate_passk(data, kmax=args.kmax)
        ks, values = zip(*sorted(passk.items()))
        label = args.labels[i] if args.labels else Path(path).stem.split("--")[1]
        plt.plot(ks, values, label=label)

    plt.xlabel("k")
    plt.ylabel("pass@k")
    plt.title("Pass@k Comparison")
    plt.legend()
    plt.grid(True)

    if args.out:
        plt.savefig(args.out, bbox_inches="tight")
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
