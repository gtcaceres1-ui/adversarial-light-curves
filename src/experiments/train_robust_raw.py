"""Train the canonical OGLE L5 model via the ablation driver."""
from __future__ import annotations

import argparse

from src.experiments import train_robust_raw_ablation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    forwarded = [
        "train_robust_raw_ablation",
        "--levels", "5",
        "--epochs", str(args.epochs),
    ]
    if args.force:
        forwarded.append("--force")
    import sys
    old = sys.argv
    try:
        sys.argv = forwarded
        train_robust_raw_ablation.main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
