"""CLI entry point for the lite training mode."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.models_lite import train_lite_models


def main():
    parser = argparse.ArgumentParser(description="Train the lightweight milling prediction models.")
    parser.add_argument("--features", required=True, help="Path to features.csv")
    parser.add_argument("--targets", required=True, help="Path to targets.csv")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    targets = pd.read_csv(args.targets)
    result = train_lite_models(features, targets)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, ensure_ascii=False, indent=2)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

