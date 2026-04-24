"""CLI entry point placeholder for research-mode training."""

import argparse

from src.models_research import build_research_model


def main():
    parser = argparse.ArgumentParser(description="Build the research-mode model skeleton.")
    parser.add_argument("--n-features", type=int, required=True, help="Feature dimension")
    parser.add_argument("--n-freq-bins", type=int, default=8, help="Frequency-bin target count")
    args = parser.parse_args()
    model = build_research_model(args.n_features, args.n_freq_bins)
    print(model)


if __name__ == "__main__":
    main()

