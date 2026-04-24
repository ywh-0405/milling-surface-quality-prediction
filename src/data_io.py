"""Data preparation utilities for synthetic and future real milling datasets."""

from pathlib import Path

import pandas as pd

from src.schemas import (
    BASE_COLUMNS,
    DERIVED_COLUMNS,
    REQUIRED_INPUT_COLUMNS,
    TARGET_COLUMNS,
    derive_process_columns,
)


def _validate_columns(df, required_columns):
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_summary_table(input_csv, output_dir):
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    _validate_columns(df, BASE_COLUMNS + REQUIRED_INPUT_COLUMNS + TARGET_COLUMNS)

    derived = df.apply(lambda row: pd.Series(derive_process_columns(row)), axis=1)
    features = pd.concat(
        [df[BASE_COLUMNS + REQUIRED_INPUT_COLUMNS].copy(), derived[DERIVED_COLUMNS]], axis=1
    )
    features = features.loc[:, ~features.columns.duplicated()]
    targets = df[["sample_id"] + TARGET_COLUMNS].copy()

    features.to_csv(output_dir / "features.csv", index=False)
    targets.to_csv(output_dir / "targets.csv", index=False)
    return {"features": features, "targets": targets}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare a summary CSV into processed features.csv and targets.csv files."
    )
    parser.add_argument("input", help="Path to the summary CSV file")
    parser.add_argument("output", help="Directory where processed CSV files will be written")
    args = parser.parse_args()
    prepare_summary_table(args.input, args.output)
    print(f"Processed files written to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

