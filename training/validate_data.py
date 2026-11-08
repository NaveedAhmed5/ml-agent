"""Data validation and preparation stage for DVC pipeline.

Verifies integrity of data/dataset.csv (schema, row counts, missing values)
and outputs data validation metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd


def validate_data(input_csv: str = "data/dataset.csv", output_report: str = "data/validation_report.json") -> dict:
    """Validate tabular dataset schema and data quality."""
    csv_path = Path(input_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    # Basic quality checks
    missing_count = int(df.isnull().sum().sum())
    num_rows = len(df)
    num_cols = len(df.columns)
    has_target = "target" in df.columns

    if num_rows < 10:
        raise ValueError(f"Dataset too small: only {num_rows} rows found.")
    if not has_target:
        raise ValueError("Missing required column 'target'.")

    report = {
        "status": "VALID",
        "num_rows": num_rows,
        "num_columns": num_cols,
        "missing_values": missing_count,
        "features": [c for c in df.columns if c not in ("target", "dataset_version", "generated_at")],
        "dataset_version": str(df["dataset_version"].iloc[0]) if "dataset_version" in df.columns else "unknown",
    }

    out_path = Path(output_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[validate_data] Data validation succeeded: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DVC dataset")
    parser.add_argument("--input", type=str, default="data/dataset.csv", help="Input dataset path")
    parser.add_argument("--output", type=str, default="data/validation_report.json", help="Report output path")
    args = parser.parse_args()

    validate_data(input_csv=args.input, output_report=args.output)


if __name__ == "__main__":
    main()

