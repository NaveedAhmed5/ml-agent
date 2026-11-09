"""Synthetic Classification Dataset Generator with Versioning.

Generates reproducible synthetic tabular datasets for classification modeling,
supporting v1 and v2 profiles to evaluate data drift, versioning, and DVC tracking.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification


def generate_dataset(
    version: str = "v1",
    n_samples: int | None = None,
    output_path: str = "data/dataset.csv",
    random_state: int = 42,
) -> Path:
    """Generate synthetic classification dataset with embedded version metadata.

    Args:
        version: Version marker ('v1' or 'v2').
        n_samples: Total number of records (defaults: v1=1000, v2=1250).
        output_path: Target CSV file location.
        random_state: Seed for reproducibility.

    Returns:
        Path to the generated CSV file.
    """
    version_clean = version.lower().strip()
    if version_clean == "v1":
        samples = n_samples or 1000
        shift = 0.0
        flip_y = 0.02
        version_label = "v1.0.0"
    elif version_clean == "v2":
        samples = n_samples or 1250
        shift = 0.35  # Feature distribution shift
        flip_y = 0.05
        version_label = "v2.0.0"
    else:
        samples = n_samples or 1000
        shift = 0.0
        flip_y = 0.02
        version_label = version

    # Generate synthetic features
    X, y = make_classification(
        n_samples=samples,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_clusters_per_class=2,
        flip_y=flip_y,
        random_state=random_state,
    )

    # Apply version-specific shift
    X = X + shift

    feature_cols = [f"feature_{i+1}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df["target"] = y
    df["dataset_version"] = version_label
    df["generated_at"] = datetime.now(UTC).isoformat()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    print(
        f"[generate_data] Created dataset version '{version_label}' "
        f"with {len(df)} rows and {len(feature_cols)} features at: {out_file.resolve()}"
    )
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic classification dataset")
    parser.add_argument("--version", type=str, default="v1", choices=["v1", "v2"], help="Dataset version (v1 or v2)")
    parser.add_argument("--samples", type=int, default=None, help="Number of rows")
    parser.add_argument("--output", type=str, default="data/dataset.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_dataset(
        version=args.version,
        n_samples=args.samples,
        output_path=args.output,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()

