"""Preprocessing utilities for the credit scoring pipeline."""

from pathlib import Path
import pandas as pd


def load_raw_data(data_path: Path) -> pd.DataFrame:
    """Load raw financial dataset from a CSV path."""
    return pd.read_csv(data_path)


def save_processed_data(data: pd.DataFrame, output_path: Path) -> None:
    """Save processed data to disk in a reproducible format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
