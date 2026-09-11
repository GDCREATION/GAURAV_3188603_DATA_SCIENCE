"""Load and clean the IBM Telco Customer Churn dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_PATH, ID_COL, TARGET_COL


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Load the Telco CSV from disk.

    Raises
    ------
    FileNotFoundError
        If the CSV is missing — download it into ``data/`` first.
    """
    csv_path = Path(path) if path is not None else DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Download TelcoCustomerChurn.csv into the data/ folder."
        )
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dtypes, handle blank TotalCharges, and drop duplicate customer IDs.

    Steps
    -----
    1. Copy the frame to avoid mutating the caller's object.
    2. Coerce ``TotalCharges`` to numeric (blank strings become NaN).
    3. Drop duplicate ``customerID`` rows if any (keep first).
    4. Cast ``SeniorCitizen`` to object so it is treated as categorical later.
    """
    out = df.copy()

    # Blank strings appear for brand-new customers (tenure == 0)
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")

    if ID_COL in out.columns:
        n_dupes = out.duplicated(subset=[ID_COL]).sum()
        if n_dupes:
            out = out.drop_duplicates(subset=[ID_COL], keep="first")

    # 0/1 flag — treat as category for one-hot encoding
    if "SeniorCitizen" in out.columns:
        out["SeniorCitizen"] = out["SeniorCitizen"].astype(str)

    return out


def load_and_clean(path: Path | None = None) -> pd.DataFrame:
    """Convenience: load CSV then clean."""
    return clean_data(load_raw_data(path))


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return X (features without ID) and y (Churn as Yes/No strings)."""
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found")

    y = df[TARGET_COL].copy()
    drop_cols = [TARGET_COL]
    if ID_COL in df.columns:
        drop_cols.append(ID_COL)
    X = df.drop(columns=drop_cols)
    return X, y


def encode_target(y: pd.Series) -> pd.Series:
    """Map Churn Yes/No to 1/0 for sklearn classifiers."""
    mapping = {"Yes": 1, "No": 0}
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)
    encoded = y.map(mapping)
    if encoded.isna().any():
        bad = y[encoded.isna()].unique().tolist()
        raise ValueError(f"Unexpected Churn labels (expected Yes/No): {bad}")
    return encoded.astype(int)
