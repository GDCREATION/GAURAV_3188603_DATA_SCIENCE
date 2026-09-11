"""Load saved pipeline and run churn predictions for new customers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import FEATURE_COLS, MODEL_PATH

_pipeline = None


def load_model(path: Path | None = None):
    """Load (and cache) the saved sklearn Pipeline."""
    global _pipeline
    model_path = Path(path) if path is not None else MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train and save it (Story 7) first."
        )
    _pipeline = joblib.load(model_path)
    return _pipeline


def get_pipeline():
    """Return cached pipeline, loading from disk if needed."""
    global _pipeline
    if _pipeline is None:
        return load_model()
    return _pipeline


def customer_to_dataframe(customer: dict[str, Any]) -> pd.DataFrame:
    """Convert a single customer dict to a one-row DataFrame."""
    row = {col: customer.get(col) for col in FEATURE_COLS}
    # Allow optional engineered columns if caller already computed them —
    # FeatureEngineer will recompute from raw fields.
    return pd.DataFrame([row], columns=FEATURE_COLS)


def predict_churn(
    customer: dict[str, Any],
    pipeline=None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Predict churn label and probability for one customer.

    Returns
    -------
    dict with keys ``prediction`` (\"Yes\"/\"No\") and ``churn_probability`` (float).
    """
    pipe = pipeline if pipeline is not None else get_pipeline()
    X = customer_to_dataframe(customer)

    if hasattr(pipe, "predict_proba"):
        proba = float(pipe.predict_proba(X)[0][1])
    else:
        # Fallback if model has no probabilities
        pred = int(pipe.predict(X)[0])
        proba = float(pred)

    label = "Yes" if proba >= threshold else "No"
    return {"prediction": label, "churn_probability": round(proba, 4)}
