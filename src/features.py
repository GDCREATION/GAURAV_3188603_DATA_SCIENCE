"""Feature engineering transformers for churn prediction."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import SERVICE_COLS


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add engineered features before imputation/encoding.

    Features
    --------
    AvgMonthlyCharge
        ``TotalCharges / (tenure + 1)`` — spending intensity; ``+1`` avoids
        division by zero when tenure is 0.
    HasMultipleServices
        Count of add-on services equal to ``"Yes"`` — proxy for product
        engagement / switching cost.
    """

    def __init__(self, service_cols: list[str] | None = None):
        self.service_cols = service_cols

    def fit(self, X, y=None):
        # Stateless transformer — fit is a no-op for sklearn Pipeline API
        return self

    def transform(self, X):
        X = X.copy()
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer expects a pandas DataFrame")

        tenure = pd.to_numeric(X["tenure"], errors="coerce").fillna(0)
        total = pd.to_numeric(X["TotalCharges"], errors="coerce")
        X["AvgMonthlyCharge"] = total / (tenure + 1)

        cols = self.service_cols if self.service_cols is not None else SERVICE_COLS
        present = [c for c in cols if c in X.columns]
        X["HasMultipleServices"] = (
            X[present].eq("Yes").sum(axis=1).astype(int) if present else 0
        )
        return X


def engineered_numeric_columns(base_numeric: list[str]) -> list[str]:
    """Extend numeric column list with engineered numeric features."""
    extra = ["AvgMonthlyCharge", "HasMultipleServices"]
    return list(base_numeric) + [c for c in extra if c not in base_numeric]
