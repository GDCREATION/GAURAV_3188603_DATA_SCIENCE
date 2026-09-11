"""Preprocessing transformers and column splits (fit on train only)."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import CATEGORICAL_COLS, NUMERIC_COLS


def get_feature_columns(
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return numeric and categorical column lists (defaults from config)."""
    num = list(numeric_cols) if numeric_cols is not None else list(NUMERIC_COLS)
    cat = list(categorical_cols) if categorical_cols is not None else list(CATEGORICAL_COLS)
    return num, cat


def build_preprocessor(
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric + categorical features.

    Numeric: median imputation (handles TotalCharges NaNs after coercion).
    Categorical: most-frequent imputation + one-hot encoding with
    ``handle_unknown='ignore'`` so unseen categories at inference do not crash.

    Important: fit this transformer only on the training set (or as part of a
    Pipeline fitted on training data) to avoid data leakage.
    """
    num_cols, cat_cols = get_feature_columns(numeric_cols, categorical_cols)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )
