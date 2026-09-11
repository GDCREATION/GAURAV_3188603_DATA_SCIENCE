"""Train and compare Decision Tree (and optional) churn models."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import CATEGORICAL_COLS, NUMERIC_COLS, RANDOM_STATE
from src.features import FeatureEngineer, engineered_numeric_columns
from src.preprocessing import build_preprocessor


def build_model_pipeline(
    clf,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> Pipeline:
    """Full Pipeline: feature engineering → preprocess → classifier."""
    base_num = list(numeric_cols) if numeric_cols is not None else list(NUMERIC_COLS)
    cat = list(categorical_cols) if categorical_cols is not None else list(CATEGORICAL_COLS)
    num = engineered_numeric_columns(base_num)

    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocess", build_preprocessor(numeric_cols=num, categorical_cols=cat)),
            ("model", clf),
        ]
    )


def decision_tree_configs(random_state: int = RANDOM_STATE) -> dict[str, DecisionTreeClassifier]:
    """Return at least two Decision Tree configurations for comparison."""
    return {
        "DT_Baseline": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=10,
            random_state=random_state,
        ),
        "DT_Deep": DecisionTreeClassifier(
            max_depth=None,
            min_samples_split=2,
            random_state=random_state,
        ),
    }


def evaluate_binary(y_true, y_pred) -> dict[str, float]:
    """Accuracy, precision, recall, F1 for the positive (churn) class."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def train_and_compare(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    configs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Pipeline], str]:
    """Train each config, score on test, select best by F1 then recall.

    Returns
    -------
    comparison_df, fitted_pipelines, best_name
    """
    if configs is None:
        configs = decision_tree_configs()

    rows = []
    pipes: dict[str, Pipeline] = {}

    for name, clf in configs.items():
        pipe = build_model_pipeline(clf)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = evaluate_binary(y_test, y_pred)
        metrics["model"] = name
        rows.append(metrics)
        pipes[name] = pipe

    comparison = pd.DataFrame(rows).set_index("model")[
        ["accuracy", "precision", "recall", "f1"]
    ]
    # Prefer higher F1; break ties with recall (retention priority)
    ranked = comparison.sort_values(by=["f1", "recall"], ascending=False)
    best_name = str(ranked.index[0])
    return comparison, pipes, best_name
