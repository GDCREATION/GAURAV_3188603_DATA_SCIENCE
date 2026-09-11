"""Build notebook/churn_analysis.ipynb with all story sections."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebook" / "churn_analysis.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    md(
        """# Customer Churn Prediction — Telco

**Business problem:** Identify customers likely to churn so the retention team can engage proactively.

**Dataset:** IBM Telco Customer Churn · **Target:** `Churn` (Yes / No)

**Reproducibility:** `random_state = 42` · **Train/Test:** 70:30 (stratified)

## Table of Contents

1. Data Understanding & Preparation
2. Exploratory Data Analysis
3. Feature Engineering
4. Model Development (Decision Tree)
5. Model Evaluation
6. Model Interpretation
7. Model Saving
8. API notes (see `app.py`)
9. Bonus Improvements (optional)

> Reusable logic lives in `src/` so the FastAPI service applies the same preprocessing as this notebook."""
    ),
    md(
        """# 1. Data Understanding & Preparation

We load the Telco CSV, check structure and quality, clean types, encode categoricals inside a sklearn pipeline (fit on train only), and create a stratified 70:30 split."""
    ),
    code(
        """import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# Make project root importable when running the notebook from notebook/
ROOT = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RANDOM_STATE, TEST_SIZE, TARGET_COL, ID_COL, NUMERIC_COLS, CATEGORICAL_COLS
from src.data_loader import load_raw_data, clean_data, split_features_target, encode_target, load_and_clean
from src.preprocessing import build_preprocessor

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)
print("RANDOM_STATE =", RANDOM_STATE, "| TEST_SIZE =", TEST_SIZE)"""
    ),
    code(
        """# Load & basic structure
raw = load_raw_data()
print("Shape:", raw.shape)
print("\\nDtypes:\\n", raw.dtypes)
raw.head()"""
    ),
    code(
        """# Missing values (including blank TotalCharges before cleaning)
print("Null counts (raw):\\n", raw.isna().sum())
print("\\nTotalCharges blank/whitespace rows:", (raw["TotalCharges"].astype(str).str.strip() == "").sum())
print("Duplicate customerIDs:", raw.duplicated(subset=[ID_COL]).sum())"""
    ),
    code(
        """# Clean: TotalCharges → numeric, SeniorCitizen → categorical string, drop dupe IDs
df = clean_data(raw)
print("Shape after clean:", df.shape)
print("TotalCharges dtype:", df["TotalCharges"].dtype)
print("TotalCharges NaNs:", df["TotalCharges"].isna().sum())
print("\\nNumeric cols:", NUMERIC_COLS)
print("Categorical cols:", CATEGORICAL_COLS)"""
    ),
    code(
        """# Target-variable analysis
churn_counts = df[TARGET_COL].value_counts()
churn_pct = df[TARGET_COL].value_counts(normalize=True) * 100
print(churn_counts)
print("\\nChurn rate (%):\\n", churn_pct.round(2))
print("\\nObservation: classes are imbalanced (~26% churn). We use a stratified split.")"""
    ),
    code(
        """# Features / target + stratified 70:30 split (BEFORE fitting any encoder/imputer)
X, y_raw = split_features_target(df)
y = encode_target(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train churn rate:", y_train.mean().round(4), "| Test churn rate:", y_test.mean().round(4))

# Preprocessor is defined here but fitted later inside the modeling Pipeline (no leakage)
preprocessor = build_preprocessor()
preprocessor"""
    ),
    md(
        """### Preprocessing decisions (Story 1)

- **`TotalCharges` → numeric:** stored as strings with blanks for tenure-0 customers; coerce + median impute inside the pipeline.
- **Drop `customerID`:** identifier, not predictive; would leak unique IDs.
- **Fit pipeline on train only:** prevents test information from entering medians / category levels.
- **Stratified 70:30 split, `random_state=42`:** keeps churn rate similar in train/test and meets assignment reproducibility rules."""
    ),
    md(
        """# 2. Exploratory Data Analysis

EDA uses the **full cleaned** frame for business understanding (plots only). Modeling still uses the train/test split above."""
    ),
    code(
        """# Plot 1 — Churn distribution
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x=TARGET_COL, hue=TARGET_COL, palette="Set2", legend=False, ax=ax)
ax.set_title("Churn Distribution")
ax.set_xlabel("Churn")
ax.set_ylabel("Number of customers")
plt.tight_layout()
plt.show()"""
    ),
    md(
        """**Business insight:** Roughly one in four customers churned. Retention budget and campaign capacity should be sized around this baseline risk, and models must not ignore the minority (churn) class."""
    ),
    code(
        """# Plot 2 — Churn by Contract
fig, ax = plt.subplots(figsize=(7, 4))
sns.countplot(data=df, x="Contract", hue=TARGET_COL, palette="Set2", ax=ax)
ax.set_title("Churn by Contract Type")
ax.set_xlabel("Contract")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()"""
    ),
    md(
        """**Business insight:** Month-to-month contracts show much higher churn than one- or two-year deals. Retention should prioritize converting month-to-month users into longer commitments (discounts, loyalty perks)."""
    ),
    code(
        """# Plot 3 — Churn by InternetService
fig, ax = plt.subplots(figsize=(7, 4))
sns.countplot(data=df, x="InternetService", hue=TARGET_COL, palette="Set2", ax=ax)
ax.set_title("Churn by Internet Service")
ax.set_xlabel("InternetService")
ax.set_ylabel("Count")
plt.tight_layout()
plt.show()"""
    ),
    md(
        """**Business insight:** Fiber optic customers churn at a higher rate than DSL or no-internet customers — often linked to price or expectation gaps. Service quality and pricing reviews for fiber segments are high-leverage."""
    ),
    code(
        """# Plot 4 — Tenure distribution by churn
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=df, x=TARGET_COL, y="tenure", hue=TARGET_COL, palette="Set2", legend=False, ax=ax)
ax.set_title("Customer Tenure by Churn")
ax.set_xlabel("Churn")
ax.set_ylabel("Tenure (months)")
plt.tight_layout()
plt.show()"""
    ),
    md(
        """**Business insight:** Churners tend to have shorter tenure. Early-lifecycle onboarding and check-ins (first 6–12 months) can reduce drop-off before loyalty builds."""
    ),
    code(
        """# Plot 5 — PaymentMethod vs churn rate
pay = (
    df.groupby("PaymentMethod")[TARGET_COL]
    .apply(lambda s: (s == "Yes").mean() * 100)
    .sort_values(ascending=False)
)
fig, ax = plt.subplots(figsize=(8, 4))
pay.plot(kind="bar", color="salmon", ax=ax)
ax.set_title("Churn Rate (%) by Payment Method")
ax.set_ylabel("Churn rate (%)")
ax.set_xlabel("PaymentMethod")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()"""
    ),
    md(
        """**Business insight:** Electronic check users show elevated churn. Encouraging autopay / card / bank transfer may coincide with more stable relationships (correlation, not proven causation)."""
    ),
    code(
        """# Plot 6 (optional) — Numeric correlations
num_df = df[NUMERIC_COLS].copy()
num_df["ChurnFlag"] = (df[TARGET_COL] == "Yes").astype(int)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(num_df.corr(numeric_only=True), annot=True, cmap="coolwarm", center=0, ax=ax)
ax.set_title("Numeric Feature Correlations")
plt.tight_layout()
plt.show()"""
    ),
    md(
        """**Business insight:** Tenure and TotalCharges are strongly related (longer customers accumulate more spend). MonthlyCharges has a milder link to churn — pricing alone is not the whole story; contract and tenure matter more in later models."""
    ),
    md(
        """# 3. Feature Engineering

We create two features that may improve churn prediction, then include them in the modeling pipeline via `FeatureEngineer` (runs before imputation/encoding, fit only through the train Pipeline)."""
    ),
    code(
        """from src.features import FeatureEngineer, engineered_numeric_columns

fe = FeatureEngineer()
X_train_fe = fe.fit_transform(X_train)
X_test_fe = fe.transform(X_test)

print(X_train_fe[["tenure", "TotalCharges", "AvgMonthlyCharge", "HasMultipleServices"]].head())
print("\\nAvgMonthlyCharge describe:\\n", X_train_fe["AvgMonthlyCharge"].describe())
print("HasMultipleServices value counts:\\n", X_train_fe["HasMultipleServices"].value_counts().sort_index())"""
    ),
    md(
        """### Engineered features

1. **`AvgMonthlyCharge` = TotalCharges / (tenure + 1)**  
   Captures spending intensity. High average charges at low tenure may indicate price sensitivity or plan mismatch.

2. **`HasMultipleServices`** — count of Yes among OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies.  
   Customers with more add-ons often have higher switching costs and lower churn."""
    ),
    md(
        """# 4. Model Development — Decision Tree

We train **two** Decision Tree configurations inside a full Pipeline (features → preprocess → model), compare test metrics, and select a final model."""
    ),
    code(
        """from src.train import decision_tree_configs, train_and_compare, build_model_pipeline

configs = decision_tree_configs(RANDOM_STATE)
for name, clf in configs.items():
    print(name, "→", clf.get_params())

comparison, pipes, best_name = train_and_compare(X_train, y_train, X_test, y_test, configs)
print("\\nTest-set comparison:\\n", comparison.round(4))
print("\\nSelected final model:", best_name)
final_model = pipes[best_name]
final_model"""
    ),
    md(
        """### Model selection rationale

- **DT_Baseline** (`max_depth=5`, `min_samples_leaf=10`): constrained tree — more interpretable, less prone to overfit.
- **DT_Deep** (unlimited depth): higher capacity but often overfits Telco noise.

We select the model with the best **F1**, breaking ties with **recall**, because retention teams care about catching churners without extreme false-alarm rates. The chosen model is stored in `final_model`."""
    ),
    md(
        """# 5. Model Evaluation

Evaluate the final model on the held-out 30% test set with Accuracy, Precision, Recall, F1, and a confusion matrix."""
    ),
    code(
        """from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.train import evaluate_binary

y_pred = final_model.predict(X_test)
metrics = evaluate_binary(y_test, y_pred)
print("Metrics:", {k: round(v, 4) for k, v in metrics.items()})
print("\\nClassification report:\\n", classification_report(y_test, y_pred, target_names=["No", "Yes"]))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(ax=ax, cmap="Blues")
ax.set_title("Confusion Matrix — Final Model")
plt.tight_layout()
plt.show()"""
    ),
    md(
        """### Business interpretation

| Error | Meaning | Cost to telecom |
|-------|---------|-----------------|
| **False Negative** | Churner predicted as stay | Lost revenue; no retention outreach |
| **False Positive** | Loyal customer flagged as churn | Extra call/discount cost |

**Precision vs Recall:** For identifying customers who *may* churn, we prioritize **Recall**. Missing a true churner (FN) usually costs more than occasionally contacting a safe customer (FP), as long as campaign cost stays reasonable. If discounts are very expensive, raise the probability threshold to improve precision.

Accuracy alone is misleading with ~74% non-churners — a naive “always No” model looks strong but helps nobody."""
    ),
    md(
        """# 6. Model Interpretation

What drives the tree’s churn predictions? Feature importances and a shallow tree diagram."""
    ),
    code(
        """from sklearn.tree import plot_tree

# Recover feature names after ColumnTransformer
pre = final_model.named_steps["preprocess"]
feat_names = list(pre.get_feature_names_out())
importances = final_model.named_steps["model"].feature_importances_
imp_df = (
    pd.DataFrame({"feature": feat_names, "importance": importances})
    .sort_values("importance", ascending=False)
)
print("Top 10 features:\\n", imp_df.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=imp_df.head(10), y="feature", x="importance", color="steelblue", ax=ax)
ax.set_title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()"""
    ),
    code(
        """# Tree visualization (depth limited for readability)
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    final_model.named_steps["model"],
    feature_names=feat_names,
    class_names=["No", "Yes"],
    filled=True,
    max_depth=3,
    fontsize=8,
    ax=ax,
)
ax.set_title("Decision Tree (first 3 levels)")
plt.tight_layout()
plt.show()"""
    ),
    md(
        """### Key findings

Top splits typically involve **Contract** (month-to-month), **tenure**, **InternetService / OnlineSecurity**, and charges. That aligns with EDA: short-tenure, month-to-month, fiber / low-security customers are higher risk. The retention team can operationalize these as segments for proactive offers."""
    ),
    md(
        """# 7. Model Saving

Persist the **entire** Pipeline (feature engineering + preprocessing + Decision Tree) so the API applies identical transforms to new customers."""
    ),
    code(
        """import joblib
from src.config import MODEL_PATH, MODEL_DIR
from src.predict import predict_churn, load_model

MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(final_model, MODEL_PATH)
print("Saved pipeline →", MODEL_PATH)

# Round-trip verification on one test customer
sample_customer = X_test.iloc[0].to_dict()
loaded = load_model(MODEL_PATH)
from_notebook = predict_churn(sample_customer, pipeline=final_model)
from_disk = predict_churn(sample_customer, pipeline=loaded)
print("Notebook model:", from_notebook)
print("Loaded model:  ", from_disk)
assert from_notebook["prediction"] == from_disk["prediction"]
print("Round-trip OK")"""
    ),
    md(
        """# 8. API

The REST API lives in `app.py` (`POST /predict`). See `sample_request.json` and `docs/stories/story-08-rest-api.md`.

```bash
uvicorn app:app --reload --port 8000
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @sample_request.json
```"""
    ),
    md(
        """# 9. Bonus Improvements

Optional enhancements: class weights, GridSearchCV, extra models, cross-validation, and threshold tuning."""
    ),
    code(
        """from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# 9a — Class imbalance: balanced Decision Tree
balanced_clf = DecisionTreeClassifier(
    max_depth=5, min_samples_leaf=10, class_weight="balanced", random_state=RANDOM_STATE
)
balanced_pipe = build_model_pipeline(balanced_clf)
balanced_pipe.fit(X_train, y_train)
bal_metrics = evaluate_binary(y_test, balanced_pipe.predict(X_test))
print("Balanced DT metrics:", {k: round(v, 4) for k, v in bal_metrics.items()})

# 9b — GridSearchCV on Decision Tree hyperparameters (CV on train only)
param_grid = {
    "model__max_depth": [3, 5, 8, None],
    "model__min_samples_leaf": [5, 10, 20],
    "model__criterion": ["gini", "entropy"],
}
base = build_model_pipeline(DecisionTreeClassifier(random_state=RANDOM_STATE))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(base, param_grid, scoring="f1", cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV F1:", round(grid.best_score_, 4))
grid_test = evaluate_binary(y_test, grid.predict(X_test))
print("Grid best on test:", {k: round(v, 4) for k, v in grid_test.items()})"""
    ),
    code(
        """# 9c — Additional models comparison
extra_configs = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
}
extra_rows = []
for name, clf in extra_configs.items():
    pipe = build_model_pipeline(clf)
    pipe.fit(X_train, y_train)
    m = evaluate_binary(y_test, pipe.predict(X_test))
    m["model"] = name
    extra_rows.append(m)

bonus_table = pd.concat([
    comparison.reset_index().rename(columns={"index": "model"}),
    pd.DataFrame([{"model": "DT_Balanced", **bal_metrics}]),
    pd.DataFrame([{"model": "DT_GridSearch", **grid_test}]),
    pd.DataFrame(extra_rows),
], ignore_index=True)
print(bonus_table.round(4))"""
    ),
    code(
        """# 9d — Stratified 5-fold CV F1 for final Decision Tree pipeline
cv_scores = cross_val_score(build_model_pipeline(configs[best_name]), X_train, y_train, cv=cv, scoring="f1")
print("CV F1 scores:", np.round(cv_scores, 4), "mean=", round(cv_scores.mean(), 4))

# 9e — Threshold tuning to favor recall
probas = final_model.predict_proba(X_test)[:, 1]
threshold_rows = []
for t in [0.3, 0.4, 0.5, 0.6]:
    pred_t = (probas >= t).astype(int)
    m = evaluate_binary(y_test, pred_t)
    m["threshold"] = t
    threshold_rows.append(m)
thr_df = pd.DataFrame(threshold_rows)[["threshold", "precision", "recall", "f1", "accuracy"]]
print(thr_df.round(4))
print("\\nLower thresholds increase recall (more churners caught) at the cost of precision.")"""
    ),
    md(
        """### Bonus takeaways

- **`class_weight='balanced'`** often lifts recall on the minority churn class.
- **GridSearchCV** (on train folds only) finds a stronger depth/leaf/criterion combo without peeking at the test set.
- **Logistic Regression / Random Forest** provide baselines or upgrades; Random Forest frequently edges a single tree on F1.
- **Threshold tuning** lets the business dial recall vs precision after the model is trained — useful when campaign cost constraints change."""
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.0"},
    },
    "cells": cells,
}

NB_PATH.parent.mkdir(parents=True, exist_ok=True)
NB_PATH.write_text(json.dumps(nb, indent=1))
print("Wrote", NB_PATH, "cells=", len(cells))
