"""Project-wide configuration constants and paths."""

from pathlib import Path

# Repository root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILENAME = "TelcoCustomerChurn.csv"
DATA_PATH = DATA_DIR / DATA_FILENAME

# Model artifacts
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "churn_pipeline.pkl"

# Notebook / docs
NOTEBOOK_DIR = PROJECT_ROOT / "notebook"
DOCS_DIR = PROJECT_ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figures"
DATA_DICT_FILENAME = "TelcoCustomerChurn-Data-Dictionary.csv"
DATA_DICT_PATH = DOCS_DIR / DATA_DICT_FILENAME

# Train/test split (assignment requirements)
RANDOM_STATE = 42
TEST_SIZE = 0.30

# Target and ID columns
TARGET_COL = "Churn"
ID_COL = "customerID"

# All CSV columns in file order (must match docs/TelcoCustomerChurn-Data-Dictionary.csv)
ALL_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

# Modeling features: ALL_COLUMNS minus ID and target (same order as the CSV)
FEATURE_COLS = [c for c in ALL_COLUMNS if c not in (ID_COL, TARGET_COL)]

# Numeric columns in the raw Telco dataset (after TotalCharges cleaning)
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Categorical feature columns (excluding ID and target; CSV order among categoricals)
CATEGORICAL_COLS = [
    c for c in FEATURE_COLS if c not in NUMERIC_COLS
]

# Service columns used for engineered feature HasMultipleServices
SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
