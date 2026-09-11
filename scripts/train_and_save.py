"""Train final churn pipeline, save model, and emit sample request JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MODEL_DIR, MODEL_PATH, RANDOM_STATE, TEST_SIZE  # noqa: E402
from src.data_loader import encode_target, load_and_clean, split_features_target  # noqa: E402
from src.predict import predict_churn  # noqa: E402
from src.train import train_and_compare  # noqa: E402


def main() -> None:
    df = load_and_clean()
    X, y_raw = split_features_target(df)
    y = encode_target(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    comparison, pipes, best_name = train_and_compare(X_train, y_train, X_test, y_test)
    print("Comparison:\n", comparison)
    print("Selected:", best_name)

    best = pipes[best_name]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    print("Saved:", MODEL_PATH)

    # Round-trip check on first test row
    sample = X_test.iloc[0].to_dict()
    result = predict_churn(sample, pipeline=best)
    print("Sample prediction:", result)

    sample_path = ROOT / "sample_request.json"
    # Prefer a high-risk looking month-to-month fiber customer if present
    risky = X_test[
        (X_test.get("Contract") == "Month-to-month")
        if "Contract" in X_test.columns
        else slice(None)
    ]
    row = risky.iloc[0] if len(risky) else X_test.iloc[0]
    payload = {k: (None if (isinstance(v, float) and v != v) else v) for k, v in row.to_dict().items()}
    # JSON-serializable types
    clean = {}
    for k, v in payload.items():
        if hasattr(v, "item"):
            clean[k] = v.item()
        else:
            clean[k] = v
    sample_path.write_text(json.dumps(clean, indent=2))
    print("Wrote", sample_path)
    print("sample_request prediction:", predict_churn(clean, pipeline=best))


if __name__ == "__main__":
    main()
