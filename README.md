# Customer Churn Prediction (Telco)

Recording Link : https://drive.google.com/drive/folders/19z9ydveS75TSH21tDuYROCzIL1fU_dl8?usp=share_link

End-to-end machine learning project that predicts whether a telecom customer will churn, using the IBM Telco Customer Churn dataset. The solution includes exploratory analysis, feature engineering, Decision Tree models, evaluation, model interpretation, a saved sklearn pipeline, and a FastAPI prediction service.

## Project structure

├── data/                     # Place Telco CSV here
├── notebook/churn_analysis.ipynb
├── model/churn_pipeline.pkl  
├── src/                      # Reusable Python modules
├── app.py                    # FastAPI app
├── sample_request.json
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.10 or newer

### Virtual environment and dependencies

```bash
cd GAURAV_3188603_DATA_SCIENCE
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the notebook

```bash
source .venv/bin/activate
jupyter notebook notebook/churn_analysis.ipynb
```

Constants: `random_state=42`, train/test split **70:30**.

## Run the API

```bash
source .venv/bin/activate
uvicorn app:app --reload --port 8000
```

Open interactive docs: http://127.0.0.1:8000/docs

### Sample request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

### Sample response

```json
{
  "prediction": "Yes",
  "churn_probability": 0.5247
}
```

Exact probability depends on the trained model and input row; response fields are always `prediction` and `churn_probability`.

## Assignment mapping

| Requirement | Where |
|-------------|--------|
| Data prep + 70:30 split, seed 42 | Notebook 1, `src/` |
| EDA (≥5 plots) | Notebook 2 |
| ≥2 engineered features | Notebook 3, `src/features.py` |
| ≥2 Decision Tree configs | Notebook 4, `src/train.py` |
| Metrics + confusion matrix | Notebook 5 |
| Feature importance / tree viz | Notebook 6 |
| Saved pipeline | `model/churn_pipeline.pkl` |
| REST API | `app.py` |
