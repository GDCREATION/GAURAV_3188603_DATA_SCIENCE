"""FastAPI service for Telco customer churn prediction."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.predict import load_model, predict_churn


YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]
YesNoNoPhone = Literal["Yes", "No", "No phone service"]
PaymentMethodValue = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


class CustomerInput(BaseModel):
    """Raw customer features matching the Telco CSV (without customerID / Churn)."""

    model_config = ConfigDict(extra="ignore")

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal["0", "1"] | int = Field(..., description="0/1 or '0'/'1'")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(..., ge=0)
    PhoneService: YesNo
    MultipleLines: YesNoNoPhone
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethodValue
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


class PredictResponse(BaseModel):
    prediction: Literal["Yes", "No"]
    churn_probability: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_model()
    except FileNotFoundError as exc:
        # Allow app to start so /health can explain the issue; /predict will 503
        app.state.model_loaded = False
        app.state.model_error = str(exc)
    else:
        app.state.model_loaded = True
        app.state.model_error = None
    yield


app = FastAPI(
    title="Telco Customer Churn API",
    description="Predict whether a telecom customer is likely to churn.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ok" if getattr(app.state, "model_loaded", False) else "model_missing",
        "model_loaded": getattr(app.state, "model_loaded", False),
        "detail": getattr(app.state, "model_error", None),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(customer: CustomerInput):
    if not getattr(app.state, "model_loaded", False):
        raise HTTPException(
            status_code=503,
            detail=getattr(app.state, "model_error", "Model not loaded"),
        )
    try:
        payload = customer.model_dump()
        # Keep SeniorCitizen as string for consistency with training pipeline
        payload["SeniorCitizen"] = str(payload["SeniorCitizen"])
        result = predict_churn(payload)
        return PredictResponse(**result)
    except Exception as exc:  # noqa: BLE001 — return safe 500 to clients
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
