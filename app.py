"""FastAPI service for Telco customer churn prediction."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.predict import load_model, predict_churn


class CustomerInput(BaseModel):
    """Raw customer features matching the Telco CSV (without customerID / Churn)."""

    model_config = ConfigDict(extra="ignore")

    gender: Literal["Male", "Female"]
    SeniorCitizen: int | str = Field(..., description="0/1 or '0'/'1'")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: str
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: str
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
