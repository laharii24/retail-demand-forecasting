"""
Day 23: FastAPI service serving demand forecasts from the final ensemble
(LightGBM + CatBoost, 50/50 blend).
"""
import joblib
import pandas as pd
from pathlib import Path
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostRegressor

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
ENSEMBLE_WEIGHT = 0.5

app = FastAPI(title="Retail Demand Forecast API")

lgbm = joblib.load(MODELS_DIR / "ensemble_lgbm.pkl")
cb = CatBoostRegressor()
cb.load_model(str(MODELS_DIR / "ensemble_catboost.cbm"))
meta = joblib.load(MODELS_DIR / "ensemble_meta.pkl")


class ForecastRequest(BaseModel):
    price: float
    revenue: float
    daily_sales_lag_14: float
    daily_sales_lag_7: float
    daily_rolling_mean_7: float
    daily_rolling_std_7: float
    customer_segments: str
    seasonality_factors: str
    demand_trend: str
    promotions: Union[bool, float, str]  # accepts True/False, 0/1, or a category string


@app.get("/")
def root():
    return {"status": "ok", "message": "Retail Demand Forecast API is running"}


@app.post("/predict")
def predict(req: ForecastRequest):
    row = pd.DataFrame([req.model_dump()])[meta["feature_order"]]

    for col in row.columns:
        if col in meta["cat_cols"]:
            row[col] = pd.Categorical(row[col], categories=meta["cat_categories"][col])
        else:
            row[col] = pd.to_numeric(row[col], errors="coerce")

    pred_lgbm = lgbm.predict(row)[0]
    pred_cb = cb.predict(row)[0]
    forecast = ENSEMBLE_WEIGHT * pred_lgbm + (1 - ENSEMBLE_WEIGHT) * pred_cb

    return {
        "forecasted_demand": round(float(forecast), 2),
        "lightgbm_prediction": round(float(pred_lgbm), 2),
        "catboost_prediction": round(float(pred_cb), 2),
    }