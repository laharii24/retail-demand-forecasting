"""
Day 25: Streamlit dashboard for the retail demand forecasting project.
Visualizes forecast vs actual, WAPE by store, and feature importance
for the final ensemble (LightGBM + CatBoost, 50/50 blend).

Run with: streamlit run src/dashboard/app.py
"""
import sys
from pathlib import Path

# Allow importing from src/modeling
sys.path.append(str(Path(__file__).resolve().parents[1] / "modeling"))

import pandas as pd
import numpy as np
import streamlit as st
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from train_lightgbm import load_data, split_features_target, evaluate

st.set_page_config(page_title="Retail Demand Forecast Dashboard", layout="wide")

KEEP_FEATURES = [
    "price", "revenue", "daily_sales_lag_14", "daily_sales_lag_7",
    "daily_rolling_mean_7", "daily_rolling_std_7", "customer_segments",
    "seasonality_factors", "demand_trend", "promotions",
]
LGBM_PARAMS = {
    "n_estimators": 653, "learning_rate": 0.026, "num_leaves": 117,
    "max_depth": 11, "min_child_samples": 9, "random_state": 42,
}
ENSEMBLE_WEIGHT = 0.5


@st.cache_data
def get_predictions():
    train, test = load_data()
    X_train, y_train, cat_cols = split_features_target(train)
    X_test, y_test, _ = split_features_target(test)
    for col in cat_cols:
        X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

    X_train_r = X_train[[c for c in KEEP_FEATURES if c in X_train.columns]]
    X_test_r = X_test[[c for c in KEEP_FEATURES if c in X_test.columns]]
    cat_cols_r = [c for c in cat_cols if c in X_train_r.columns]

    lgbm = LGBMRegressor(**LGBM_PARAMS)
    lgbm.fit(X_train_r, y_train, categorical_feature=cat_cols_r)

    cat_idx = [X_train_r.columns.get_loc(c) for c in cat_cols_r]
    cb = CatBoostRegressor(iterations=653, learning_rate=0.026, depth=10, random_state=42, verbose=False)
    cb.fit(X_train_r, y_train, cat_features=cat_idx)

    preds = ENSEMBLE_WEIGHT * lgbm.predict(X_test_r) + (1 - ENSEMBLE_WEIGHT) * cb.predict(X_test_r)

    results = test.reset_index(drop=True).copy()
    results["forecast"] = preds
    results["abs_error"] = (results["sales_quantity"] - results["forecast"]).abs()

    importance = pd.Series(lgbm.feature_importances_, index=X_train_r.columns).sort_values(ascending=False)
    return results, importance, evaluate(y_test.values, preds)


st.title("📊 Retail Demand Forecast Dashboard")
st.caption("Final ensemble: LightGBM + CatBoost (50/50 blend)")

with st.spinner("Loading model and generating predictions..."):
    results, importance, overall_metrics = get_predictions()

# --- Top-line metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("WAPE", f"{overall_metrics['WAPE (%)']:.3f}%")
col2.metric("MAPE", f"{overall_metrics['MAPE (%)']:.3f}%")
col3.metric("MAE", f"{overall_metrics['MAE']:.2f}")
col4.metric("RMSE", f"{overall_metrics['RMSE']:.2f}")

st.divider()

# --- Forecast vs Actual over time ---
st.subheader("Forecast vs Actual (daily aggregate)")
daily = results.groupby("date")[["sales_quantity", "forecast"]].sum().reset_index()
daily = daily.rename(columns={"sales_quantity": "Actual", "forecast": "Forecast"})
st.line_chart(daily.set_index("date")[["Actual", "Forecast"]])

st.divider()

# --- WAPE by store ---
st.subheader("WAPE by Store")

def wape_group(g):
    return (g["abs_error"].sum() / g["sales_quantity"].sum()) * 100

by_store = results.groupby("store_id").apply(wape_group, include_groups=False).sort_values(ascending=False)

col_a, col_b = st.columns(2)
with col_a:
    st.write("**Worst 10 stores**")
    st.bar_chart(by_store.head(10))
with col_b:
    st.write("**Best 10 stores**")
    st.bar_chart(by_store.tail(10))

st.divider()

# --- Feature importance ---
st.subheader("Feature Importance (LightGBM)")
st.bar_chart(importance)

st.divider()
st.caption("Retail Demand Forecasting — 28-day build. Day 25: Streamlit dashboard.")