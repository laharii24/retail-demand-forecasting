"""
Day 24: Basic drift monitoring — compares train vs test feature and
prediction distributions using Population Stability Index (PSI).
Reusable: rerun periodically against new incoming data to catch when
the model may need retraining.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from train_lightgbm import load_data, split_features_target

KEEP_FEATURES = [
    "price", "revenue", "daily_sales_lag_14", "daily_sales_lag_7",
    "daily_rolling_mean_7", "daily_rolling_std_7", "customer_segments",
    "seasonality_factors", "demand_trend", "promotions",
]
LGBM_PARAMS = {
    "n_estimators": 653, "learning_rate": 0.026, "num_leaves": 117,
    "max_depth": 11, "min_child_samples": 9, "random_state": 42,
}
NUMERIC_FEATURES = [
    "price", "revenue", "daily_sales_lag_14", "daily_sales_lag_7",
    "daily_rolling_mean_7", "daily_rolling_std_7",
]
CATEGORICAL_FEATURES = ["customer_segments", "seasonality_factors", "demand_trend"]


def psi_numeric(expected, actual, bins=10):
    """PSI for a numeric feature using quantile bins from the expected (train) distribution."""
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    breakpoints = np.unique(breakpoints)

    expected_pct = pd.cut(expected, breakpoints).value_counts(normalize=True).sort_index()
    actual_pct = pd.cut(actual, breakpoints).value_counts(normalize=True).sort_index()

    expected_pct = expected_pct.replace(0, 1e-4)
    actual_pct = actual_pct.replace(0, 1e-4)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


def psi_categorical(expected, actual):
    """PSI for a categorical feature comparing proportion per category."""
    expected_pct = expected.value_counts(normalize=True)
    actual_pct = actual.value_counts(normalize=True)
    categories = set(expected_pct.index) | set(actual_pct.index)

    psi = 0.0
    for cat in categories:
        e = expected_pct.get(cat, 1e-4)
        a = actual_pct.get(cat, 1e-4)
        psi += (a - e) * np.log(a / e)
    return psi


def interpret(psi):
    if psi < 0.1:
        return "no significant shift"
    elif psi < 0.25:
        return "MODERATE shift - monitor"
    else:
        return "SIGNIFICANT shift - investigate/retrain"


def main():
    train, test = load_data()

    print("--- Feature drift (train vs test) ---")
    for col in NUMERIC_FEATURES:
        psi = psi_numeric(train[col].dropna(), test[col].dropna())
        print(f"{col:<25} PSI={psi:.4f}  ({interpret(psi)})")

    for col in CATEGORICAL_FEATURES:
        psi = psi_categorical(train[col], test[col])
        print(f"{col:<25} PSI={psi:.4f}  ({interpret(psi)})")

    print("\n--- Prediction drift ---")
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

    train_preds = 0.5 * lgbm.predict(X_train_r) + 0.5 * cb.predict(X_train_r)
    test_preds = 0.5 * lgbm.predict(X_test_r) + 0.5 * cb.predict(X_test_r)

    pred_psi = psi_numeric(pd.Series(train_preds), pd.Series(test_preds))
    print(f"{'prediction distribution':<25} PSI={pred_psi:.4f}  ({interpret(pred_psi)})")

    print(f"\nTrain predictions: mean={train_preds.mean():.2f}, std={train_preds.std():.2f}")
    print(f"Test predictions:  mean={test_preds.mean():.2f}, std={test_preds.std():.2f}")


if __name__ == "__main__":
    main()