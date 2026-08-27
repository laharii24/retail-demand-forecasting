"""
Day 16: Benchmark XGBoost and CatBoost against the LightGBM baseline,
using the same reduced feature set from Day 15.
"""
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from train_lightgbm import load_data, split_features_target, evaluate

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Day 15's kept features
KEEP_FEATURES = [
    "price", "revenue", "daily_sales_lag_14", "daily_sales_lag_7",
    "daily_rolling_mean_7", "daily_rolling_std_7", "customer_segments",
    "seasonality_factors", "demand_trend", "promotions",
]

LGBM_PARAMS = {
    "n_estimators": 653, "learning_rate": 0.026, "num_leaves": 117,
    "max_depth": 11, "min_child_samples": 9, "random_state": 42,
}


def main():
    print("Loading train/test data ...")
    train, test = load_data()
    X_train, y_train, cat_cols = split_features_target(train)
    X_test, y_test, _ = split_features_target(test)
    for col in cat_cols:
        X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

    X_train = X_train[[c for c in KEEP_FEATURES if c in X_train.columns]]
    X_test = X_test[[c for c in KEEP_FEATURES if c in X_test.columns]]
    cat_cols = [c for c in cat_cols if c in X_train.columns]

    results = {}

    # --- LightGBM ---
    print("\nTraining LightGBM ...")
    lgbm = LGBMRegressor(**LGBM_PARAMS)
    lgbm.fit(X_train, y_train, categorical_feature=cat_cols)
    results["LightGBM"] = evaluate(y_test.values, lgbm.predict(X_test))

    # --- XGBoost (needs numeric encoding, not pandas 'category' dtype directly for old versions,
    # but modern xgboost supports enable_categorical=True) ---
    print("Training XGBoost ...")
    xgb = XGBRegressor(
        n_estimators=653, learning_rate=0.026, max_depth=11,
        random_state=42, enable_categorical=True, tree_method="hist",
    )
    xgb.fit(X_train, y_train)
    results["XGBoost"] = evaluate(y_test.values, xgb.predict(X_test))

    # --- CatBoost (handles categoricals natively via cat_features) ---
    print("Training CatBoost ...")
    cat_feature_idx = [X_train.columns.get_loc(c) for c in cat_cols]
    cb = CatBoostRegressor(
        iterations=653, learning_rate=0.026, depth=10,
        random_state=42, verbose=False,
    )
    cb.fit(X_train, y_train, cat_features=cat_feature_idx)
    results["CatBoost"] = evaluate(y_test.values, cb.predict(X_test))

    # --- Comparison table ---
    print("\n--- Day 16: Model Benchmark ---")
    print(f"{'Model':<12}{'MAE':<10}{'RMSE':<10}{'MAPE (%)':<12}{'WAPE (%)'}")
    for name, m in results.items():
        print(f"{name:<12}{m['MAE']:<10.3f}{m['RMSE']:<10.3f}{m['MAPE (%)']:<12.3f}{m['WAPE (%)']:.3f}")

    best = min(results.items(), key=lambda kv: kv[1]["WAPE (%)"])
    print(f"\nBest model by WAPE: {best[0]} ({best[1]['WAPE (%)']:.3f}%)")


if __name__ == "__main__":
    main()