"""
Day 23 prep: Train and save the final ensemble models (LightGBM + CatBoost)
so the API can load them instantly instead of retraining on every request.
"""
import joblib
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from train_lightgbm import load_data, split_features_target

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS_DIR.mkdir(exist_ok=True)

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
    train, test = load_data()
    X_train, y_train, cat_cols = split_features_target(train)
    X_train = X_train[[c for c in KEEP_FEATURES if c in X_train.columns]]
    cat_cols = [c for c in cat_cols if c in X_train.columns]

    print("Training LightGBM ...")
    lgbm = LGBMRegressor(**LGBM_PARAMS)
    lgbm.fit(X_train, y_train, categorical_feature=cat_cols)
    joblib.dump(lgbm, MODELS_DIR / "ensemble_lgbm.pkl")

    print("Training CatBoost ...")
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]
    cb = CatBoostRegressor(iterations=653, learning_rate=0.026, depth=10, random_state=42, verbose=False)
    cb.fit(X_train, y_train, cat_features=cat_idx)
    cb.save_model(str(MODELS_DIR / "ensemble_catboost.cbm"))

    # Save the category mappings so the API can encode incoming requests the same way
    cat_categories = {col: X_train[col].cat.categories.tolist() for col in cat_cols}
    joblib.dump({"cat_cols": cat_cols, "cat_categories": cat_categories, "feature_order": KEEP_FEATURES},
                MODELS_DIR / "ensemble_meta.pkl")

    print(f"\nSaved models to {MODELS_DIR}")


if __name__ == "__main__":
    main()