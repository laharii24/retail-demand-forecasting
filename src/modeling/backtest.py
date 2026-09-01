"""
Day 21: Backtesting framework — walk-forward evaluation of the FINAL
pipeline (Day 15 features + Day 17 ensemble), across multiple rolling
time windows. This is the "would this have worked in production"
check, reusable before any future deployment.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from train_lightgbm import evaluate

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

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
N_WINDOWS = 5


def load_full_data():
    train = pd.read_csv(PROCESSED_DIR / "train.csv", parse_dates=["date"])
    test = pd.read_csv(PROCESSED_DIR / "test.csv", parse_dates=["date"])
    return pd.concat([train, test]).sort_values("date").reset_index(drop=True)


def split_features_target(df):
    keep = [c for c in KEEP_FEATURES if c in df.columns]
    X = df[keep].copy()
    y = df["sales_quantity"]
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype("category")
    return X, y, cat_cols


def train_and_predict(X_train, y_train, X_test, cat_cols):
    lgbm = LGBMRegressor(**LGBM_PARAMS)
    lgbm.fit(X_train, y_train, categorical_feature=cat_cols)
    preds_lgbm = lgbm.predict(X_test)

    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]
    cb = CatBoostRegressor(iterations=653, learning_rate=0.026, depth=10, random_state=42, verbose=False)
    cb.fit(X_train, y_train, cat_features=cat_idx)
    preds_cb = cb.predict(X_test)

    return ENSEMBLE_WEIGHT * preds_lgbm + (1 - ENSEMBLE_WEIGHT) * preds_cb


def main():
    full = load_full_data()
    unique_dates = sorted(full["date"].unique())
    fold_size = len(unique_dates) // (N_WINDOWS + 1)

    results = []
    for i in range(1, N_WINDOWS + 1):
        cutoff = unique_dates[fold_size * i]
        test_end = unique_dates[min(fold_size * (i + 1) - 1, len(unique_dates) - 1)]

        train_fold = full[full["date"] < cutoff]
        test_fold = full[(full["date"] >= cutoff) & (full["date"] <= test_end)]
        if len(test_fold) == 0:
            continue

        X_train, y_train, cat_cols = split_features_target(train_fold)
        X_test, y_test, _ = split_features_target(test_fold)
        for col in cat_cols:
            X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

        preds = train_and_predict(X_train, y_train, X_test, cat_cols)
        metrics = evaluate(y_test.values, preds)
        metrics["fold"] = i
        metrics["train_size"] = len(train_fold)
        metrics["test_size"] = len(test_fold)
        results.append(metrics)

        print(f"Fold {i}: train={len(train_fold)} test={len(test_fold)} "
              f"WAPE={metrics['WAPE (%)']:.3f}%")

    df_results = pd.DataFrame(results)
    print("\n--- Day 21: Final pipeline backtest summary ---")
    print(df_results[["fold", "MAE", "RMSE", "MAPE (%)", "WAPE (%)"]].to_string(index=False))
    print(f"\nMean WAPE: {df_results['WAPE (%)'].mean():.3f}%  (std: {df_results['WAPE (%)'].std():.3f}%)")
    print(f"Mean MAPE: {df_results['MAPE (%)'].mean():.3f}%  (std: {df_results['MAPE (%)'].std():.3f}%)")

    df_results.to_csv(PROCESSED_DIR / "backtest_results.csv", index=False)
    print(f"\nSaved to {PROCESSED_DIR / 'backtest_results.csv'}")


if __name__ == "__main__":
    main()