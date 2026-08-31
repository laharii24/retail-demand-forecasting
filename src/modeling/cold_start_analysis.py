"""
Day 19: Cold-start analysis. With ~99% of test combos unseen in train,
this checks whether the model's error is meaningfully different on
cold-start rows vs the rare "seen before" rows, and evaluates a simple
store-level average fallback as an alternative.
"""
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from train_lightgbm import load_data, split_features_target, evaluate

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


def main():
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
    preds_lgbm = lgbm.predict(X_test_r)

    cat_idx = [X_train_r.columns.get_loc(c) for c in cat_cols_r]
    cb = CatBoostRegressor(iterations=653, learning_rate=0.026, depth=10, random_state=42, verbose=False)
    cb.fit(X_train_r, y_train, cat_features=cat_idx)
    preds_cb = cb.predict(X_test_r)

    preds = ENSEMBLE_WEIGHT * preds_lgbm + (1 - ENSEMBLE_WEIGHT) * preds_cb

    results = test.reset_index(drop=True).copy()
    results["y_true"] = y_test.values
    results["y_pred"] = preds
    results["abs_error"] = (results["y_true"] - results["y_pred"]).abs()

    train_combos = set(zip(train["product_id"], train["store_id"]))
    results["is_cold_start"] = ~results.apply(lambda r: (r["product_id"], r["store_id"]) in train_combos, axis=1)

    def wape(g):
        return (g["abs_error"].sum() / g["y_true"].sum()) * 100

    cold = results[results["is_cold_start"]]
    warm = results[~results["is_cold_start"]]

    print(f"Cold-start rows: {len(cold)} ({len(cold)/len(results)*100:.1f}%)  WAPE: {wape(cold):.3f}%")
    print(f"Warm rows:       {len(warm)} ({len(warm)/len(results)*100:.1f}%)  WAPE: {wape(warm):.3f}%")

    # Compare against a naive store-average fallback
    store_avg = train.groupby("store_id")["sales_quantity"].mean()
    results["store_avg_pred"] = results["store_id"].map(store_avg)
    results["store_avg_error"] = (results["y_true"] - results["store_avg_pred"]).abs()
    naive_wape = (results["store_avg_error"].sum() / results["y_true"].sum()) * 100
    print(f"\nNaive store-average fallback WAPE: {naive_wape:.3f}%  (vs ensemble {wape(results):.3f}%)")


if __name__ == "__main__":
    main()                                              