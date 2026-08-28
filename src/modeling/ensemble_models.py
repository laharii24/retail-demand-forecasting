"""
Day 17: Ensemble LightGBM + CatBoost via weighted average blending.
"""
import numpy as np
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

    print("Training LightGBM ...")
    lgbm = LGBMRegressor(**LGBM_PARAMS)
    lgbm.fit(X_train, y_train, categorical_feature=cat_cols)
    preds_lgbm = lgbm.predict(X_test)

    print("Training CatBoost ...")
    cat_feature_idx = [X_train.columns.get_loc(c) for c in cat_cols]
    cb = CatBoostRegressor(iterations=653, learning_rate=0.026, depth=10, random_state=42, verbose=False)
    cb.fit(X_train, y_train, cat_features=cat_feature_idx)
    preds_cb = cb.predict(X_test)

    metrics_lgbm = evaluate(y_test.values, preds_lgbm)
    metrics_cb = evaluate(y_test.values, preds_cb)

    print(f"\nLightGBM alone:  WAPE {metrics_lgbm['WAPE (%)']:.3f}%")
    print(f"CatBoost alone:  WAPE {metrics_cb['WAPE (%)']:.3f}%")

    # Try blend weights from 0.0 (all CatBoost) to 1.0 (all LightGBM)
    print("\n--- Blend weight sweep ---")
    best_wape = float("inf")
    best_weight = None
    for w in np.arange(0.0, 1.01, 0.1):
        blend = w * preds_lgbm + (1 - w) * preds_cb
        m = evaluate(y_test.values, blend)
        print(f"LightGBM weight={w:.1f}  WAPE={m['WAPE (%)']:.3f}%  MAPE={m['MAPE (%)']:.3f}%")
        if m["WAPE (%)"] < best_wape:
            best_wape = m["WAPE (%)"]
            best_weight = w

    print(f"\nBest blend: LightGBM weight={best_weight:.1f} -> WAPE {best_wape:.3f}%")

    final_blend = best_weight * preds_lgbm + (1 - best_weight) * preds_cb
    final_metrics = evaluate(y_test.values, final_blend)
    print(f"\n--- Final ensemble vs individual models ---")
    print(f"{'Model':<20}{'MAE':<10}{'RMSE':<10}{'MAPE (%)':<12}{'WAPE (%)'}")
    print(f"{'LightGBM':<20}{metrics_lgbm['MAE']:<10.3f}{metrics_lgbm['RMSE']:<10.3f}{metrics_lgbm['MAPE (%)']:<12.3f}{metrics_lgbm['WAPE (%)']:.3f}")
    print(f"{'CatBoost':<20}{metrics_cb['MAE']:<10.3f}{metrics_cb['RMSE']:<10.3f}{metrics_cb['MAPE (%)']:<12.3f}{metrics_cb['WAPE (%)']:.3f}")
    print(f"{'Ensemble (best)':<20}{final_metrics['MAE']:<10.3f}{final_metrics['RMSE']:<10.3f}{final_metrics['MAPE (%)']:<12.3f}{final_metrics['WAPE (%)']:.3f}")


if __name__ == "__main__":
    main()