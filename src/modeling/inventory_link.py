"""
Day 20: Link demand forecast with inventory monitoring.

Joins the Day 17 ensemble's forecasted demand with current inventory
levels (per product_id + store_id) to flag potential stockout and
overstock risk.
"""
import pandas as pd
from pathlib import Path
import sqlite3
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from train_lightgbm import load_data, split_features_target, evaluate

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
DB_PATH = PROCESSED_DIR / "warehouse.db"

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

    forecast = ENSEMBLE_WEIGHT * preds_lgbm + (1 - ENSEMBLE_WEIGHT) * preds_cb

    results = test.reset_index(drop=True).copy()
    results["forecasted_demand"] = forecast

    print(f"Overall forecast WAPE: {evaluate(y_test.values, forecast)['WAPE (%)']:.3f}%")

    # --- Load inventory and join on product_id + store_id ---
    conn = sqlite3.connect(DB_PATH)
    inventory = pd.read_sql(
        "SELECT product_id, store_id, stock_levels, supplier_lead_time_days, "
        "stockout_frequency, reorder_point, warehouse_capacity FROM inventory_monitoring",
        conn,
    )
    conn.close()

    merged = results.merge(inventory, on=["product_id", "store_id"], how="left")
    n_unmatched = merged["stock_levels"].isna().sum()
    print(f"\nRows with no inventory match: {n_unmatched} / {len(merged)}")

    matched = merged.dropna(subset=["stock_levels"]).copy()

    # --- Flag risk ---
    matched["stockout_risk"] = matched["forecasted_demand"] > matched["stock_levels"]
    matched["overstock_risk"] = matched["forecasted_demand"] < (matched["stock_levels"] * 0.2)  # demand under 20% of stock

    print(f"\nStockout risk rows: {matched['stockout_risk'].sum()} ({matched['stockout_risk'].mean()*100:.1f}%)")
    print(f"Overstock risk rows: {matched['overstock_risk'].sum()} ({matched['overstock_risk'].mean()*100:.1f}%)")
    print(f"Neither (healthy): {(~matched['stockout_risk'] & ~matched['overstock_risk']).sum()} "
          f"({(~matched['stockout_risk'] & ~matched['overstock_risk']).mean()*100:.1f}%)")

    print("\n--- Sample stockout-risk rows ---")
    print(matched[matched["stockout_risk"]][
        ["product_id", "store_id", "forecasted_demand", "stock_levels", "reorder_point", "supplier_lead_time_days"]
    ].head(10).to_string(index=False))

    out_path = PROCESSED_DIR / "forecast_inventory_flags.csv"
    matched.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()