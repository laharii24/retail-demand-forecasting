"""
Day 22: SHAP explainability for the final ensemble.
Computes SHAP values for LightGBM (the ensemble's primary model) to
explain global feature importance and a few individual predictions.
"""
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from lightgbm import LGBMRegressor

from train_lightgbm import load_data, split_features_target

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
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train, categorical_feature=cat_cols)

    print("Computing SHAP values ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # --- Global feature importance (mean absolute SHAP value per feature) ---
    mean_abs_shap = pd.Series(
        np.abs(shap_values.values).mean(axis=0), index=X_test.columns
    ).sort_values(ascending=False)

    print("\n--- Global feature importance (mean |SHAP value|) ---")
    print(mean_abs_shap.to_string())

    # --- Explain a few individual predictions ---
    print("\n--- Individual prediction explanations (first 3 test rows) ---")
    for i in range(3):
        row = X_test.iloc[i]
        actual = y_test.iloc[i]
        pred = model.predict(X_test.iloc[[i]])[0]
        contributions = pd.Series(shap_values.values[i], index=X_test.columns).sort_values(
            key=abs, ascending=False
        )
        print(f"\nRow {i}: actual={actual:.1f}, predicted={pred:.1f}, base_value={shap_values.base_values[i]:.1f}")
        print("Top contributing features:")
        print(contributions.head(5).to_string())

    # --- Save summary plot ---
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend, saves to file instead of showing
    import matplotlib.pyplot as plt

    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    out_path = PROCESSED_DIR.parent / "shap_summary_plot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved SHAP summary plot to {out_path}")


if __name__ == "__main__":
    main()