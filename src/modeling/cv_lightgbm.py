"""
Day 13: Walk-forward cross-validation for LightGBM.
Evaluates model stability across multiple rolling time windows instead of
a single train/test split.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor

from train_lightgbm import evaluate  # reuse your MAE/RMSE/MAPE/WAPE function

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

BEST_PARAMS = {
    # paste in Day 12's best_params here so CV uses the tuned model
    "n_estimators": 653,
    "learning_rate": 0.026,
    "num_leaves": 117,
    "max_depth": 11,
    "min_child_samples": 9,
    "random_state": 42,
}


def load_full_data():
    # Combine train+test back into one time-ordered dataframe, sorted by date
    train = pd.read_csv(PROCESSED_DIR / "train.csv", parse_dates=["date"])
    test = pd.read_csv(PROCESSED_DIR / "test.csv", parse_dates=["date"])
    full = pd.concat([train, test]).sort_values("date").reset_index(drop=True)
    return full


def split_features_target(df):
    drop_cols = ["date", "product_id", "store_id", "sales_quantity"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["sales_quantity"]
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype("category")
    return X, y, cat_cols


def main():
    full = load_full_data()
    unique_dates = sorted(full["date"].unique())

    n_windows = 5  # number of rolling folds
    fold_size = len(unique_dates) // (n_windows + 1)

    results = []
    for i in range(1, n_windows + 1):
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

        model = LGBMRegressor(**BEST_PARAMS)
        model.fit(X_train, y_train, categorical_feature=cat_cols)
        preds = model.predict(X_test)
        metrics = evaluate(y_test.values, preds)
        metrics["fold"] = i
        metrics["train_size"] = len(train_fold)
        metrics["test_size"] = len(test_fold)
        results.append(metrics)

        print(f"Fold {i}: train={len(train_fold)} test={len(test_fold)} "
              f"WAPE={metrics['WAPE (%)']:.3f}% MAPE={metrics['MAPE (%)']:.3f}%")

    df_results = pd.DataFrame(results)
    print("\n--- Walk-forward CV summary ---")
    print(df_results[["fold", "MAE", "RMSE", "MAPE (%)", "WAPE (%)"]].to_string(index=False))
    print(f"\nMean WAPE: {df_results['WAPE (%)'].mean():.3f}%  (std: {df_results['WAPE (%)'].std():.3f}%)")
    print(f"Mean MAPE: {df_results['MAPE (%)'].mean():.3f}%  (std: {df_results['MAPE (%)'].std():.3f}%)")


if __name__ == "__main__":
    main()