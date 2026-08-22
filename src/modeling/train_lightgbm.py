"""
Day 11: Train a baseline LightGBM model on the Day 9 train/test split.

Loads train.csv / test.csv, trains a LightGBM regressor to predict
sales_quantity, evaluates it (MAE / RMSE / MAPE), prints feature
importance, and saves the trained model for later reuse.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- Paths (match prepare_features.py's layout) ----
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_PATH = MODELS_DIR / "lightgbm_baseline.pkl"

# Column we're trying to predict
TARGET_COL = "sales_quantity"

# Columns that should never be used as features (identifiers / raw dates)
DROP_COLS = ["date", "product_id", "store_id"]


def load_data():
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TRAIN_PATH} or {TEST_PATH}. Run "
            f"src/modeling/prepare_features.py first (Day 9)."
        )
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def split_features_target(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise KeyError(
            f"Expected target column '{TARGET_COL}' but got: {list(df.columns)}. "
            f"Update TARGET_COL in this script to match your real column name."
        )
    drop_cols = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop_cols + [TARGET_COL])
    y = df[TARGET_COL]

    # LightGBM can handle categorical (text) columns natively if we mark
    # them as pandas 'category' dtype -- no need to one-hot encode by hand.
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype("category")

    return X, y, cat_cols


def evaluate(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE: guard against divide-by-zero on rows where actual sales are 0
    nonzero = y_true != 0
    mape = (
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        if nonzero.any()
        else float("nan")
    )

    # WAPE: sums errors and actuals before dividing, so it doesn't break
    # down on individual zero-sales rows the way MAPE does. This is the
    # "honest baseline" metric established on Day 10.
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "WAPE (%)": wape}


def main():
    print("Loading train/test data ...")
    train, test = load_data()
    print(f"Train: {train.shape[0]:,} rows | Test: {test.shape[0]:,} rows")

    X_train, y_train, cat_cols = split_features_target(train)
    X_test, y_test, _ = split_features_target(test)

    # Make sure test set uses the exact same categories as train, so
    # LightGBM doesn't choke on a category it's never seen before.
    for col in cat_cols:
        X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

    print(f"\nFeatures used ({X_train.shape[1]}): {list(X_train.columns)}")
    print(f"Categorical features: {cat_cols}")

    print("\nTraining baseline LightGBM model ...")
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        categorical_feature=cat_cols,
    )

    print("\nEvaluating on test set ...")
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test.to_numpy(), y_pred)
    for name, value in metrics.items():
        print(f"  {name}: {value:.3f}")

    print("\n--- Feature importance (top 15) ---")
    importance = (
        pd.Series(model.feature_importances_, index=X_train.columns)
        .sort_values(ascending=False)
        .head(15)
    )
    print(importance.to_string())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
