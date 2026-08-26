"""
Day 15: Feature selection — drop low-importance features and confirm
performance holds with a simpler, leaner feature set.
"""
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor

from train_lightgbm import load_data, split_features_target, evaluate

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

BEST_PARAMS = {
    "n_estimators": 653,
    "learning_rate": 0.026,
    "num_leaves": 117,
    "max_depth": 11,
    "min_child_samples": 9,
    "random_state": 42,
}

# Features with importance below this fraction of the TOP feature's
# importance get dropped. 0.02 = drop anything under 2% of the top score.
IMPORTANCE_THRESHOLD = 0.02


def train_and_get_importance(X_train, y_train, X_test, y_test, cat_cols):
    model = LGBMRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train, categorical_feature=cat_cols)
    preds = model.predict(X_test)
    metrics = evaluate(y_test.values, preds)

    importance = pd.Series(model.feature_importances_, index=X_train.columns)
    importance = importance.sort_values(ascending=False)
    return model, metrics, importance


def main():
    print("Loading train/test data ...")
    train, test = load_data()
    X_train, y_train, cat_cols = split_features_target(train)
    X_test, y_test, _ = split_features_target(test)
    for col in cat_cols:
        X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

    print("\n--- Training with ALL features ---")
    model_full, metrics_full, importance = train_and_get_importance(
        X_train, y_train, X_test, y_test, cat_cols
    )
    print(f"WAPE (all features): {metrics_full['WAPE (%)']:.3f}%")
    print(f"\nFeature importance (all {len(importance)} features):")
    print(importance.to_string())

    # Drop features below the importance threshold
    top_importance = importance.max()
    keep_features = importance[importance >= top_importance * IMPORTANCE_THRESHOLD].index.tolist()
    dropped_features = importance[importance < top_importance * IMPORTANCE_THRESHOLD].index.tolist()

    print(f"\nDropping {len(dropped_features)} low-importance features: {dropped_features}")
    print(f"Keeping {len(keep_features)} features: {keep_features}")

    X_train_reduced = X_train[keep_features]
    X_test_reduced = X_test[keep_features]
    cat_cols_reduced = [c for c in cat_cols if c in keep_features]

    print("\n--- Training with REDUCED features ---")
    model_reduced, metrics_reduced, importance_reduced = train_and_get_importance(
        X_train_reduced, y_train, X_test_reduced, y_test, cat_cols_reduced
    )

    print("\n--- Comparison: all features vs reduced ---")
    print(f"{'Metric':<12}{'All features':<16}{'Reduced'}")
    print(f"{'MAE':<12}{metrics_full['MAE']:<16.3f}{metrics_reduced['MAE']:.3f}")
    print(f"{'RMSE':<12}{metrics_full['RMSE']:<16.3f}{metrics_reduced['RMSE']:.3f}")
    print(f"{'MAPE (%)':<12}{metrics_full['MAPE (%)']:<16.3f}{metrics_reduced['MAPE (%)']:.3f}")
    print(f"{'WAPE (%)':<12}{metrics_full['WAPE (%)']:<16.3f}{metrics_reduced['WAPE (%)']:.3f}")


if __name__ == "__main__":
    main()