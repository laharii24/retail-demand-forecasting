"""
Day 12: Hyperparameter tuning for LightGBM using Optuna.
Optimizes on WAPE (not MAPE) — consistent with the Day 10 honest-baseline standard.
"""
import numpy as np
import optuna
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

from train_lightgbm import load_data, split_features_target, evaluate

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100


def objective(trial, X_tr, y_tr, X_val, y_val, cat_cols):
    params = {
        "objective": "regression",
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "random_state": 42,
    }
    model = LGBMRegressor(**params)
    model.fit(X_tr, y_tr, categorical_feature=cat_cols)
    preds = model.predict(X_val)
    return wape(y_val, preds)


def main():
    print("Loading train/test data ...")
    train, test = load_data()

    X_train, y_train, cat_cols = split_features_target(train)
    X_test, y_test, _ = split_features_target(test)
    for col in cat_cols:
        X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

    # Carve a validation split out of TRAIN only — never touch X_test during tuning
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=False
    )

    print("Starting Optuna study (50 trials) ...")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: objective(t, X_tr, y_tr, X_val, y_val, cat_cols),
        n_trials=50,
    )

    print("\nBest params:", study.best_params)
    print(f"Best validation WAPE: {study.best_value:.3f}%")

    # Retrain on full train set with best params, evaluate ONCE on real test set
    print("\nRetraining final model on full train set ...")
    final_model = LGBMRegressor(**study.best_params, objective="regression", random_state=42)
    final_model.fit(X_train, y_train, categorical_feature=cat_cols)
    y_pred = final_model.predict(X_test)
    metrics = evaluate(y_test.values, y_pred)

    print("\n--- Day 12 tuned model vs Day 11 baseline ---")
    print(f"{'Metric':<12}{'Day 11 (baseline)':<20}{'Day 12 (tuned)'}")
    print(f"{'MAE':<12}{3.356:<20}{metrics['MAE']:.3f}")
    print(f"{'RMSE':<12}{4.717:<20}{metrics['RMSE']:.3f}")
    print(f"{'MAPE (%)':<12}{3.074:<20}{metrics['MAPE (%)']:.3f}")
    print(f"{'WAPE (%)':<12}{1.371:<20}{metrics['WAPE (%)']:.3f}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODELS_DIR / "lightgbm_tuned.pkl")
    print(f"\nSaved tuned model to {MODELS_DIR / 'lightgbm_tuned.pkl'}")


if __name__ == "__main__":
    import pandas as pd
    main()