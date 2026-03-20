from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier

# Ensure project root is on sys.path when running as a script (e.g. `python models/train_models.py`)
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from features.feature_engineering import add_fraud_features
from utils.data_loader import load_raw_data
from utils.preprocessing import apply_smote, build_preprocessor, TARGET_COL


MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)


def stratified_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.")

    sss1 = StratifiedShuffleSplit(
        n_splits=1, train_size=train_size, random_state=random_state
    )
    train_idx, temp_idx = next(sss1.split(X, y))

    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]

    val_ratio = val_size / (val_size + test_size)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, train_size=val_ratio, random_state=random_state
    )
    val_idx, test_idx = next(sss2.split(X_temp, y_temp))

    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_models(random_state: int = 42) -> Dict[str, object]:
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            solver="saga",
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_state,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    return models


def main(nrows: int | None = None) -> None:
    df = load_raw_data(nrows=nrows)
    df = add_fraud_features(df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' is required in the dataset.")

    preprocessor, X_processed, y = build_preprocessor(df)

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test_split(
        X_processed, y
    )

    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    models = build_models()

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_bal, y_train_bal)
        model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"Saved {name} model to {model_path}")

    meta = {
        "feature_names": getattr(preprocessor, "get_feature_names_out", lambda: None)(),
        "preprocessor": preprocessor,
        "splits": {
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        },
    }
    meta_path = os.path.join(MODELS_DIR, "metadata.joblib")
    joblib.dump(meta, meta_path)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Optional number of rows from dataset.csv to use for training (for quick experiments).",
    )
    args = parser.parse_args()

    main(nrows=args.nrows)

