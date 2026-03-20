from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TRANSACTION_TYPE_COL = "type"
TARGET_COL = "isFraud"


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, np.ndarray, np.ndarray]:
    """
    Build preprocessing pipeline and return transformed features and target.

    Steps:
    - One-hot encode transaction type column.
    - Standard scale numerical columns.
    - Do NOT apply SMOTE here (only fit transforms). SMOTE is applied later on train split.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in DataFrame.")

    y = df[TARGET_COL].values.astype(int)
    X = df.drop(columns=[TARGET_COL])

    categorical_cols = [c for c in [TRANSACTION_TYPE_COL] if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    transformers = []
    if categorical_cols:
        transformers.append(
            (
                "type_ohe",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )
    if numeric_cols:
        transformers.append(("num_scaler", StandardScaler(), numeric_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    X_processed = preprocessor.fit_transform(X)

    return preprocessor, X_processed, y


def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to handle class imbalance on the feature matrix.

    Important: Should be applied on the training split only.
    """
    # n_jobs is not available in all imbalanced-learn versions; rely on default parallelism.
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


__all__ = ["build_preprocessor", "apply_smote", "TRANSACTION_TYPE_COL", "TARGET_COL"]

