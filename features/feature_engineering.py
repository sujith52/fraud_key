from __future__ import annotations

import numpy as np
import pandas as pd


def add_fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered fraud-related features to the DataFrame.

    New features:
    - balanceOrigDiff = oldbalanceOrg - newbalanceOrig
    - balanceDestDiff = newbalanceDest - oldbalanceDest
    - errorBalanceOrig = oldbalanceOrg - amount - newbalanceOrig
    - errorBalanceDest = oldbalanceDest + amount - newbalanceDest
    - isLargeTransaction = (amount > 200000).astype(int)
    - transactionHour = step % 24
    """
    df = df.copy()

    required_cols = [
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "amount",
        "step",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {missing}")

    df["balanceOrigDiff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balanceDestDiff"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["errorBalanceOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    df["isLargeTransaction"] = (df["amount"] > 200000).astype(int)
    df["transactionHour"] = df["step"] % 24

    return df


__all__ = ["add_fraud_features"]

