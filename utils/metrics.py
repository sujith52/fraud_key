from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn import metrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> Dict[str, float]:
    """
    Compute standard classification metrics for binary fraud detection.
    """
    acc = metrics.accuracy_score(y_true, y_pred)
    prec = metrics.precision_score(y_true, y_pred, zero_division=0)
    rec = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)

    if y_proba is not None:
        roc_auc = metrics.roc_auc_score(y_true, y_proba)
        precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_true, y_proba)
        pr_auc = metrics.auc(recall_curve, precision_curve)
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    tpr, tnr, fpr, fnr = compute_rate_metrics(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
    }


def compute_rate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute TPR, TNR, FPR, FNR from confusion matrix.
    """
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall / sensitivity
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # specificity
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return tpr, tnr, fpr, fnr


__all__ = ["compute_classification_metrics", "compute_rate_metrics"]

