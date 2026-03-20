from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: str,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
) -> str:
    _ensure_dir(save_dir)
    cm = metrics.confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["Non-Fraud", "Fraud"],
        yticklabels=["Non-Fraud", "Fraud"],
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path = os.path.join(save_dir, "confusion_matrix.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_roc_curves(
    y_true: np.ndarray,
    y_probas: Dict[str, np.ndarray],
    save_dir: str,
    title: str = "ROC Curves",
) -> str:
    _ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, proba in y_probas.items():
        fpr, tpr, _ = metrics.roc_curve(y_true, proba)
        roc_auc = metrics.auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path = os.path.join(save_dir, "roc_curves.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_probas: Dict[str, np.ndarray],
    save_dir: str,
    title: str = "Precision-Recall Curves",
) -> str:
    _ensure_dir(save_dir)
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, proba in y_probas.items():
        precision, recall, _ = metrics.precision_recall_curve(y_true, proba)
        pr_auc = metrics.auc(recall, precision)
        ax.plot(recall, precision, label=f"{model_name} (AUC = {pr_auc:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    out_path = os.path.join(save_dir, "precision_recall_curves.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_model_comparison_bar(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_key: str,
    save_dir: str,
    title: Optional[str] = None,
) -> str:
    _ensure_dir(save_dir)
    model_names = list(metrics_dict.keys())
    values = [metrics_dict[m][metric_key] for m in model_names]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(model_names, values, color="steelblue")
    ax.set_ylabel(metric_key)
    ax.set_title(title or f"Model Comparison ({metric_key})")
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    fig.tight_layout()
    out_path = os.path.join(save_dir, f"model_comparison_{metric_key}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_feature_importances(
    feature_names: List[str],
    importances: np.ndarray,
    save_dir: str,
    title: str = "Feature Importances",
    top_n: Optional[int] = 20,
) -> str:
    _ensure_dir(save_dir)
    idx = np.argsort(importances)[::-1]
    if top_n is not None:
        idx = idx[:top_n]
    sorted_names = [feature_names[i] for i in idx]
    sorted_importances = importances[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, len(sorted_names) * 0.25)))
    ax.barh(sorted_names[::-1], sorted_importances[::-1], color="darkgreen")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    out_path = os.path.join(save_dir, "feature_importances.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


__all__ = [
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_precision_recall_curves",
    "plot_model_comparison_bar",
    "plot_feature_importances",
]

