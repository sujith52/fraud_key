from __future__ import annotations

import os
import sys
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import shap

# Ensure project root is on sys.path when running as a script (e.g. `python evaluation/evaluate_models.py`)
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.metrics import compute_classification_metrics
from utils.visualization import (
    plot_confusion_matrix,
    plot_feature_importances,
    plot_model_comparison_bar,
    plot_precision_recall_curves,
    plot_roc_curves,
)


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved_models")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def load_models() -> Dict[str, object]:
    models = {}
    for name in ["logistic_regression", "xgboost", "random_forest", "lightgbm"]:
        path = os.path.join(MODELS_DIR, f"{name}.joblib")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    if not models:
        raise RuntimeError("No trained models found in 'models/saved_models/'. Run train_models.py first.")
    return models


def main() -> None:
    meta_path = os.path.join(MODELS_DIR, "metadata.joblib")
    if not os.path.exists(meta_path):
        raise RuntimeError("Metadata file not found. Run train_models.py first.")

    meta = joblib.load(meta_path)
    feature_names = meta.get("feature_names")
    preprocessor = meta.get("preprocessor")
    splits = meta.get("splits")

    X_test: np.ndarray = splits["X_test"]
    y_test: np.ndarray = splits["y_test"]

    models = load_models()

    metrics_table: Dict[str, Dict[str, float]] = {}
    y_probas: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        print(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            from sklearn.metrics import roc_auc_score

            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            y_proba = scores
        else:
            y_proba = y_pred.astype(float)

        y_probas[name] = y_proba
        metrics_table[name] = compute_classification_metrics(y_test, y_pred, y_proba)

        cm_dir = os.path.join(PLOTS_DIR, name)
        os.makedirs(cm_dir, exist_ok=True)
        plot_confusion_matrix(y_test, y_pred, save_dir=cm_dir, title=f"{name} - Confusion Matrix")

    metrics_df = pd.DataFrame.from_dict(metrics_table, orient="index")
    metrics_csv_path = os.path.join(PLOTS_DIR, "metrics_table.csv")
    metrics_df.to_csv(metrics_csv_path)
    print(f"Saved metrics table to {metrics_csv_path}")

    plot_roc_curves(y_test, y_probas, save_dir=PLOTS_DIR)
    plot_precision_recall_curves(y_test, y_probas, save_dir=PLOTS_DIR)

    for m in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        plot_model_comparison_bar(metrics_table, metric_key=m, save_dir=PLOTS_DIR)

    if feature_names is None and hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out()

    for name in ["xgboost", "random_forest", "lightgbm"]:
        model = models.get(name)
        if model is None or feature_names is None:
            continue
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            continue
        fi_dir = os.path.join(PLOTS_DIR, name)
        os.makedirs(fi_dir, exist_ok=True)
        plot_feature_importances(
            feature_names=list(feature_names),
            importances=importances,
            save_dir=fi_dir,
            title=f"{name} Feature Importances",
        )

    explainer_plots_dir = os.path.join(PLOTS_DIR, "shap")
    os.makedirs(explainer_plots_dir, exist_ok=True)

    shap_sample_size = min(2000, X_test.shape[0])
    shap_idx = np.random.RandomState(42).choice(X_test.shape[0], shap_sample_size, replace=False)
    X_shap = X_test[shap_idx]

    tree_model_name = None
    for candidate in ["lightgbm", "xgboost", "random_forest"]:
        if candidate in models:
            tree_model_name = candidate
            break

    if tree_model_name:
        model = models[tree_model_name]
        print(f"Computing SHAP values for {tree_model_name}...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

        try:
            shap.summary_plot(
                shap_values if isinstance(shap_values, np.ndarray) else shap_values[1],
                X_shap,
                feature_names=feature_names,
                show=False,
            )
            shap_summary_path = os.path.join(explainer_plots_dir, f"{tree_model_name}_shap_summary.png")
            import matplotlib.pyplot as plt

            plt.tight_layout()
            plt.savefig(shap_summary_path, dpi=200, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Failed to create SHAP summary plot for {tree_model_name}: {e}")

        try:
            shap.summary_plot(
                shap_values if isinstance(shap_values, np.ndarray) else shap_values[1],
                X_shap,
                feature_names=feature_names,
                plot_type="bar",
                show=False,
            )
            shap_bar_path = os.path.join(explainer_plots_dir, f"{tree_model_name}_shap_importance.png")
            import matplotlib.pyplot as plt

            plt.tight_layout()
            plt.savefig(shap_bar_path, dpi=200, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Failed to create SHAP bar plot for {tree_model_name}: {e}")

    lr_model = models.get("logistic_regression")
    if lr_model is not None:
        try:
            print("Computing SHAP values for Logistic Regression (LinearExplainer)...")
            import matplotlib.pyplot as plt

            lr_explainer = shap.LinearExplainer(lr_model, X_shap)
            lr_shap_values = lr_explainer.shap_values(X_shap)

            shap.summary_plot(
                lr_shap_values,
                X_shap,
                feature_names=feature_names,
                show=False,
            )
            lr_shap_path = os.path.join(explainer_plots_dir, "logistic_regression_shap_summary.png")
            plt.tight_layout()
            plt.savefig(lr_shap_path, dpi=200, bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"Failed to compute SHAP for Logistic Regression: {e}")


if __name__ == "__main__":
    main()

