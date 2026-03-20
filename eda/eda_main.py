from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Ensure project root is on sys.path when running as a script (e.g. `python eda/eda_main.py`)
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.data_loader import load_raw_data
from features.feature_engineering import add_fraud_features


PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_fig(fig, name: str) -> None:
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def data_overview(df: pd.DataFrame) -> None:
    info_path = os.path.join(PLOTS_DIR, "data_info.txt")
    with open(info_path, "w") as f:
        f.write("DATASET INFO\n")
        df.info(buf=f)
        f.write("\n\nBASIC DESCRIPTIVE STATISTICS\n")
        f.write(df.describe(include="all").to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, ax=ax)
    ax.set_title("Missing Values Heatmap")
    save_fig(fig, "missing_values_heatmap.png")

    if "isFraud" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="isFraud", ax=ax)
        ax.set_title("Fraud Distribution")
        ax.set_xticklabels(["Non-Fraud", "Fraud"])
        save_fig(fig, "fraud_distribution.png")


def distributions(df: pd.DataFrame) -> None:
    if "amount" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df["amount"], bins=100, kde=False, ax=ax)
        ax.set_title("Amount Distribution")
        save_fig(fig, "amount_histogram.png")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(np.log1p(df["amount"]), bins=100, kde=False, ax=ax)
        ax.set_title("Log(Amount + 1) Distribution")
        save_fig(fig, "log_amount_histogram.png")

    balance_cols = [c for c in df.columns if "balance" in c]
    if balance_cols:
        fig, axes = plt.subplots(len(balance_cols), 1, figsize=(8, 4 * len(balance_cols)))
        if len(balance_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, balance_cols):
            sns.histplot(df[col], bins=100, kde=False, ax=ax)
            ax.set_title(f"Distribution of {col}")
        fig.tight_layout()
        save_fig(fig, "balance_distributions.png")


def fraud_analysis(df: pd.DataFrame) -> None:
    if "isFraud" not in df.columns:
        return

    if "type" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        type_fraud = (
            df.groupby("type")["isFraud"].mean().sort_values(ascending=False)
        )
        sns.barplot(x=type_fraud.index, y=type_fraud.values, ax=ax)
        ax.set_ylabel("Fraud Rate")
        ax.set_title("Fraud Rate by Transaction Type")
        save_fig(fig, "fraud_vs_transaction_type.png")

    if "amount" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df.sample(min(len(df), 50000), random_state=42), x="isFraud", y="amount", ax=ax)
        ax.set_title("Fraud vs Amount")
        ax.set_xticklabels(["Non-Fraud", "Fraud"])
        save_fig(fig, "fraud_vs_amount_boxplot.png")

    balance_cols = [c for c in df.columns if "balance" in c]
    for col in balance_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=df.sample(min(len(df), 50000), random_state=42),
            x="isFraud",
            y=col,
            ax=ax,
        )
        ax.set_title(f"Fraud vs {col}")
        ax.set_xticklabels(["Non-Fraud", "Fraud"])
        save_fig(fig, f"fraud_vs_{col}_boxplot.png")


def correlation_analysis(df: pd.DataFrame) -> None:
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    save_fig(fig, "correlation_heatmap.png")


def outlier_analysis(df: pd.DataFrame) -> None:
    num_df = df.select_dtypes(include=[np.number])
    sample_df = num_df.sample(min(len(num_df), 50000), random_state=42)

    fig, axes = plt.subplots(
        min(4, sample_df.shape[1]),
        1,
        figsize=(8, 4 * min(4, sample_df.shape[1])),
    )
    if sample_df.shape[1] == 1:
        axes = [axes]
    for ax, col in zip(axes, sample_df.columns[:4]):
        sns.boxplot(x=sample_df[col], ax=ax)
        ax.set_title(f"Boxplot of {col}")
    fig.tight_layout()
    save_fig(fig, "outliers_boxplots.png")

    iqr_info_path = os.path.join(PLOTS_DIR, "iqr_outliers_summary.txt")
    with open(iqr_info_path, "w") as f:
        f.write("IQR-BASED OUTLIER COUNTS (approximate, per feature)\n")
        for col in sample_df.columns:
            q1 = sample_df[col].quantile(0.25)
            q3 = sample_df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = ((sample_df[col] < lower) | (sample_df[col] > upper)).sum()
            f.write(f"{col}: {outliers} outliers (sample size={len(sample_df)})\n")


def feature_relationships(df: pd.DataFrame) -> None:
    num_df = df.select_dtypes(include=[np.number])
    cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    cols = [c for c in cols if c in num_df.columns]
    if len(cols) >= 2:
        sample_df = df.sample(min(len(df), 30000), random_state=42)
        fig = sns.pairplot(
            sample_df[cols + (["isFraud"] if "isFraud" in sample_df.columns else [])],
            hue="isFraud" if "isFraud" in sample_df.columns else None,
            corner=True,
        )
        fig.fig.suptitle("Feature Relationships (Sample)", y=1.02)
        save_fig(fig.fig, "feature_relationships_pairplot.png")


def clustering_analysis(df: pd.DataFrame) -> None:
    num_df = df.select_dtypes(include=[np.number]).drop(columns=["isFraud"], errors="ignore")
    sample_df = num_df.sample(min(len(num_df), 100000), random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sample_df)

    inertias = []
    ks = range(2, 11)
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(ks), inertias, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method for Optimal k")
    save_fig(fig, "kmeans_elbow.png")

    best_k = 4
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab10", s=5, alpha=0.6)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title(f"KMeans Clusters (k={best_k}) in PCA Space")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    save_fig(fig, "kmeans_pca_clusters.png")


def main() -> None:
    df = load_raw_data()
    df = add_fraud_features(df)

    data_overview(df)
    distributions(df)
    fraud_analysis(df)
    correlation_analysis(df)
    outlier_analysis(df)
    feature_relationships(df)
    clustering_analysis(df)


if __name__ == "__main__":
    main()

