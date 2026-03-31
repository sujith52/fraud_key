from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path so we can reuse existing modules
ROOT_DIR = os.path.dirname(__file__)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from features.feature_engineering import add_fraud_features
from utils.data_loader import load_raw_data
from utils.preprocessing import TARGET_COL



MODELS_DIR = os.path.join(ROOT_DIR, "models", "saved_models")
EDA_PLOTS_DIR = os.path.join(ROOT_DIR, "eda", "plots")
EVAL_PLOTS_DIR = os.path.join(ROOT_DIR, "evaluation", "plots")



import hashlib

USERS_FILE = "users.csv"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists(USERS_FILE):
        return pd.DataFrame(columns=["username", "password"])
    return pd.read_csv(USERS_FILE)

def save_user(username, password):
    df = load_users()

    # prevent duplicate users
    if username in df["username"].values:
        return False

    new_user = pd.DataFrame([{
        "username": username,
        "password": hash_password(password)
    }])

    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USERS_FILE, index=False)
    return True

def authenticate(username, password):
    df = load_users()
    hashed = hash_password(password)

    user = df[(df["username"] == username) & (df["password"] == hashed)]
    return not user.empty


def inject_custom_css() -> None:
    """Inject comprehensive custom CSS for a modern, beautifully styled dashboard."""
    custom_css = """
    <style>
    /* === ROOT VARIABLES === */
    :root {
        --primary: #00d4ff;
        --secondary: #ff006e;
        --accent: #00f5ff;
        --success: #00ff88;
        --warning: #ffb700;
        --error: #ff4757;
        --dark-bg: #0a0e27;
        --darker-bg: #050812;
        --card-bg: #1a1f3a;
        --text-primary: #e8e8ff;
        --text-secondary: #9095b0;
        --border-color: #2d3561;
        --gradient-1: linear-gradient(135deg, #00d4ff 0%, #00f5ff 100%);
        --gradient-2: linear-gradient(135deg, #ff006e 0%, #ff4757 100%);
    }

    /* === GLOBAL STYLES === */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1228 100%) !important;
        color: var(--text-primary) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }

    /* === SIDEBAR STYLING === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f3a 0%, #0f1228 100%) !important;
        border-right: 2px solid var(--primary) !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: var(--text-primary) !important;
    }

    /* === HEADINGS === */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
    }

    h1 {
        font-size: 2.5rem !important;
        color: #ffffff !important; 
        background: var(--gradient-1) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        margin-bottom: 1.5rem !important;
        animation: slideInDown 0.6s ease-out;
    }

    h2 {
        font-size: 1.8rem !important;
        color: var(--accent) !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid var(--primary) !important;
        animation: slideInLeft 0.5s ease-out;
    }

    h3 {
        font-size: 1.4rem !important;
        color: var(--primary) !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.8rem !important;
    }

    /* === BUTTONS === */
    button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%) !important;
        color: #000 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        cursor: pointer !important;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.5) !important;
        background: linear-gradient(135deg, var(--accent) 0%, var(--primary) 100%) !important;
    }

    button:active {
        transform: translateY(-1px) !important;
    }

    /* Primary action buttons (Run, Login, etc) */
    [data-testid="stButton"] > button {
        width: 100% !important;
        background: var(--gradient-1) !important;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4) !important;
    }

    [data-testid="stButton"] > button:hover {
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.6) !important;
    }

    /* === TEXT INPUTS === */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    input[type="text"],
    input[type="password"],
    input[type="number"] {
        background: var(--card-bg) !important;
        border: 2px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        border-radius: 6px !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus,
    input[type="text"]:focus,
    input[type="password"]:focus,
    input[type="number"]:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
        background: rgba(0, 212, 255, 0.05) !important;
    }

    input::placeholder {
        color: var(--text-secondary) !important;
    }

    /* === DROPDOWNS/SELECTBOX === */
    [data-testid="stSelectbox"] select,
    select {
        background: linear-gradient(90deg, var(--card-bg), rgba(0, 212, 255, 0.1)) !important;
        border: 2px solid var(--primary) !important;
        color: var(--text-primary) !important;
        border-radius: 6px !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        min-height: 40px !important;
    }

    [data-testid="stSelectbox"] select:hover,
    select:hover {
        border-color: var(--accent) !important;
        box-shadow: 0 0 15px rgba(0, 245, 255, 0.3) !important;
    }

    [data-testid="stSelectbox"] select:focus,
    select:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 25px rgba(0, 245, 255, 0.5) !important;
        outline: none !important;
    }

    /* === SLIDERS === */
    [data-testid="stSlider"] {
        padding: 1rem 0 !important;
    }

    [data-testid="stSlider"] [role="slider"] {
        background: var(--primary) !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.5) !important;
    }

    /* === FILE UPLOADER === */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(255, 0, 110, 0.1)) !important;
        border: 2px dashed var(--primary) !important;
        border-radius: 10px !important;
        padding: 2rem !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent) !important;
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(255, 0, 110, 0.15)) !important;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.3) !important;
    }

    [data-testid="stFileUploader"] label {
        color: var(--primary) !important;
        font-weight: 600 !important;
    }

    /* === TABS === */
    [data-testid="stTabs"] [role="tablist"] {
        border-bottom: 2px solid var(--border-color) !important;
    }

    [data-testid="stTabs"] [role="tab"] {
        color: var(--text-secondary) !important;
        border-bottom: 3px solid transparent !important;
        transition: all 0.3s ease !important;
        padding: 1rem 1.5rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stTabs"] [role="tab"]:hover {
        color: var(--primary) !important;
        border-bottom-color: var(--primary) !important;
    }

    [data-testid="stTabs"] [aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom-color: var(--accent) !important;
        box-shadow: 0 2px 10px rgba(0, 245, 255, 0.3) !important;
    }

    /* === INFO/SUCCESS/WARNING/ERROR BOXES === */
    [data-testid="stAlert"] {
        border-radius: 8px !important;
        border-left: 4px solid !important;
        padding: 1rem !important;
        background: var(--card-bg) !important;
    }

    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
        color: var(--text-primary) !important;
    }

    /* Success (Green) */
    [data-testid="stAlert"] {
        border-left-color: var(--success) !important;
        background: linear-gradient(90deg, rgba(0, 255, 136, 0.1), transparent) !important;
    }

    /* === DATAFRAME/TABLE === */
    [data-testid="stDataFrame"] {
        border: 2px solid var(--border-color) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    [data-testid="stDataFrame"] th {
        background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
        color: #000 !important;
        font-weight: 700 !important;
    }

    [data-testid="stDataFrame"] td {
        border-color: var(--border-color) !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stDataFrame"] tr:hover {
        background: rgba(0, 212, 255, 0.1) !important;
    }

    /* === FORMS === */
    [data-testid="stForm"] {
        border: 2px solid var(--border-color) !important;
        border-radius: 10px !important;
        padding: 2rem !important;
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.05), rgba(255, 0, 110, 0.05)) !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stForm"]:hover {
        border-color: var(--primary) !important;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.15) !important;
    }

    /* === MARKDOWN TEXT === */
    [data-testid="stMarkdownContainer"] {
        color: var(--text-primary) !important;
    }

    [data-testid="stMarkdownContainer"] strong {
        color: var(--accent) !important;
        font-weight: 700 !important;
    }

    [data-testid="stMarkdownContainer"] em {
        color: var(--primary) !important;
    }

    /* === CUSTOM ANIMATIONS === */
    @keyframes slideInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }

    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }

    /* === METRIC CARDS === */
    [data-testid="stMetricContainer"] {
        background: linear-gradient(135deg, var(--card-bg), rgba(0, 212, 255, 0.05)) !important;
        border: 2px solid var(--primary) !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2) !important;
    }

    /* === SIDEBAR ELEMENTS === */
    [data-testid="stSidebar"] [data-testid="stButton"] > button {
        background: linear-gradient(135deg, var(--secondary), var(--error)) !important;
        width: 100% !important;
        margin-top: 1rem !important;
        box-shadow: 0 4px 15px rgba(255, 0, 110, 0.3) !important;
    }

    [data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
        box-shadow: 0 8px 25px rgba(255, 0, 110, 0.5) !important;
        transform: translateY(-3px) !important;
    }

    /* === SCROLLBAR STYLING === */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: var(--darker-bg);
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--primary), var(--secondary));
        border-radius: 10px;
        border: 2px solid var(--darker-bg);
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--accent), var(--secondary));
    }

    /* === RESPONSIVE === */
    @media (max-width: 768px) {
        h1 {
            font-size: 1.8rem !important;
        }

        h2 {
            font-size: 1.4rem !important;
        }

        button {
            padding: 0.6rem 1.2rem !important;
            font-size: 0.85rem !important;
        }
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def login_register():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # ✅ IF already logged in → DON'T show login page
    if st.session_state.logged_in:
        return

    menu = st.sidebar.selectbox("🔐 Menu", ["Login", "Register"])

    if menu == "Login":
        st.subheader("🔓 Login to Dashboard")

        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")

        if st.button("🚀 Login"):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Please try again.")

    else:
        st.subheader("📝 Create New Account")

        new_user = st.text_input("👤 Create Username")
        new_pass = st.text_input("🔑 Create Password", type="password")

        if st.button("✍️ Register"):
            if new_user == "" or new_pass == "":
                st.warning("⚠️ Please enter both username and password")
            else:
                if save_user(new_user, new_pass):
                    st.success("✅ Account registered successfully!")
                else:
                    st.error("❌ User already exists. Please try a different username.")

@st.cache_resource(show_spinner=False)
def load_metadata_and_models() -> tuple[Optional[dict], Dict[str, object]]:
    """Load saved metadata (preprocessor, splits) and trained models."""
    meta_path = os.path.join(MODELS_DIR, "metadata.joblib")
    meta: Optional[dict] = None
    if os.path.exists(meta_path):
        meta = joblib.load(meta_path)

    models: Dict[str, object] = {}
    for name in ["lightgbm", "xgboost", "random_forest", "logistic_regression"]:
        model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        if os.path.exists(model_path):
            models[name] = joblib.load(model_path)

    return meta, models


@st.cache_data(show_spinner=False)
def load_metrics_table() -> Optional[pd.DataFrame]:
    metrics_path = os.path.join(EVAL_PLOTS_DIR, "metrics_table.csv")
    if not os.path.exists(metrics_path):
        return None
    return pd.read_csv(metrics_path, index_col=0)


@st.cache_data(show_spinner=False)
def load_sample_data(nrows: int = 10000) -> Optional[pd.DataFrame]:
    """
    Load a small sample of the original dataset for demos / auto-fill.
    """
    dataset_path = os.path.join(ROOT_DIR, "dataset.csv")
    if not os.path.exists(dataset_path):
        return None
    try:
        df = load_raw_data(data_path=dataset_path, nrows=nrows)
        return df
    except Exception:
        return None


def render_eda_section() -> None:
    st.subheader("📊 Exploratory Data Analysis (EDA)")
    st.markdown(
        "This section displays comprehensive EDA plots generated by `eda/eda_main.py`. "
        "Analyze data distributions, correlations, and patterns."
    )

    eda_files = [
        ("Fraud distribution", "fraud_distribution.png"),
        ("Amount distribution", "amount_histogram.png"),
        ("Log amount distribution", "log_amount_histogram.png"),
        ("Balance distributions", "balance_distributions.png"),
        ("Fraud vs transaction type", "fraud_vs_transaction_type.png"),
        ("Fraud vs amount", "fraud_vs_amount_boxplot.png"),
        ("Correlation heatmap", "correlation_heatmap.png"),
        ("Outliers (boxplots)", "outliers_boxplots.png"),
        ("KMeans elbow curve", "kmeans_elbow.png"),
        ("KMeans PCA clusters", "kmeans_pca_clusters.png"),
        ("Feature relationships (pairplot)", "feature_relationships_pairplot.png"),
    ]

    available_plots = [
        (title, os.path.join(EDA_PLOTS_DIR, fname))
        for title, fname in eda_files
        if os.path.exists(os.path.join(EDA_PLOTS_DIR, fname))
    ]

    if not available_plots:
        st.info(
            "📁 No EDA plots found. Run `python eda/eda_main.py` first to generate visualizations."
        )
        return

    for title, path in available_plots:
        st.markdown(f"**{title}**")
        st.image(path)


def render_evaluation_section() -> None:
    st.subheader("⚖️ Model Evaluation & Performance")
    st.markdown(
        "View detailed model evaluation metrics and comparison plots from "
        "`evaluation/evaluate_models.py`."
    )

    metrics_df = load_metrics_table()
    if metrics_df is None:
        st.info(
            "📊 Metrics table not found. Run `python evaluation/evaluate_models.py` "
            "after training models to generate evaluation results."
        )
    else:
        st.markdown("**📋 Metrics table (test set)**")
        st.dataframe(metrics_df.style.format("{:.4f}"))

    # High-level comparison plots
    comparison_plots = [
        ("ROC curves", "roc_curves.png"),
        ("Precision–Recall curves", "precision_recall_curves.png"),
        ("Accuracy comparison", "model_comparison_accuracy.png"),
        ("Precision comparison", "model_comparison_precision.png"),
        ("Recall comparison", "model_comparison_recall.png"),
        ("F1-score comparison", "model_comparison_f1.png"),
        ("ROC AUC comparison", "model_comparison_roc_auc.png"),
        ("PR AUC comparison", "model_comparison_pr_auc.png"),
    ]

    available_eval_plots = [
        (title, os.path.join(EVAL_PLOTS_DIR, fname))
        for title, fname in comparison_plots
        if os.path.exists(os.path.join(EVAL_PLOTS_DIR, fname))
    ]

    if available_eval_plots:
        st.markdown("**🏆 Global evaluation plots**")
        for title, path in available_eval_plots:
            st.markdown(f"**{title}**")
            st.image(path)

    # Per-model confusion matrices
    st.markdown("**🎯 Confusion matrices (per model)**")
    for model_name in ["logistic_regression", "random_forest", "xgboost", "lightgbm"]:
        cm_path = os.path.join(EVAL_PLOTS_DIR, model_name, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.markdown(f"**{model_name.replace('_', ' ').title()}**")
            st.image(cm_path, use_column_width=True)

    # SHAP explainability plots (if available)
    shap_dir = os.path.join(EVAL_PLOTS_DIR, "shap")
    if os.path.isdir(shap_dir):
        shap_files = [
            f for f in os.listdir(shap_dir) if f.endswith(".png")
        ]
        if shap_files:
            st.markdown("**🔍 SHAP Explainability Plots**")
            for fname in shap_files:
                st.markdown(f"`{fname}`")
                st.image(os.path.join(shap_dir, fname))


def prepare_inference_features(
    df_raw: pd.DataFrame, meta: Optional[dict]
) -> tuple[np.ndarray, Optional[np.ndarray], pd.DataFrame]:
    """
    Apply feature engineering and preprocessing to a raw DataFrame for inference.

    Returns:
        X_processed: transformed features ready for model.predict / predict_proba
        y_true: optional ground-truth labels if present
        df_with_features: engineered feature DataFrame aligned with training schema
    """
    if meta is None:
        raise RuntimeError("Model metadata not found. Train models before prediction.")

    preprocessor = meta.get("preprocessor")
    if preprocessor is None:
        raise RuntimeError("Preprocessor not found in metadata.")

    df = df_raw.copy()

    # Optional ground truth if user provided labels
    y_true: Optional[np.ndarray] = None
    if TARGET_COL in df.columns:
        y_true = df[TARGET_COL].values.astype(int)

    # Apply the same feature engineering used during training
    df = add_fraud_features(df)

    # Drop target column for preprocessing, if present
    if TARGET_COL in df.columns:
        df_features = df.drop(columns=[TARGET_COL])
    else:
        df_features = df

    # Use the fitted preprocessor from training
    X_processed = preprocessor.transform(df_features)
    return X_processed, y_true, df_features


def render_prediction_section(meta: Optional[dict], models: Dict[str, object]) -> None:
    st.subheader("🚨 Fraud Prediction & Risk Analysis")
    st.markdown(
        "Run intelligent fraud risk predictions using trained models on batch transactions "
        "(CSV) or individual transactions."
    )

    if not models:
        st.warning(
            "⚠️ No trained models found in `models/saved_models/`. "
            "Run `python models/train_models.py` first."
        )
        return

    default_model_order = ["lightgbm", "xgboost", "random_forest", "logistic_regression"]
    available_model_names = [m for m in default_model_order if m in models]
    if not available_model_names:
        available_model_names = list(models.keys())

    col1, col2 = st.columns(2)
    
    with col1:
        selected_model_name = st.selectbox(
            "🤖 Select prediction model",
            options=available_model_names,
            index=0,
        )
    
    with col2:
        threshold = st.slider(
            "🎚️ Fraud probability threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.5,
            step=0.05,
        )

    tab_batch, tab_single = st.tabs(["📦 Batch CSV", "📝 Single Entry"])

    # --- Batch CSV tab ---
    with tab_batch:
        col_upload, col_sample = st.columns(2)
        with col_upload:
            uploaded_file = st.file_uploader(
                "📤 Upload CSV with transaction data",
                type=["csv"],
                key="batch_csv_uploader",
            )
        with col_sample:
            use_sample = False

        df_raw: Optional[pd.DataFrame] = None

        if use_sample:
            sample_df = load_sample_data(nrows=5000)
            if sample_df is None:
                st.error("Could not load sample from `dataset.csv`. Make sure the file exists.")
                return
            df_raw = sample_df
            st.success("✅ Loaded 5,000 sample rows from `dataset.csv`.")
        elif uploaded_file is not None:
            try:
                df_raw = pd.read_csv(uploaded_file)
            except Exception as exc:
                st.error(f"❌ Failed to read CSV: {exc}")
                return

        if df_raw is None:
            st.info(
                "📌 Upload a CSV file or click 'Load sample from dataset.csv' to run batch predictions."
            )
        else:
            st.markdown("**📋 Preview of batch data (top 5 rows)**")
            st.dataframe(df_raw.head())

            if st.button("▶️ Run batch prediction"):
                try:
                    X_processed, y_true, df_features = prepare_inference_features(df_raw, meta)
                except Exception as exc:
                    st.error(f"❌ Error preparing features: {exc}")
                    return

                model = models[selected_model_name]
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_processed)[:, 1]
                elif hasattr(model, "decision_function"):
                    scores = model.decision_function(X_processed)
                    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                    y_proba = scores
                else:
                    y_pred_raw = model.predict(X_processed)
                    y_proba = y_pred_raw.astype(float)

                y_pred = (y_proba >= threshold).astype(int)

                result_df = df_features.copy()
                result_df["fraud_probability"] = y_proba
                result_df["fraud_prediction"] = y_pred

                if y_true is not None and len(y_true) == len(result_df):
                    result_df[TARGET_COL] = y_true

                st.markdown("**✅ Batch prediction results (top 1000 rows)**")
                st.dataframe(result_df.sample(min(1000, len(result_df))))
                st.download_button(
                    "⬇️ Download Full Results",
                    result_df.to_csv(index=False),
                    "predictions.csv",
                    "text/csv"
                )

                st.markdown("**📊 Summary**")
                col_total, col_fraud = st.columns(2)
                with col_total:
                    st.metric("Total Records Scored", len(result_df))
                with col_fraud:
                    st.metric(
                        f"Predicted Fraud (≥ {threshold:.2f})",
                        int(result_df['fraud_prediction'].sum())
                    )

    # --- Single entry tab ---
    with tab_single:
        st.markdown(
            "Use the form below to enter a single transaction. "
            "Auto-fill from a random dataset row and edit fields as needed."
        )

        sample_df = load_sample_data()

        if "single_entry_values" not in st.session_state:
            if sample_df is not None and not sample_df.empty:
                st.session_state.single_entry_values = sample_df.iloc[0].to_dict()
            else:
                st.session_state.single_entry_values = {}

        if st.button("🔄 Auto-fill from dataset.csv"):
            if sample_df is None or sample_df.empty:
                st.error("Could not load sample from `dataset.csv` for auto-fill.")
            else:
                random_row = sample_df.sample(1).iloc[0]
                st.session_state.single_entry_values = random_row.to_dict()
                st.success("✅ Auto-filled fields from a random row.")

        values = st.session_state.single_entry_values

        if sample_df is None or sample_df.empty:
            st.info(
                "💡 For best results, place `dataset.csv` in the project root so the form can "
                "use real column names and value ranges."
            )
            return

        # Build form fields dynamically from sample_df schema
        with st.form("single_entry_form"):
            form_inputs: Dict[str, object] = {}

            for col in sample_df.columns:
                # Ground-truth label is optional in single-entry mode
                if col == TARGET_COL:
                    current = values.get(col, "")
                    label_option = st.selectbox(
                        f"{col} (optional ground truth)",
                        options=["(none)", 0, 1],
                        index=0 if current == "" else (2 if current == 1 else 1),
                    )
                    form_inputs[col] = None if label_option == "(none)" else int(label_option)
                    continue

                dtype = sample_df[col].dtype
                fallback_value = sample_df[col].iloc[0] if sample_df is not None and not sample_df.empty else None
                current = values.get(col, fallback_value)

                if np.issubdtype(dtype, np.number):
                    default_val = float(current) if current is not None else 0.0
                    form_inputs[col] = st.number_input(
                        col,
                        value=default_val,
                    )
                else:
                    unique_vals = None
                    try:
                        unique_vals = sorted(
                            v for v in sample_df[col].dropna().unique().tolist() if v != ""
                        )
                    except Exception:
                        unique_vals = None

                    if unique_vals:
                        default_index = 0
                        if current in unique_vals:
                            default_index = unique_vals.index(current)
                        form_inputs[col] = st.selectbox(
                            col,
                            options=unique_vals,
                            index=default_index,
                        )
                    else:
                        form_inputs[col] = st.text_input(
                            col,
                            value=str(current) if current is not None else "",
                        )

            submitted = st.form_submit_button("⚡ Run single-entry prediction")

        if submitted:
            st.session_state.single_entry_values = form_inputs

            df_single = pd.DataFrame([form_inputs])
            try:
                X_processed, y_true, df_features = prepare_inference_features(df_single, meta)
            except Exception as exc:
                st.error(f"❌ Error preparing features: {exc}")
                return

            model = models[selected_model_name]
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_processed)[:, 1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X_processed)
                scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
                y_proba = scores
            else:
                y_pred_raw = model.predict(X_processed)
                y_proba = y_pred_raw.astype(float)

            prob = float(y_proba[0])
            pred = int(prob >= threshold)

            st.markdown("**🎯 Single-entry prediction result**")
            col_prob, col_pred = st.columns(2)
            
            with col_prob:
                st.metric("Fraud Probability", f"{prob:.4f}")
            with col_pred:
                status = "🚨 FRAUD" if pred == 1 else "✅ LEGITIMATE"
                st.metric(f"Prediction (threshold {threshold:.2f})", status)

            if y_true is not None and len(y_true) == 1 and y_true[0] is not None:
                st.write(f"**Ground Truth:** `isFraud = {int(y_true[0])}`")


def main() -> None:
    st.set_page_config(
        page_title="Transaction Fraud Detection Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 🎨 Inject custom CSS
    inject_custom_css()

    # 🔐 Step 1: Show login/register page
    login_register()

    # 🚫 Step 2: Stop if not logged in
    if not st.session_state.get("logged_in", False):
        st.stop()

    # ✅ Step 3: Dashboard starts AFTER login
    st.title("🔍 Transaction Fraud Detection – Interactive Dashboard")
    st.markdown(
        "Explore comprehensive EDA results, inspect model evaluation metrics, and run intelligent fraud predictions on transaction data."
    )

    # 🔓 Logout button
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

    meta, models = load_metadata_and_models()

    section = st.sidebar.radio(
        "📌 Navigation",
        options=["📊 EDA", "⚖️ Evaluation", "🚨 Predict"],
        index=0,
    )

    if section == "📊 EDA":
        render_eda_section()
    elif section == "⚖️ Evaluation":
        render_evaluation_section()
    else:
        render_prediction_section(meta, models)


if __name__ == "__main__":
    main()