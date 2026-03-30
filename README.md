## Insurance Fraud Detection Using Machine Learning

This project implements a research-grade fraud detection pipeline on the PaySim-style `dataset.csv` (~6M rows).
It covers data loading, preprocessing, feature engineering, exploratory data analysis (EDA), model training, evaluation, and model explainability.

### Project Structure

- `config/` – reserved for configuration (paths, constants).
- `eda/`
  - `eda_main.py` – end-to-end EDA script.
  - `plots/` – saved EDA and clustering plots.
- `features/`
  - `feature_engineering.py` – engineered fraud features.
- `models/`
  - `train_models.py` – training for Logistic Regression, XGBoost, Random Forest, LightGBM.
  - `saved_models/` – trained models and metadata.
- `evaluation/`
  - `evaluate_models.py` – evaluation, metrics, and explainability.
  - `plots/` – evaluation plots, metrics tables, SHAP visualisations.
- `utils/`
  - `data_loader.py` – efficient CSV loading and identifier dropping.
  - `preprocessing.py` – one-hot encoding, scaling, SMOTE helpers.
  - `visualization.py` – reusable plotting utilities for evaluation.
  - `metrics.py` – metric computation utilities.
- `dataset.csv` – raw dataset (root directory).
- `requirements.txt` – Python dependencies.

### Data Handling

- Loads `./dataset.csv` via `utils/data_loader.py`.
- Drops high-cardinality identifiers: `nameOrig`, `nameDest`.
- Applies fraud-specific feature engineering in `features/feature_engineering.py`:
  - `balanceOrigDiff`, `balanceDestDiff`
  - `errorBalanceOrig`, `errorBalanceDest`
  - `isLargeTransaction` (`amount > 200000`)
  - `transactionHour` (`step % 24`)

### Preprocessing

Implemented in `utils/preprocessing.py`:

- One-hot encodes transaction `type` (`PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEBIT`, `CASH_IN`, etc.).
- Scales numeric features with `StandardScaler`.
- Handles class imbalance via `SMOTE` (applied to training data only).

### EDA

Run:

```bash
python eda/eda_main.py
```

Key outputs in `eda/plots/`:

- **Data overview**: dataset info, basic stats, missing values heatmap, fraud distribution.
- **Distributions**: amount histogram, log-amount histogram, source/destination balance distributions.
- **Fraud analysis**: fraud vs. transaction type, fraud vs. amount, fraud vs. balances.
- **Correlation**: numeric correlation heatmap.
- **Outliers**: boxplots and IQR-based outlier summaries.
- **Feature relationships**: pairwise scatter relationships (with fraud colouring when available).
- **Clustering**:
  - KMeans clustering, elbow method (k=2–10).
  - PCA-based 2D visualisation of clusters.

### Model Training

Run:

```bash
python models/train_models.py
# or for a quick experiment on a subset, e.g. first 500k rows:
python models/train_models.py --nrows 500000
```

Pipeline:

- Loads and augments data with engineered features.
- Builds preprocessing transformer (one-hot + scaling).
- Splits data with **stratified** 70/15/15 train/validation/test.
- Applies **SMOTE** on training data to balance classes.
- Trains:
  - Logistic Regression (fast, scales to large data).
  - XGBoost (`XGBClassifier`).
  - Random Forest (`RandomForestClassifier`).
  - LightGBM (`LGBMClassifier`).
- Saves each model to `models/saved_models/` (`<name>.joblib`).
- Saves metadata (preprocessor, feature names, held-out splits) to `metadata.joblib`.

### Model Evaluation & Explainability

Run:

```bash
python evaluation/evaluate_models.py
```

Outputs in `evaluation/plots/`:

- **Metrics table** (`metrics_table.csv`) with:
  - Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
  - TPR, TNR, FPR, FNR.
- **Visualisations**:
  - Confusion matrices (per model).
  - ROC curves.
  - Precision–recall curves.
  - Model comparison bar charts across metrics.
  - Feature importance plots for tree models.
- **Explainability (SHAP)**:
  - Tree-based SHAP summary and bar plots for the first available tree model (LightGBM, XGBoost, or Random Forest).
  - Linear SHAP summary for Logistic Regression.

### Setup

1. Create and activate a Python environment (Python 3.9+ recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place `dataset.csv` in the project root (same directory as this `README.md`).

### Notes

- The dataset is large (~6M rows). Ensure you have sufficient memory, or reduce data via `nrows` in `utils/data_loader.load_raw_data` for experimentation.
- All plotting code uses sampling where appropriate to keep runtime and memory usage manageable.  Port usage is not required for this project.

```bash
python -m streamlit run streamlit_app.py
```