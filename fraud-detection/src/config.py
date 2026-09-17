from pathlib import Path

# 1. Directory Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = ROOT_DIR / "reports"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# Input Data File
PAYSIM_FILE = DATA_DIR / "PS_20174392719_1491204439457_log.csv"

# 2. Data Preparation Parameters
# Locking the random state so the Train/Test split is exactly reproducible.
RANDOM_STATE = 42

# The dataset is split using a 90:10 ratio, exactly as specified in Batsyas and Yaduwanshi (2026).
TEST_SIZE = 0.10

# Number of nearest neighbors for SMOTE oversampling.
SMOTE_K = 5

# 3. Feature Engineering Parameters
# The target variable the algorithm is predicting.
TARGET_COL = "isFraud"

# The script drops these specific columns to replicate the feature selection 
# process in Section 2.3.1 of Batsyas and Yaduwanshi (2026).
COLS_TO_DROP = [
    "step",
    "nameOrig",
    "nameDest",
    "isFlaggedFraud"
]

# The categorical column that requires Label Encoding.
CATEGORY_COL = "type"

# The final list of continuous financial columns retained by the reference study.
NUMERIC_COLS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

# 4. Segmenting Parameters
TRANSACTION_TYPES = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
MIN_FRAUD_CASES = 50

# 5. Evaluation Parameters
# The fixed administrative cost (in Rands) of a False Positive (blocking a real customer).
ADMIN_FEE_RANDS = 50.0

# 6. Modeling Parameters
# XGBoost hyperparameters specified in Section 2.4.2 
# of Batsyas and Yaduwanshi (2026) to establish a scientifically valid baseline.
XGBOOST_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.1,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}