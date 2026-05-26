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
PAYSIM_FILE = DATA_DIR / "paysim.csv"

# 2. Data Preparation Parameters
# Locking the random state so the Train/Test split is exactly reproducible
RANDOM_STATE = 42

# Train/Test Split (80/20)
TEST_SIZE = 0.20

# Number of nearest neighbors for SMOTE oversampling
SMOTE_K = 5

# 3. Feature Engineering Parameters
# The target variable we are predicting
TARGET_COL = "isFraud"

# Columns to drop before modeling.
COLS_TO_DROP = [
    "nameOrig",
    "nameDest",
    "isFlaggedFraud"
]

# The categorical column that needs One-Hot Encoding
CATEGORY_COL = "type"

# The continuous financial columns
NUMERIC_COLS = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

# Engineered features to be created in data_prep.py
ENGINEERED_FEATURES = [
    "hour_of_day",
    # --- COMMENTED OUT TO PREVENT REVERSE-ENGINEERING THE SIMULATOR ---
    # "orig_balance_delta",
    # "dest_balance_delta",
    # "orig_zeroed_out",
    # ------------------------------------------------------------------
    "amount_to_orig_balance_ratio"
]

# Final list of continuous features (numeric + engineered)
FINAL_CONTINUOUS_FEATURES = NUMERIC_COLS + ENGINEERED_FEATURES

# 4. Segmenting Parameters
# PaySim fraud predominantly occurs in TRANSFER and CASH_OUT.
TRANSACTION_TYPES = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']

# Minimum fraud cases required in a segment to attempt training.
# Segments below this threshold are skipped with a warning to prevent SMOTE crashes.
MIN_FRAUD_CASES = 50

# 5. Evaluation Parameters
# The fixed administrative cost (in Rands) of a False Positive (blocking a real customer).
# This value is FIXED across all three phases so results are directly comparable.
ADMIN_FEE_RANDS = 50.0


# 6. Modeling Parameters
XGBOOST_PARAMS = {
    #  These hyperparameters are intentionally set to suboptimal values to create a weak baseline model. a powerful model with optimized hyperparameters would achieve much higher scores, but this is meant to be a starting point for improvement in Phases 2 and 3. Paysim data is simulated and not as complex as real-world data, so even a simple model can achieve decent performance. Paysim data is also small, so even a simple model can learn from it quickly and att the same time a model can reverse-engineer the simulator. The goal here is to have a baseline that is good enough to learn from, but still leaves significant room for improvement when we implement the more advanced techniques in the next phases.
    "n_estimators": 50,      # Dropped from 300 to 50 (forces it to stop learning early)
    "max_depth": 3,          # Dropped from 6 to 3 (stops it from building complex rules)
    "learning_rate": 0.05,   # Slowed down so it can't optimize as fast
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,  
}