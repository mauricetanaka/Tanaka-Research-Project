import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from config import (
    PAYSIM_FILE, COLS_TO_DROP, TARGET_COL, CATEGORY_COL,
    TEST_SIZE, RANDOM_STATE, SMOTE_K
)
from utils import info, warn, success, step

# ─────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Load the PaySim CSV from the path defined in config.py.
    """
    if not PAYSIM_FILE.exists():
        raise FileNotFoundError(
            f"\n❌ Dataset not found at '{PAYSIM_FILE}'.\n"
            "   Download from: https://www.kaggle.com/datasets/ealaxi/paysim1\n"
            "   Place the file at: data/paysim.csv"
        )

    df = pd.read_csv(PAYSIM_FILE)
    info(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    info(f"Fraud rate: {df[TARGET_COL].mean()*100:.4f}% "
         f"({df[TARGET_COL].sum():,} fraudulent transactions)")
    info(f"Transaction types: {sorted(df[CATEGORY_COL].unique().tolist())}")
    return df


# 2. data cleaning

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop irrelevant columns, handle nulls, and apply One-Hot Encoding
    to the categorical transaction type column.
    """
    df = df.copy()

    # Drop non-predictive columns and identifiers to prevent Data Leakage
    df.drop(columns=COLS_TO_DROP, errors='ignore', inplace=True)

    # Drop null rows
    before = len(df)
    df.dropna(inplace=True)
    dropped = before - len(df)
    if dropped:
        warn(f"Dropped {dropped:,} null rows")
    else:
        info("No null rows found")

    # One-Hot Encoding.
    df = pd.get_dummies(df, columns=[CATEGORY_COL], dtype=int)
    
    # Extract the new one-hot column names to display them
    encoded_cols = [c for c in df.columns if c.startswith(f"{CATEGORY_COL}_")]
    info(f"Applied One-Hot Encoding. New columns: {encoded_cols}")

    info(f"Clean dataset: {df.shape[0]:,} rows")
    return df

# 3. Feature selection(engineering)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features that capture fraud-relevant signals.
    """
    df = df.copy()

    df['orig_balance_delta'] = (
        df['newbalanceOrig'] - (df['oldbalanceOrg'] - df['amount'])
    )

    df['dest_balance_delta'] = (
        df['newbalanceDest'] - (df['oldbalanceDest'] + df['amount'])
    )

    df['orig_zeroed_out'] = (
        ((df['newbalanceOrig'] == 0) & (df['oldbalanceOrg'] > 0)).astype(int)
    )

    df['amount_to_orig_balance_ratio'] = (
        df['amount'] / (df['oldbalanceOrg'] + 1)
    )

    df['hour_of_day'] = df['step'] % 24

    info("Engineered features added:")
    info("  orig_balance_delta, dest_balance_delta,")
    info("  orig_zeroed_out, amount_to_orig_balance_ratio, hour_of_day")

    return df


# 4. Train Test split

def split_data(df: pd.DataFrame):
    """
    Stratified split to preserve fraud ratios. Extracts 'amount' separately 
    for the Savings Score metric.
    """
    # Dynamically select all features except the target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # Keep an isolated copy of the transaction amounts for the Cost Matrix evaluator
    amounts = df['amount']

    X_train, X_test, y_train, y_test, amounts_train, amounts_test = train_test_split(
        X, y, amounts,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    info(f"Train set: {X_train.shape[0]:,} rows  |  Fraud: {y_train.sum():,}")
    info(f"Test set:  {X_test.shape[0]:,} rows  |  Fraud: {y_test.sum():,}")
    warn("Test set remains UNTOUCHED throughout training — no data leakage.")

    return X_train, X_test, y_train, y_test, amounts_train, amounts_test


# 5. SMOTE
def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Apply SMOTE to the TRAINING SET ONLY to prevent overlap leakage.
    """
    before_legit = (y_train == 0).sum()
    before_fraud = (y_train == 1).sum()

    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    after_legit = (y_resampled == 0).sum()
    after_fraud = (y_resampled == 1).sum()

    info(f"Before SMOTE — Legitimate: {before_legit:,}  |  Fraud: {before_fraud:,}")
    info(f"After  SMOTE — Legitimate: {after_legit:,}  |  Fraud: {after_fraud:,}")

    return X_resampled, y_resampled

# 6. full pipeline in one call

def prepare_data(apply_smote_flag: bool = True):
    """
    Run the full data preparation pipeline in sequence.
    """
    step(1, 4, "Loading raw data")
    df = load_data()
    
    step(2, 4, "Cleaning & Encoding")
    df = clean_data(df)
    
    step(3, 4, "Engineering Features")
    df = engineer_features(df)

    step(4, 4, "Splitting & Oversampling")
    X_train, X_test, y_train, y_test, amounts_train, amounts_test = split_data(df)

    if apply_smote_flag:
        X_train_res, y_train_res = apply_smote(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train
        warn("SMOTE skipped — using raw imbalanced training set")

    success("Data Preparation Pipeline Complete.")
    return X_train, X_test, y_train, y_test, amounts_train, amounts_test, X_train_res, y_train_res