import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from config import (
    PAYSIM_FILE, COLS_TO_DROP, TARGET_COL, CATEGORY_COL,
    TEST_SIZE, RANDOM_STATE, SMOTE_K
)
from utils import info, warn, success, step

# 1. Loading data
def load_data() -> pd.DataFrame:
    """
    The function loads the PaySim CSV from the specified path.
    """
    if not PAYSIM_FILE.exists():
        raise FileNotFoundError(
            f"\n❌ Dataset not found at '{PAYSIM_FILE}'.\n"
        )

    df = pd.read_csv(PAYSIM_FILE)
    info(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    info(f"Fraud rate: {df[TARGET_COL].mean()*100:.4f}% "
         f"({df[TARGET_COL].sum():,} fraudulent transactions)")
    return df

# 2. Data cleaning
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    The function drops irrelevant columns and applies Label Encoding 
    to replicate the methodology of Batsyas and Yaduwanshi (2026).
    """
    df = df.copy()

    # The algorithm drops non-predictive columns and identifiers as specified in the reference study.
    df.drop(columns=COLS_TO_DROP, errors='ignore', inplace=True)

    # The script drops any remaining null rows to ensure mathematical stability.
    before = len(df)
    df.dropna(inplace=True)
    dropped = before - len(df)
    if dropped:
        warn(f"Dropped {dropped:,} null rows")

    # The algorithm applies Label Encoding to the categorical transaction type column,
    # replicating Section 2.3.2 of the reference study.
    le = LabelEncoder()
    df[CATEGORY_COL] = le.fit_transform(df[CATEGORY_COL])
    
    info(f"Applied Label Encoding to '{CATEGORY_COL}' column.")
    info(f"Final feature set contains {df.shape[1] - 1} predictors.")
    return df

# 3. Train Test split
def split_data(df: pd.DataFrame):
    """
    The function performs a stratified 90:10 split to preserve fraud ratios.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # The script isolates a copy of the transaction amounts for the Cost Matrix evaluator.
    amounts = df['amount']

    # The algorithm splits the data using a 90:10 ratio (Batsyas and Yaduwanshi, 2026).
    X_train, X_test, y_train, y_test, amounts_train, amounts_test = train_test_split(
        X, y, amounts,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    info(f"Train set: {X_train.shape[0]:,} rows  |  Fraud: {y_train.sum():,}")
    info(f"Test set:  {X_test.shape[0]:,} rows  |  Fraud: {y_test.sum():,}")
    warn("The test set remains completely untouched to prevent data leakage.")

    return X_train, X_test, y_train, y_test, amounts_train, amounts_test

# 4. SMOTE
def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """
    The function applies SMOTE strictly to the training set to achieve a 1:1 class balance.
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

# 5. Full pipeline
def prepare_data(apply_smote_flag: bool = True):
    """
    The function executes the full data preparation pipeline in chronological sequence.
    """
    step(1, 4, "Loading raw data")
    df = load_data()
    
    step(2, 4, "Cleaning & Encoding (Batsyas & Yaduwanshi, 2026)")
    df = clean_data(df)

    step(3, 4, "Splitting Data (90:10 Stratified)")
    X_train, X_test, y_train, y_test, amounts_train, amounts_test = split_data(df)

    step(4, 4, "Resampling")
    if apply_smote_flag:
        X_train_res, y_train_res = apply_smote(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train
        warn("SMOTE skipped — utilizing the raw imbalanced training set.")

    success("Data Preparation Pipeline Complete.")
    return X_train, X_test, y_train, y_test, amounts_train, amounts_test, X_train_res, y_train_res