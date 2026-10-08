"""
ulb_data_prep.py
================
Data preparation pipeline strictly for the ULB Kaggle Credit Card Fraud dataset.
Reference: Noviandy et al. (2023) - 80:20 Split, all 30 features intact.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from config import RANDOM_STATE 

ULB_DATA_FILE = "data/creditcard.csv"

def load_ulb_data(filepath: str) -> pd.DataFrame:
    """Loads the ULB dataset, maintaining all 30 features per Noviandy et al. (2023)."""
    df = pd.read_csv(filepath)
    print(f"\n[DATA PREP] Loaded raw dataset: {len(df):,} total transactions.")
    print(f"[DATA PREP] Legitimate: {sum(df['Class']==0):,} | Fraud: {sum(df['Class']==1):,}")
    return df

def split_ulb_data(df: pd.DataFrame):
    """Splits the data 80:20 stratified to match Noviandy et al. (2023)."""
    X = df.drop(columns=['Class'])
    y = df['Class']
    amounts = df['Amount']

    X_train, X_test, y_train, y_test, amounts_train, amounts_test = train_test_split(
        X, y, amounts,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"\n[DATA PREP] Splitting data 80:20 (Stratified)")
    print(f"[DATA PREP] Train Set: {len(X_train):,} rows (Fraud: {sum(y_train==1):,})")
    print(f"[DATA PREP] Test Set : {len(X_test):,} rows (Fraud: {sum(y_test==1):,})")
    
    return X_train, X_test, y_train, y_test, amounts_train, amounts_test

def apply_ulb_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """Applies SMOTE exclusively to the training data."""
    print(f"\n[DATA PREP] Applying SMOTE to training data...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train) # type: ignore
    print(f"[DATA PREP] After SMOTE: {len(X_train_res):,} total rows.")
    print(f"[DATA PREP] Legitimate: {sum(y_train_res==0):,} | Fraud: {sum(y_train_res==1):,}") # type: ignore
    return X_train_res, y_train_res

def prepare_ulb_data(apply_smote_flag: bool = False):
    """Executes the full ULB preparation pipeline."""
    df = load_ulb_data(ULB_DATA_FILE)
    
    X_train, X_test, y_train, y_test, amounts_train, amounts_test = split_ulb_data(df)
    
    if apply_smote_flag:
        X_train_res, y_train_res = apply_ulb_smote(X_train, y_train)
    else:
        print(f"\n[DATA PREP] SMOTE bypassed. Using raw imbalanced training data.")
        X_train_res, y_train_res = X_train, y_train

    return (X_train, X_test, y_train, y_test, 
            amounts_train, amounts_test, 
            X_train_res, y_train_res)