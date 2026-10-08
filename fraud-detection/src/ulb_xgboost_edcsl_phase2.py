"""
ulb_xgboost_edcsl_phase2.py
===========================
Phase 2: Global Example-Dependent Cost-Sensitive Learning (EDCSL).
Applies financial sample weights (Bahnsen et al.) to the XGBoost algorithm.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from xgboost import XGBClassifier
from config import XGBOOST_PARAMS, MODELS_DIR, ADMIN_FEE_RANDS
from utils import section, step, success, ensure_dirs
from ulb_data_prep import prepare_ulb_data
from evaluation import evaluate

def calculate_edcsl_weights(y_train: pd.Series, amounts_train: pd.Series) -> np.ndarray:
    """
    Calculates the financial weight for each transaction to feed into XGBoost.
    Legitimate (0) = Cost of Investigation (R50)
    Fraud (1) = Value of the transaction (Amount)
    Uses logarithmic scaling to prevent gradient explosion.
    """
    weights = np.zeros_like(y_train, dtype=float)
    
    mask_legit = (y_train == 0)
    mask_fraud = (y_train == 1)
    
    # Apply logarithmic scaling to the financial values
    weights[mask_legit] = np.log1p(ADMIN_FEE_RANDS)
    weights[mask_fraud] = np.log1p(amounts_train[mask_fraud])
    
    return weights

def run_ulb_phase2(use_smote: bool, phase_label: str, model_filename: str, metrics_filename: str, plot_subdir: str):
    section(f"ULB TRAINING: {phase_label}")

    # 1. Prepare Data
    step(1, 3, f"Preparing ULB Data (SMOTE = {use_smote})")
    (X_train, X_test, y_train, y_test,
     amounts_train, amounts_test,
     X_train_res, y_train_res) = prepare_ulb_data(apply_smote_flag=use_smote)

    # 2. Extract the resampled amounts and calculate financial weights
    # Because 'Amount' is still a column in X_train_res, we can easily extract it even after SMOTE.
    amounts_train_res = X_train_res['Amount']
    sample_weights = calculate_edcsl_weights(y_train_res, amounts_train_res) # type: ignore

    # 3. Train EDCSL XGBoost
    step(2, 3, "Training EDCSL XGBoost with Financial Weights")
    model = XGBClassifier(**XGBOOST_PARAMS)
    
    # The weights force the model to prioritize transactions that cost the most money
    model.fit(X_train_res, y_train_res, sample_weight=sample_weights, verbose=False)
    
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(model, model_path)
    success(f"Model saved: {model_filename}")

    # 4. Evaluate
    step(3, 3, f"Evaluating {phase_label}")
    metrics = evaluate(
        model            = model,
        X_test           = X_test,
        y_test           = y_test,
        amounts_test     = amounts_test,
        phase_label      = phase_label,
        metrics_filename = metrics_filename,
        subdir           = plot_subdir
    )
    
    return metrics

def main():
    ensure_dirs()
    
    print("\n" + "="*70)
    print("  INITIATING ULB DATASET EDCSL MATRIX (PHASE 2A & 2B)")
    print("="*70)

    # RUN 1: RAW IMBALANCED DATA (NO SMOTE)
    metrics_a = run_ulb_phase2(
        use_smote=False,
        phase_label="ULB Phase 2a (Raw Data + EDCSL)",
        model_filename="ulb_edcsl_xgboost_no_smote.pkl",
        metrics_filename="ulb_edcsl_no_smote_metrics.json",
        plot_subdir="ulb_phase2a_no_smote"
    )

    # RUN 2: SYNTHETICALLY BALANCED DATA (WITH SMOTE)
    metrics_b = run_ulb_phase2(
        use_smote=True,
        phase_label="ULB Phase 2b (SMOTE + EDCSL)",
        model_filename="ulb_edcsl_xgboost_smote.pkl",
        metrics_filename="ulb_edcsl_smote_metrics.json",
        plot_subdir="ulb_phase2b_smote"
    )

    score_2a = metrics_a["savings_score"]["savings_score_pct"]
    score_2b = metrics_b["savings_score"]["savings_score_pct"]
    
    f1_2a = metrics_a["standard_metrics"]["f1_score"]
    f1_2b = metrics_b["standard_metrics"]["f1_score"]

    print("\n" + "█"*72)
    print("  ULB EDCSL COMPARATIVE METRICS SUMMARY: PHASE 2A vs PHASE 2B")
    print("█"*72)
    print(f"  {'Metric':<25} | {'Phase 2a (Raw)':<20} | {'Phase 2b (SMOTE)':<20}")
    print("  " + "─"*70)
    print(f"  {'F1-Score':<25} | {f1_2a:<20.4f} | {f1_2b:<20.4f}")
    print(f"  {'Savings Score (%)':<25} | {score_2a:<20.4f} | {score_2b:<20.4f}")
    print("█"*72 + "\n")

if __name__ == "__main__":
    main()