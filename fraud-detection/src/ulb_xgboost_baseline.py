"""
ulb_xgboost_baseline.py
=======================
Phase 1: Cost-Blind Baseline XGBoost on the ULB Dataset.
Reference: Noviandy et al. (2023)

Executes Phase 1a (Raw Imbalanced) and Phase 1b (SMOTE Resampled)
and reports full diagnostic metrics side-by-side.
"""

import os
import sys
import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from xgboost import XGBClassifier
from config import XGBOOST_PARAMS, MODELS_DIR
from utils import section, step, success, ensure_dirs
from ulb_data_prep import prepare_ulb_data
from evaluation import evaluate

def run_ulb_baseline(use_smote: bool, phase_label: str, model_filename: str, metrics_filename: str, plot_subdir: str):
    section(f"ULB TRAINING: {phase_label}")

    step(1, 3, f"Preparing ULB Data (SMOTE = {use_smote})")
    (X_train, X_test, y_train, y_test,
     amounts_train, amounts_test,
     X_train_res, y_train_res) = prepare_ulb_data(apply_smote_flag=use_smote)

    step(2, 3, "Training Standard XGBoost")
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train_res, y_train_res, verbose=False)
    
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(model, model_path)
    success(f"Model saved: {model_filename}")

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
    print("  INITIATING ULB DATASET BASELINE MATRIX (PHASE 1A & 1B)")
    print("="*70)

    # RUN 1: RAW IMBALANCED DATA (NO SMOTE)
    metrics_a = run_ulb_baseline(
        use_smote=False,
        phase_label="ULB Phase 1a (Raw Data + Cost-Blind XGBoost)",
        model_filename="ulb_base_xgboost_no_smote.pkl",
        metrics_filename="ulb_base_no_smote_metrics.json",
        plot_subdir="ulb_phase1a_no_smote"
    )

    # RUN 2: SYNTHETICALLY BALANCED DATA (WITH SMOTE)
    metrics_b = run_ulb_baseline(
        use_smote=True,
        phase_label="ULB Phase 1b (SMOTE + Cost-Blind XGBoost)",
        model_filename="ulb_base_xgboost_smote.pkl",
        metrics_filename="ulb_base_smote_metrics.json",
        plot_subdir="ulb_phase1b_smote"
    )

    score_1a = metrics_a["savings_score"]["savings_score_pct"]
    score_1b = metrics_b["savings_score"]["savings_score_pct"]
    
    f1_1a = metrics_a["standard_metrics"]["f1_score"]
    f1_1b = metrics_b["standard_metrics"]["f1_score"]

    print("\n" + "█"*72)
    print("  ULB BASELINE COMPARATIVE METRICS SUMMARY: PHASE 1A vs PHASE 1B")
    print("█"*72)
    print(f"  {'Metric':<25} | {'Phase 1a (Raw)':<20} | {'Phase 1b (SMOTE)':<20}")
    print("  " + "─"*70)
    print(f"  {'F1-Score':<25} | {f1_1a:<20.4f} | {f1_1b:<20.4f}")
    print(f"  {'Savings Score (%)':<25} | {score_1a:<20.4f} | {score_1b:<20.4f}")
    print("█"*72 + "\n")

if __name__ == "__main__":
    main()