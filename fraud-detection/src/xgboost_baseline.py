"""
xgboost_baseline.py
===================
Phase 1a: Standard XGBoost Classifier (Raw Imbalanced Data)
=======================================================

The script trains a cost-blind XGBoost baseline without SMOTE, 
directly replicating the methodology and results found in Section 3.1.4 
and Table 4 of Batsyas and Yaduwanshi (2026).
"""

import os
import sys
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from xgboost import XGBClassifier

from config import XGBOOST_PARAMS, MODELS_DIR
from utils import section, step, info, success, ensure_dirs
from data_prep import prepare_data
from evaluation import evaluate

TOTAL_STEPS  = 4
PHASE_LABEL  = "Phase 1a — Base XGBoost (No SMOTE)"
METRICS_FILE = "base_model_no_smote_metrics.json"
PLOT_SUBDIR  = "phase1a"
MODEL_FILE   = os.path.join(MODELS_DIR, "base_xgboost_no_smote.pkl")

def main():
    ensure_dirs()
    section("PHASE 1A: XGBOOST BASELINE (NO SMOTE)")
    print(f"  Reference: Batsyas and Yaduwanshi (2026), Section 3.1.4")
    print(f"  Purpose: Establish the true academic baseline on imbalanced data.")

    # Step 1: The algorithm prepares the data, explicitly skipping SMOTE.
    step(1, TOTAL_STEPS, "Loading and preparing data")
    (X_train, X_test,
     y_train, y_test,
     amounts_train, amounts_test,
     X_train_res, y_train_res) = prepare_data(apply_smote_flag=False)

    # Step 2: The script initializes and trains the standard XGBoost classifier.
    step(2, TOTAL_STEPS, "Training standard XGBoost classifier")
    info("The model is training on raw, imbalanced data using standard log-loss.")

    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    joblib.dump(model, MODEL_FILE)
    success(f"The model weights are saved: {MODEL_FILE}")

    # Step 3: The script evaluates the model on the 10% holdout test set.
    step(3, TOTAL_STEPS, "Evaluating on untouched test set")
    metrics = evaluate(
        model            = model,
        X_test           = X_test,
        y_test           = y_test,
        amounts_test     = amounts_test,
        phase_label      = PHASE_LABEL,
        metrics_filename = METRICS_FILE,
        subdir           = PLOT_SUBDIR,
        all_phases       = None
    )

    # Step 4: The script generates the summary to compare with the reference paper.
    step(4, TOTAL_STEPS, "Phase 1a complete")
    std     = metrics["standard_metrics"]
    savings = metrics["savings_score"]

    section("PHASE 1A SUMMARY")
    print(f"  Verify these metrics against Batsyas and Yaduwanshi (2026) Table 4:")
    print(f"  F1-Score                 : {std['f1_score']:.4f}")
    print(f"  Precision                : {std['precision']:.4f}")
    print(f"  Recall                   : {std['recall']:.4f}")
    print(f"\n  Savings Score (PRIMARY)  : {savings['savings_score_pct']:.4f}%")
    print("=" * 62)
    success("Phase 1a baseline execution completed successfully.")

if __name__ == "__main__":
    main()