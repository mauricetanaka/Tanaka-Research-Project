"""
edcsl_model.py
==============
Phase 2: Global Example-Dependent Cost-Sensitive XGBoost
=========================================================

The script trains the Phase 2 model using the custom EDCSL objective.
It evaluates BOTH the Raw Data (No SMOTE) and Resampled Data (With SMOTE) 
to prove the academic interaction between synthetic data and cost matrices.
"""

import os
import sys
import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from xgboost import XGBClassifier
from config import XGBOOST_PARAMS, MODELS_DIR, ADMIN_FEE_RANDS
from utils import section, step, info, warn, success, ensure_dirs, load_metrics
from data_prep import prepare_data
from edcsl_loss import build_edcsl_objective
from evaluation import evaluate

def run_edcsl_pipeline(use_smote: bool, phase_label: str, model_filename: str, metrics_filename: str, plot_subdir: str, base_metrics_file: str):
    """Executes the complete EDCSL training and evaluation pipeline."""
    section(f"TRAINING: {phase_label}")
    
    # 1. Data Prep
    step(1, 5, f"Preparing Data (SMOTE = {use_smote})")
    (X_train, X_test, y_train, y_test,
     amounts_train, amounts_test,
     X_train_res, y_train_res) = prepare_data(apply_smote_flag=use_smote)

    # 2. Build Objective
    step(2, 5, "Building EDCSL custom objective function")
    amounts_resampled = np.asarray(X_train_res['amount'].to_numpy(), dtype=float)
    edcsl_objective = build_edcsl_objective(
        amounts = amounts_resampled,
        y_true  = np.asarray(y_train_res.to_numpy(), dtype=float),
        Ca      = ADMIN_FEE_RANDS
    )
    
    # 3. Train Model
    step(3, 5, "Training EDCSL XGBoost")
    edcsl_params = {k: v for k, v in XGBOOST_PARAMS.items() if k != 'eval_metric'}
    model = XGBClassifier(**edcsl_params)
    model.set_params(objective=edcsl_objective)
    model.fit(X_train_res, y_train_res, verbose=False)
    
    # Detach custom objective for saving
    model.set_params(objective='binary:logistic')
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(model, model_path)
    success(f"Model saved: {model_filename}")

    # 4. Load Baseline for Comparison
    step(4, 5, f"Loading Baseline ({base_metrics_file})")
    all_phases = {}
    base_score = None
    try:
        p1 = load_metrics(base_metrics_file)
        base_score = p1["savings_score"]["savings_score_pct"]
        all_phases["Phase 1 Baseline"] = base_score
        info("Baseline metrics successfully loaded.")
    except Exception:
        warn(f"Could not find {base_metrics_file}.")

    # 5. Evaluate
    step(5, 5, f"Evaluating {phase_label}")
    metrics = evaluate(
        model            = model,
        X_test           = X_test,
        y_test           = y_test,
        amounts_test     = amounts_test,
        phase_label      = phase_label,
        metrics_filename = metrics_filename,
        subdir           = plot_subdir,
        all_phases       = all_phases
    )
    
    return metrics, base_score

def main():
    ensure_dirs()
    
    print("\n" + "="*70)
    print("  INITIATING FULL PHASE 2 EXPERIMENTAL MATRIX")
    print("="*70)

    # RUN 1: NO SMOTE (Raw Data)
    metrics_a, base_score_a = run_edcsl_pipeline(
        use_smote=False,
        phase_label="Phase 2a (Raw Data + EDCSL)",
        model_filename="phase2_edcsl_no_smote.pkl",
        metrics_filename="phase2_no_smote_metrics.json",
        plot_subdir="phase2a_no_smote",
        base_metrics_file="base_model_no_smote_metrics.json"
    )

    # RUN 2: WITH SMOTE (Synthetic Data)
    metrics_b, base_score_b = run_edcsl_pipeline(
        use_smote=True,
        phase_label="Phase 2b (SMOTE + EDCSL)",
        model_filename="phase2_edcsl_smote.pkl",
        metrics_filename="phase2_smote_metrics.json",
        plot_subdir="phase2b_smote",
        base_metrics_file="base_model_smote_metrics.json"
    )

    # ==============================================================
    # THE FINAL SUMMARY MATRIX
    # ==============================================================
    score_2a = metrics_a["savings_score"]["savings_score_pct"]
    score_2b = metrics_b["savings_score"]["savings_score_pct"]
    
    delta_a = (score_2a - base_score_a) if base_score_a else 0
    delta_b = (score_2b - base_score_b) if base_score_b else 0

    print("\n" + "█"*70)
    print("  FINAL 2x2 EXPERIMENTAL MATRIX SUMMARY")
    print("█"*70)
    
    print("\n  SCENARIO 1: RAW IMBALANCED DATA (NO SMOTE)")
    print("  ────────────────────────────────────────────────────────")
    print(f"  Phase 1a (Cost-Blind Baseline) : {base_score_a:>8.4f}%" if base_score_a else "  Phase 1a: N/A")
    print(f"  Phase 2a (EDCSL Cost-Matrix)   : {score_2a:>8.4f}%")
    print(f"  Financial Delta                : {'+' if delta_a > 0 else ''}{delta_a:>7.4f}%")
    if delta_a > 0:
        print("  ✅ HYPOTHESIS PROVEN: EDCSL successfully optimized raw data.")
    else:
        print("  ⚠️ EDCSL underperformed baseline.")

    print("\n  SCENARIO 2: SYNTHETIC BALANCED DATA (WITH SMOTE)")
    print("  ────────────────────────────────────────────────────────")
    print(f"  Phase 1b (Cost-Blind Baseline) : {base_score_b:>8.4f}%" if base_score_b else "  Phase 1b: N/A")
    print(f"  Phase 2b (EDCSL Cost-Matrix)   : {score_2b:>8.4f}%")
    print(f"  Financial Delta                : {'+' if delta_b > 0 else ''}{delta_b:>7.4f}%")
    if delta_b < 0:
        print("  ⚠️ COST-SENSITIVE AMPLIFICATION: Synthetic data poisons the cost matrix.")
    
    print("\n" + "█"*70 + "\n")

if __name__ == "__main__":
    main()