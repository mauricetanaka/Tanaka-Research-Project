"""
category_edcsl_model.py
=======================
Phase 3: Category-Specific Example-Dependent Cost-Sensitive Models

The script trains independent cost-sensitive models for each transaction 
category. It runs the full experimental matrix (With and Without SMOTE) 
to evaluate the financial impact of algorithmic segmentation.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from xgboost import XGBClassifier
from config import XGBOOST_PARAMS, MODELS_DIR, ADMIN_FEE_RANDS, CATEGORY_COL, MIN_FRAUD_CASES
from utils import section, step, info, warn, success, ensure_dirs, load_metrics
from data_prep import prepare_data
from edcsl_loss import build_edcsl_objective
from evaluation import evaluate

class CategoryEnsemble:
    """Routes transactions to their specific category model for classification."""
    def __init__(self, models_dict: dict, category_col: str):
        self.models_dict = models_dict
        self.category_col = category_col

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = pd.Series(0, index=X.index)
        for cat, model in self.models_dict.items():
            if model is not None:
                mask = (X[self.category_col] == cat)
                if mask.sum() > 0:
                    preds.loc[mask] = model.predict(X[mask])
        return preds.to_numpy(dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = pd.Series(0.0, index=X.index)
        for cat, model in self.models_dict.items():
            if model is not None:
                mask = (X[self.category_col] == cat)
                if mask.sum() > 0:
                    proba.loc[mask] = model.predict_proba(X[mask])[:, 1]
        
        proba_values = proba.to_numpy(dtype=float)
        return np.column_stack((1.0 - proba_values, proba_values))

    @property
    def feature_importances_(self):
        importances = [m.feature_importances_ for m in self.models_dict.values() if m is not None]
        return np.mean(importances, axis=0) if importances else np.array([])


def run_category_pipeline(use_smote: bool, phase_label: str, metrics_filename: str, plot_subdir: str, base_metrics_file: str, smote_label: str):
    section(f"TRAINING: {phase_label}")

    # 1. Data Prep
    step(1, 6, f"Preparing Data (SMOTE = {use_smote})")
    (X_train, X_test, y_train, y_test,
     amounts_train, amounts_test,
     X_train_res, y_train_res) = prepare_data(apply_smote_flag=use_smote)

    unique_categories = X_train_res[CATEGORY_COL].unique()
    trained_models = {}

    step(2, 6, "Training independent EDCSL models per category")
    edcsl_params = {k: v for k, v in XGBOOST_PARAMS.items() if k != 'eval_metric'}

    # 3. Train Category Models
    for cat in unique_categories:
        mask_train = (X_train_res[CATEGORY_COL] == cat)
        X_cat_train = X_train_res[mask_train]
        y_cat_train = y_train_res[mask_train]
        
        fraud_count = y_cat_train.sum()
        info(f"Category '{cat}': {len(X_cat_train):,} total rows | {fraud_count:,} fraud cases")

        if fraud_count < MIN_FRAUD_CASES:
            warn(f"Category '{cat}' has insufficient fraud. Predicting 0 (Legitimate).")
            trained_models[cat] = None
            continue
        
        amounts_cat = np.asarray(X_cat_train['amount'].to_numpy(), dtype=float)
        edcsl_objective = build_edcsl_objective(
            amounts = amounts_cat,
            y_true  = np.asarray(y_cat_train.to_numpy(), dtype=float),
            Ca      = ADMIN_FEE_RANDS
        )
        
        model = XGBClassifier(**edcsl_params)
        model.set_params(objective=edcsl_objective)
        model.fit(X_cat_train, y_cat_train, verbose=False)

        # Detach custom objective to allow pickling
        model.set_params(objective='binary:logistic')
        trained_models[cat] = model
        
        model_path = MODELS_DIR / f"phase3_edcsl_{smote_label}_cat_{cat}.pkl"
        joblib.dump(model, model_path)
        success(f"Saved EDCSL model for Category '{cat}'.")

    # 4. Assemble Ensemble
    step(4, 6, "Assembling Category Ensemble Wrapper")
    ensemble_model = CategoryEnsemble(trained_models, CATEGORY_COL)

    # 5. Load Baseline
    step(5, 6, f"Loading Baseline ({base_metrics_file})")
    all_phases = {}
    base_score = None
    try:
        p1 = load_metrics(base_metrics_file)
        base_score = p1["savings_score"]["savings_score_pct"]
        all_phases["Phase 1 Baseline"] = base_score
        info("Baseline metrics successfully loaded.")
    except Exception:
        warn(f"Could not find {base_metrics_file}.")

    # 6. Evaluate
    step(6, 6, f"Evaluating {phase_label}")
    metrics = evaluate(
        model            = ensemble_model,
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
    print("  INITIATING FULL PHASE 3 EXPERIMENTAL MATRIX")
    print("="*70)

    # RUN 1: NO SMOTE (Raw Data)
    metrics_a, base_score_a = run_category_pipeline(
        use_smote=False,
        phase_label="Phase 3a (Raw Data + Category EDCSL)",
        metrics_filename="phase3_no_smote_metrics.json",
        plot_subdir="phase3a_no_smote",
        base_metrics_file="base_model_no_smote_metrics.json",
        smote_label="no_smote"
    )

    # RUN 2: WITH SMOTE (Synthetic Data)
    metrics_b, base_score_b = run_category_pipeline(
        use_smote=True,
        phase_label="Phase 3b (SMOTE + Category EDCSL)",
        metrics_filename="phase3_smote_metrics.json",
        plot_subdir="phase3b_smote",
        base_metrics_file="base_model_smote_metrics.json",
        smote_label="smote"
    )

    # ==============================================================
    # THE FINAL SUMMARY MATRIX
    # ==============================================================
    score_3a = metrics_a["savings_score"]["savings_score_pct"]
    score_3b = metrics_b["savings_score"]["savings_score_pct"]
    
    delta_a = (score_3a - base_score_a) if base_score_a else 0
    delta_b = (score_3b - base_score_b) if base_score_b else 0

    print("\n" + "█"*70)
    print("  FINAL 2x2 EXPERIMENTAL MATRIX SUMMARY (PHASE 3)")
    print("█"*70)
    
    print("\n  SCENARIO 1: RAW IMBALANCED DATA (NO SMOTE)")
    print("  ────────────────────────────────────────────────────────")
    print(f"  Phase 1a (Cost-Blind Baseline) : {base_score_a:>8.4f}%" if base_score_a else "  Phase 1a: N/A")
    print(f"  Phase 3a (Category EDCSL)      : {score_3a:>8.4f}%")
    print(f"  Financial Delta                : {'+' if delta_a > 0 else ''}{delta_a:>7.4f}%")
    if delta_a > 0:
        print("  ✅ HYPOTHESIS PROVEN: Categorical EDCSL optimized raw data.")
    else:
        print("  ⚠️ Categorical EDCSL underperformed baseline.")

    print("\n  SCENARIO 2: SYNTHETIC BALANCED DATA (WITH SMOTE)")
    print("  ────────────────────────────────────────────────────────")
    print(f"  Phase 1b (Cost-Blind Baseline) : {base_score_b:>8.4f}%" if base_score_b else "  Phase 1b: N/A")
    print(f"  Phase 3b (Category EDCSL)      : {score_3b:>8.4f}%")
    print(f"  Financial Delta                : {'+' if delta_b > 0 else ''}{delta_b:>7.4f}%")
    if delta_b < 0:
        print("  ⚠️ COST-SENSITIVE AMPLIFICATION: Synthetic data poisons the segmented matrices.")
    
    print("\n" + "█"*70 + "\n")

if __name__ == "__main__":
    main()