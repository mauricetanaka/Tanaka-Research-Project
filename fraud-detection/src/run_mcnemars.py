"""
run_mcnemars_final.py
=====================
Calculates statistical significance between Phase 1a (Cost-Blind Baseline) 
and Phase 3a (Category-Specific EDCSL).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from statsmodels.stats.contingency_tables import mcnemar
from ulb_data_prep import prepare_ulb_data
from config import MODELS_DIR

# --- THE FIX FOR THE PICKLE ERROR ---
# Import the custom class and map it to __main__ so joblib can unpickle it
from ulb_xgboost_edcsl_phase3 import ClusterEnsemble
import __main__
__main__.ClusterEnsemble = ClusterEnsemble
# ------------------------------------

def main():
    print("\n[MCNEMAR'S TEST] Loading data and models...")
    # 1. Load the exact same test set
    _, X_test, _, y_test, _, _, _, _ = prepare_ulb_data(apply_smote_flag=False)
    y_true = np.array(y_test)

    # 2. Load Phase 1a and Phase 3a models
    try:
        model_1a = joblib.load(os.path.join(MODELS_DIR, "ulb_base_xgboost_no_smote.pkl"))
        model_3a = joblib.load(os.path.join(MODELS_DIR, "ulb_edcsl_phase3_no_smote.pkl"))
    except FileNotFoundError as e:
        print(f"Error: Models not found. {e}")
        return

    # 3. Generate predictions
    print("[MCNEMAR'S TEST] Generating predictions...")
    preds_1a = model_1a.predict(X_test)
    preds_3a = model_3a.predict(X_test)

    # 4. Calculate where models were correct
    correct_1a = (preds_1a == y_true)
    correct_3a = (preds_3a == y_true)

    # 5. Build the Contingency Table
    cell_a = np.sum(correct_1a & correct_3a)   # Both correct
    cell_b = np.sum(correct_1a & ~correct_3a)  # 1a right, 3a wrong
    cell_c = np.sum(~correct_1a & correct_3a)  # 1a wrong, 3a right
    cell_d = np.sum(~correct_1a & ~correct_3a) # Both wrong

    table = [[cell_a, cell_b],
             [cell_c, cell_d]]

    print("\n" + "█"*65)
    print("  MCNEMAR'S TEST: PHASE 1A (BASELINE) vs PHASE 3A (CATEGORY EDCSL)")
    print("█"*65)
    print(f"  Both Models Correct      : {cell_a:,}")
    print(f"  Only Phase 1a Correct    : {cell_b:,}")
    print(f"  Only Phase 3a Correct    : {cell_c:,}")
    print(f"  Both Models Wrong        : {cell_d:,}")
    print("─"*65)

    # 6. Calculate Significance
    result = mcnemar(table, exact=True)
    p_value = result.pvalue

    print(f"  p-value = {p_value:.6f}")
    if p_value < 0.05:
        print("  ✅ STATISTICALLY SIGNIFICANT")
        print("  The Category EDCSL architecture fundamentally changed the algorithm's performance.")
    else:
        print("  ⚠️ NOT SIGNIFICANT")
        print("  The difference in performance is likely due to random variance.")
    print("█"*65 + "\n")

if __name__ == "__main__":
    main()