"""
xgboost_baseline.py
Phase 1: Standard XGBoost Binary Classifier — Base Model

"""

from xgboost import XGBClassifier
import warnings

# Import our custom architectural modules
from config import XGBOOST_PARAMS
from utils import ensure_dirs, section, step, success
from data_prep import prepare_data
from evaluation import evaluate
from utils import save_model

warnings.filterwarnings('ignore')

def run_baseline():
    section("PHASE 1: STANDARD XGBOOST BASELINE")
    
    ensure_dirs()
    
    # 1. Execute Data Pipeline
    # One single call to the full data preparation pipeline.
    (X_train, X_test, y_train, y_test, 
     amounts_train, amounts_test, 
     X_train_res, y_train_res) = prepare_data(apply_smote_flag=True)
    
    # 2. Initialize Model
    step(5, 7, "Initializing standard XGBoost classifier")
    # ** unpacks the dictionary from config.py directly into the model
    model = XGBClassifier(**XGBOOST_PARAMS)
    
    # 3. Train Model
    step(6, 7, "Training XGBoost on SMOTE-balanced data")
    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_test, y_test)],
        verbose=50  # Prints training progress every 50 trees
    )
    
    # Save the trained model weights
    save_model(model, "base_xgboost.pkl")
    
    # 4. Evaluate Pipeline
    step(7, 7, "Running evaluation and calculating Savings Score")
    
    # The evaluate function handles all standard metrics, the Savings Score math,
    # prints the formatted console output, generates all charts, and saves the JSON.
    metrics = evaluate(
        model=model,
        X_test=X_test,
        y_test=y_test,
        amounts_test=amounts_test,
        phase_label="Phase 1 — Base XGBoost",
        metrics_filename="base_model_metrics.json",
        subdir="phase1"
    )
    
    success("Phase 1 Baseline execution completed successfully.")
    return metrics

if __name__ == "__main__":
    run_baseline()