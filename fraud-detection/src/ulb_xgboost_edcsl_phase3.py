"""
ulb_xgboost_edcsl_phase3.py
===========================
Phase 3: Category-Specific Example-Dependent Cost-Sensitive Learning (EDCSL).
Uses Unsupervised K-Means Clustering on PCA features to discover latent behavioral 
categories, then trains independent EDCSL models per cluster.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from xgboost import XGBClassifier
from config import XGBOOST_PARAMS, MODELS_DIR, ADMIN_FEE_RANDS
from utils import section, step, success, ensure_dirs
from ulb_data_prep import prepare_ulb_data
from evaluation import evaluate

def calculate_edcsl_weights(y_train, amounts_train):
    """Calculates financial weights using Bahnsen et al.'s logarithmic approach."""
    y_array = np.array(y_train)
    amounts_array = np.array(amounts_train)
    weights = np.zeros_like(y_array, dtype=float)
    
    mask_legit = (y_array == 0)
    mask_fraud = (y_array == 1)
    
    weights[mask_legit] = np.log1p(ADMIN_FEE_RANDS)
    weights[mask_fraud] = np.log1p(amounts_array[mask_fraud])
    
    return weights

class ClusterEnsemble:
    """
    Custom wrapper that handles K-Means clustering and routes data to 
    cluster-specific XGBoost EDCSL models. Acts like a standard scikit-learn model.
    """
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        # n_init="auto" suppresses sklearn warnings
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=random_state, n_init="auto")
        self.models = {}
        self.feature_importances_ = None 

    def fit(self, X, y, sample_weight=None):
        print(f"\n      [K-MEANS] Discovering {self.n_clusters} latent behavioral clusters...")
        # Scale features for distance-based K-Means (but keep raw features for XGBoost)
        X_scaled = self.scaler.fit_transform(X)
        clusters = self.kmeans.fit_predict(X_scaled)
        
        y_array = np.array(y)
        if sample_weight is not None:
            sw_array = np.array(sample_weight)
            
        print("      [K-MEANS] Training independent EDCSL models per cluster:")
        for c in range(self.n_clusters):
            mask = (clusters == c)
            X_c = X.iloc[mask]
            y_c = y_array[mask]
            sw_c = sw_array[mask] if sample_weight is not None else None
            
            fraud_count = np.sum(y_c == 1)
            print(f"        -> Cluster {c}: {len(X_c):,} transactions ({fraud_count} frauds)")
            
            # If a cluster has only legitimate transactions, save a hardcoded 0 predictor
            if len(np.unique(y_c)) == 1:
                self.models[c] = np.unique(y_c)[0]
            else:
                model = XGBClassifier(**XGBOOST_PARAMS)
                model.fit(X_c, y_c, sample_weight=sw_c, verbose=False)
                self.models[c] = model
                
        # Mock global feature importances by averaging the valid cluster models
        valid_models = [m for m in self.models.values() if hasattr(m, 'feature_importances_')]
        if valid_models:
            self.feature_importances_ = np.mean([m.feature_importances_ for m in valid_models], axis=0)
        else:
            self.feature_importances_ = np.zeros(X.shape[1])

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        clusters = self.kmeans.predict(X_scaled)
        preds = np.zeros(len(X))
        
        for c in range(self.n_clusters):
            mask = (clusters == c)
            if not np.any(mask): continue
            
            if isinstance(self.models[c], (int, np.integer)):
                preds[mask] = self.models[c]
            else:
                preds[mask] = self.models[c].predict(X.iloc[mask])
        return preds

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        clusters = self.kmeans.predict(X_scaled)
        probas = np.zeros((len(X), 2))
        
        for c in range(self.n_clusters):
            mask = (clusters == c)
            if not np.any(mask): continue
            
            if isinstance(self.models[c], (int, np.integer)):
                probas[mask, self.models[c]] = 1.0
            else:
                probas[mask] = self.models[c].predict_proba(X.iloc[mask])
        return probas

def run_ulb_phase3(use_smote: bool, phase_label: str, model_filename: str, metrics_filename: str, plot_subdir: str):
    section(f"ULB TRAINING: {phase_label}")

    step(1, 3, f"Preparing ULB Data (SMOTE = {use_smote})")
    (X_train, X_test, y_train, y_test,
     amounts_train, amounts_test,
     X_train_res, y_train_res) = prepare_ulb_data(apply_smote_flag=use_smote)

    amounts_train_res = X_train_res['Amount']
    sample_weights = calculate_edcsl_weights(y_train_res, amounts_train_res)

    step(2, 3, "Executing K-Means and Category-Specific EDCSL")
    # Using k=5 to mirror the 5 original categories in PaySim
    ensemble = ClusterEnsemble(n_clusters=5, random_state=42)
    ensemble.fit(X_train_res, y_train_res, sample_weight=sample_weights)
    
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(ensemble, model_path)
    success(f"Cluster Ensemble saved: {model_filename}")

    step(3, 3, f"Evaluating {phase_label}")
    metrics = evaluate(
        model            = ensemble,
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
    print("  INITIATING ULB DATASET CATEGORY EDCSL (PHASE 3A & 3B)")
    print("="*70)

    # RUN 1: RAW IMBALANCED DATA (NO SMOTE)
    metrics_a = run_ulb_phase3(
        use_smote=False,
        phase_label="ULB Phase 3a (Category Clustering + EDCSL)",
        model_filename="ulb_edcsl_phase3_no_smote.pkl",
        metrics_filename="ulb_edcsl_phase3_no_smote_metrics.json",
        plot_subdir="ulb_phase3a_no_smote"
    )

    # RUN 2: SYNTHETICALLY BALANCED DATA (WITH SMOTE)
    metrics_b = run_ulb_phase3(
        use_smote=True,
        phase_label="ULB Phase 3b (SMOTE + Category Clustering + EDCSL)",
        model_filename="ulb_edcsl_phase3_smote.pkl",
        metrics_filename="ulb_edcsl_phase3_smote_metrics.json",
        plot_subdir="ulb_phase3b_smote"
    )

    score_3a = metrics_a["savings_score"]["savings_score_pct"]
    score_3b = metrics_b["savings_score"]["savings_score_pct"]
    
    f1_3a = metrics_a["standard_metrics"]["f1_score"]
    f1_3b = metrics_b["standard_metrics"]["f1_score"]

    print("\n" + "█"*72)
    print("  PHASE 3 COMPARATIVE METRICS SUMMARY: PHASE 3A vs PHASE 3B")
    print("█"*72)
    print(f"  {'Metric':<25} | {'Phase 3a (Raw)':<20} | {'Phase 3b (SMOTE)':<20}")
    print("  " + "─"*70)
    print(f"  {'F1-Score':<25} | {f1_3a:<20.4f} | {f1_3b:<20.4f}")
    print(f"  {'Savings Score (%)':<25} | {score_3a:<20.4f} | {score_3b:<20.4f}")
    print("█"*72 + "\n")

if __name__ == "__main__":
    main()