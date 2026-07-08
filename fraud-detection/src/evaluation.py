import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    average_precision_score
)

from config import ADMIN_FEE_RANDS
from utils import section, save_plot, save_metrics


# 1. Standard metrics

def compute_standard_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              y_proba: np.ndarray) -> dict:
    """
    Compute standard binary classification metrics (cost-BLIND).
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1_score":  round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "roc_auc":   round(float(roc_auc_score(y_true, y_proba)), 6),
        "pr_auc":    round(float(average_precision_score(y_true, y_proba)), 6),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)
        }
    }


# 2. Savings score (Bahnsen et al., 2014)

def compute_savings_score(y_true: np.ndarray, y_pred: np.ndarray,
                           amounts: np.ndarray) -> dict:
    """
    Compute the Savings Score as defined by Bahnsen et al. (2014).
    """
    y_true   = np.array(y_true)
    y_pred   = np.array(y_pred)
    amounts  = np.array(amounts)

    # Boolean masks for each outcome type
    mask_actual_fraud = (y_true == 1)
    mask_fn           = (y_true == 1) & (y_pred == 0)   # missed fraud
    mask_fp           = (y_true == 0) & (y_pred == 1)   # false alarm

    # Cost components
    cost_baseline = amounts[mask_actual_fraud].sum()
    cost_fn       = amounts[mask_fn].sum()
    cost_fp       = ADMIN_FEE_RANDS * mask_fp.sum()
    cost_model    = cost_fn + cost_fp

    # Savings Score
    if cost_baseline == 0:
        savings_score = 0.0
    else:
        savings_score = float((cost_baseline - cost_model) / cost_baseline)

    financial_loss_prevented = cost_baseline - cost_model

    return {
        "Ca_rands":                      ADMIN_FEE_RANDS,
        "savings_score":                 round(savings_score, 6),
        "savings_score_pct":             round(savings_score * 100, 4),
        "cost_baseline_rands":           round(float(cost_baseline), 2),
        "cost_model_rands":              round(float(cost_model), 2),
        "cost_fn_rands":                 round(float(cost_fn), 2),
        "cost_fp_rands":                 round(float(cost_fp), 2),
        "financial_loss_prevented_rands":round(float(financial_loss_prevented), 2),
        "false_negatives_count":         int(mask_fn.sum()),
        "false_positives_count":         int(mask_fp.sum()),
        "total_fraud_transactions":      int(mask_actual_fraud.sum()),
    }


# 3. Plot funcions
def plot_confusion_matrix(y_true, y_pred, phase_label: str, subdir: str = "") -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=['Legitimate', 'Fraud'],
        yticklabels=['Legitimate', 'Fraud'],
        xlabel='Predicted Label',
        ylabel='True Label',
        title=f'Confusion Matrix — {phase_label}'
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]:,}',
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=12, fontweight='bold')
    plt.tight_layout()
    return save_plot(fig, "confusion_matrix.png", subdir)


def plot_roc_curve(y_true, y_proba, roc_auc: float,
                   phase_label: str, subdir: str = "") -> str:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, color='#2563EB',
            label=f'{phase_label} (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#2563EB')
    ax.set(
        xlabel='False Positive Rate', ylabel='True Positive Rate',
        title=f'ROC Curve — {phase_label}',
        xlim=[0, 1], ylim=[0, 1]
    )
    ax.legend(loc='lower right')
    plt.tight_layout()
    return save_plot(fig, "roc_curve.png", subdir), fpr, tpr


def plot_feature_importance(model, feature_names: list,
                             phase_label: str, subdir: str = "") -> str:
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind='barh', ax=ax, color='#2563EB', edgecolor='white')
    ax.set(
        title=f'Feature Importance — {phase_label}',
        xlabel='Importance Score (Gain)'
    )
    plt.tight_layout()
    return save_plot(fig, "feature_importance.png", subdir)


def plot_savings_score(savings: dict, phase_label: str,
                       all_phases: dict = None, subdir: str = "") -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: cost breakdown ──
    ax = axes[0]
    labels = ['Baseline\n(do nothing)', 'FN Losses\n(missed fraud)',
              'FP Costs\n(false alarms)', 'Total\nModel Cost']
    values = [
        savings['cost_baseline_rands'],
        savings['cost_fn_rands'],
        savings['cost_fp_rands'],
        savings['cost_model_rands'],
    ]
    colours = ['#94a3b8', '#fc8181', '#f6c90e', '#63b3ed']
    bars = ax.bar(labels, [v / 1e6 for v in values],
                  color=colours, edgecolor='white', linewidth=0.8)
    ax.set_ylabel('Amount (R millions)')
    ax.set_title(f'Cost Breakdown — {phase_label}')
    ax.set_ylim(0, max(v / 1e6 for v in values) * 1.2)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) / 1e6 * 0.02,
                f'R{val/1e6:.2f}M', ha='center', va='bottom', fontsize=9)

    # ── Right: phase comparison ──
    ax2 = axes[1]
    if all_phases is None:
        all_phases = {phase_label: savings['savings_score_pct']}

    phase_names = [
        'Phase 1\nBase XGBoost',
        'Phase 2\nGlobal EDCSL',
        'Phase 3\nCategory EDCSL'
    ]
    default_labels = list(all_phases.keys())
    bar_vals = []
    bar_cols = []
    for name in phase_names:
        matched = next((v for k, v in all_phases.items() if k in name or name in k), None)
        bar_vals.append(matched if matched is not None else 0)
        bar_cols.append('#63b3ed' if matched is not None else '#374151')

    bars2 = ax2.bar(phase_names, bar_vals, color=bar_cols, edgecolor='white', linewidth=0.8)
    ax2.set_ylabel('Savings Score (%)')
    ax2.set_ylim(0, 100)
    ax2.set_title('Savings Score — Phase Comparison')
    for bar, val in zip(bars2, bar_vals):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, val + 1,
                     f'{val:.2f}%', ha='center', va='bottom',
                     fontsize=10, fontweight='bold')

    plt.suptitle(
        f'Savings Score = {savings["savings_score"]:.4f}  '
        f'|  Ca = R{savings["Ca_rands"]}  '
        f'|  Bahnsen et al. (2014)',
        fontsize=9, color='#6b7280', y=0.01
    )
    plt.tight_layout()
    return save_plot(fig, "savings_score.png", subdir)


# Full evaluation method

def evaluate(model, X_test, y_test, amounts_test,
             phase_label: str, metrics_filename: str,
             subdir: str = "", all_phases: dict = None) -> dict:
    
    y_true   = np.array(y_test)
    y_pred   = model.predict(X_test)
    y_proba  = model.predict_proba(X_test)[:, 1]
    amounts  = np.array(amounts_test)
    
    # Extract dynamic feature names from the DataFrame
    feature_names = list(X_test.columns)

    # ── Standard metrics ──
    std = compute_standard_metrics(y_true, y_pred, y_proba)
    tn, fp, fn, tp = (std['confusion_matrix']['tn'], std['confusion_matrix']['fp'],
                      std['confusion_matrix']['fn'], std['confusion_matrix']['tp'])

    # ── Savings Score ──
    savings = compute_savings_score(y_true, y_pred, amounts)

    # ── Console output ──
    section(f"RESULTS — {phase_label}")
    print(f"  {'─'*56}")
    print(f"  STANDARD METRICS (cost-blind)")
    print(f"  {'─'*56}")
    print(f"  Accuracy   : {std['accuracy']:.4f}   ← misleading on imbalanced data")
    print(f"  Precision  : {std['precision']:.4f}")
    print(f"  Recall     : {std['recall']:.4f}")
    print(f"  F1-Score   : {std['f1_score']:.4f}")
    print(f"  ROC-AUC    : {std['roc_auc']:.4f}")
    print(f"  PR-AUC     : {std['pr_auc']:.4f}   ← KEY METRIC")
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted 0    Predicted 1")
    print(f"  Actual 0  :  {tn:>12,}   {fp:>12,}   (TN / FP)")
    print(f"  Actual 1  :  {fn:>12,}   {tp:>12,}   (FN / TP)")
    print(f"\n  {'─'*56}")
    print(f"  SAVINGS SCORE (Bahnsen et al., 2014) — PRIMARY METRIC")
    print(f"  {'─'*56}")
    print(f"  Ca (investigation cost)            : R{savings['Ca_rands']:,.2f}")
    print(f"  Baseline cost (flag nothing)       : R{savings['cost_baseline_rands']:>15,.2f}")
    print(f"  FN loss (missed fraud)             : R{savings['cost_fn_rands']:>15,.2f}")
    print(f"  FP cost (false alarm overhead)     : R{savings['cost_fp_rands']:>15,.2f}")
    print(f"  Total model cost                   : R{savings['cost_model_rands']:>15,.2f}")
    print(f"  Financial loss prevented           : R{savings['financial_loss_prevented_rands']:>15,.2f}")
    print(f"\n  ┌──────────────────────────────────────────────────────┐")
    print(f"  │  SAVINGS SCORE = {savings['savings_score']:.4f}  ({savings['savings_score_pct']:.2f}%)               │")
    print(f"  └──────────────────────────────────────────────────────┘")

    print(f"\n  {classification_report(y_true, y_pred, target_names=['Legitimate','Fraud'])}")

    # ── Plots ──
    plot_confusion_matrix(y_true, y_pred, phase_label, subdir)
    roc_result = plot_roc_curve(y_true, y_proba, std['roc_auc'], phase_label, subdir)
    _, fpr_arr, tpr_arr = roc_result
    plot_feature_importance(model, feature_names, phase_label, subdir)
    plot_savings_score(savings, phase_label, all_phases, subdir)

    # ── Assemble and save metrics dict ──
    fpr_list = [round(float(v), 4) for v in fpr_arr[::50]]
    tpr_list = [round(float(v), 4) for v in tpr_arr[::50]]

    metrics = {
        "phase_label":   phase_label,
        "standard_metrics": std,
        "savings_score": savings,
        "roc_curve": {"fpr": fpr_list, "tpr": tpr_list},
        "feature_importances": {
            feat: round(float(imp), 6)
            for feat, imp in zip(feature_names, model.feature_importances_)
        }
    }

    save_metrics(metrics, metrics_filename)
    return metrics