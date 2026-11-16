# Beyond Binary Classification 🛡️

**Latent Behavioural Segmentation and Cost-Sensitive Machine Learning for Credit Card Fraud Detection**

Honours dissertation (BSc Honours in Information Technology, Data Science), Eduvos, 2026.
Author: Tanaka Maurice Munyoro. Supervisor: Dr Sandura Shumba. Co-supervisor: Ndai Makhurane (Mapaso).

This repository contains the code, results, and an interactive simulated dashboard for the experiments reported in the dissertation.

---

## 1. What this study asks

Fraud models are usually trained and judged with cost-blind metrics such as the F1-score, which treat a missed R12 fraud and a missed R5,000 fraud as the same error. Example-Dependent Cost-Sensitive Learning (EDCSL; Bahnsen et al., 2014) puts the monetary value of each transaction into the loss function instead.

EDCSL is normally applied to one global model. This study tests whether **training a separate EDCSL model per behavioural segment** produces greater financial savings than one global EDCSL model.

The ULB credit card dataset is anonymised with PCA, so explicit transaction categories do not exist. Segments are therefore discovered with **unsupervised K-Means clustering** on the PCA features.

**Research questions**

1. How much financial loss does a standard (cost-blind) XGBoost classifier incur, and how does that compare with its F1-based performance?
2. How much does a global EDCSL model reduce financial loss relative to that baseline?
3. Does clustering transactions before EDCSL training produce a greater Savings Score than the global EDCSL model, and is the difference statistically significant?

---

## 2. Headline results

Test set: 56,962 transactions, of which 98 are fraud. Maximum possible loss (no detection) is **R10,644.93**. Administrative cost per flagged transaction is **Ca = R50**.

| Phase | Model | TP | FP | FN | Precision | Recall | F1 | PR-AUC | Admin cost | Savings Score |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 1a | XGBoost, cost-blind | 76 | 11 | 22 | 0.8736 | 0.7755 | 0.8216 | 0.8065 | R550 | 49.74% |
| 1b | XGBoost + SMOTE | 88 | 403 | 10 | 0.1792 | 0.8980 | 0.2988 | 0.7607 | R20,150 | -101.48% |
| 2a | Global EDCSL | 70 | 11 | 28 | 0.8642 | 0.7143 | 0.7821 | 0.8119 | R550 | 55.39% |
| 2b | Global EDCSL + SMOTE | 87 | 303 | 11 | 0.2231 | 0.8878 | 0.3566 | 0.8157 | R15,150 | -54.52% |
| 3a | **Category-specific EDCSL (K-Means, k=5)** | 73 | 5 | 25 | **0.9359** | 0.7449 | **0.8295** | 0.8301 | **R250** | **57.81%** |
| 3b | Category-specific EDCSL + SMOTE | 86 | 204 | 12 | 0.2966 | 0.8776 | 0.4433 | 0.8389 | R10,200 | -8.98% |

### What the results show

- **Financial improvement at each stage (raw data).** The Savings Score rose from 49.74% (baseline) to 55.39% (global EDCSL) to 57.81% (category-specific EDCSL). The step from global to category-specific EDCSL is 2.42 percentage points, about R258 on this test set.
- **Cost-sensitivity trades Recall for money.** Phase 2a caught fewer frauds than Phase 1a (70 vs 76) yet saved more, because the missed frauds were worth less in total (R4,198.42 vs R4,800.26 unrecovered).
- **Category-specific EDCSL had the fewest false alarms.** Phase 3a produced 5 false positives against 11 for Phases 1a and 2a, and also had the highest Precision and F1 among the raw-data models.
- **SMOTE is financially harmful under this cost model.** Cost-blind SMOTE (1b) raised Recall to 89.8% but produced 403 false positives. The resulting R20,150 of administrative cost, plus R1,297.50 of missed fraud, was about twice the R10,644.93 that no detection would have cost. EDCSL (2b) and segmentation (3b) progressively reduced this damage but did not remove it.
- **Statistical significance was not reached.** McNemar's exact test comparing Phase 1a and Phase 3a gave p = 0.6636 (see section 5). The financial improvement is therefore directional evidence, not a statistically confirmed difference.

---

## 3. Experimental design

Each phase changes one thing and keeps everything else identical (same split, same hyperparameters, same seed).

```text
Phase 1  Cost-blind XGBoost                      (1a raw | 1b + SMOTE)
   |  + transaction-level cost weights
Phase 2  Global EDCSL XGBoost                    (2a raw | 2b + SMOTE)
   |  + K-Means segmentation, one model per cluster
Phase 3  Category-specific EDCSL (ClusterEnsemble) (3a raw | 3b + SMOTE)
```

Segmentation without cost-sensitivity (a cluster-wise cost-blind model) was not run, so the study isolates the effect of EDCSL and of segmentation on top of EDCSL, not segmentation alone.

### Setup

| Item | Value |
| :-- | :-- |
| Dataset | ULB European Credit Card Fraud (Kaggle, mlg-ulb/creditcardfraud) |
| Size | 284,807 transactions, 492 fraud (0.172%) |
| Features | V1-V28 (PCA), Time, Amount |
| Split | 80:20 stratified; train 227,845 (394 fraud), test 56,962 (98 fraud) |
| Classifier | XGBoost, max_depth=3, n_estimators=50 |
| Seed | random_state=42 for split, SMOTE, K-Means and XGBoost |
| SMOTE | Training data only. In Phase 3b it is applied inside each cluster, after clustering |
| Clustering | K-Means on V1-V28, k = 5, fitted on training data only |
| Inference (Phase 3) | Each test transaction goes to its nearest training centroid and is scored by that cluster's model |
| Administrative cost | Ca = R50 per flagged transaction (fixed experimental assumption) |
| Cost weighting (EDCSL) | Weights are applied as XGBoost sample weights. Fraudulent transactions get a weight derived from the transaction Amount (the cost of a false negative). Legitimate transactions get a weight derived from a logarithmic transformation of Ca (the cost of a false positive). See `src/ulb_xgboost_edcsl_phase2.py`. |

### Savings Score

```text
Savings = 1 - (sum of Amount over missed frauds + Ca x number of false positives)
              / (sum of Amount over all frauds in the test set)
```

Here the denominator is R10,644.93, the loss when nothing is flagged. A positive score means the model costs less than doing nothing. A negative score means it costs more.

### Latent clusters (training set, Phase 3a)

| Cluster | Transactions | Fraud | Fraud rate |
| :-- | --: | --: | --: |
| 0 | 60,743 | 109 | 0.179% |
| 1 | 39,610 | 120 | 0.303% |
| 2 | 58,436 | 113 | 0.193% |
| 3 | 55,270 | 52 | 0.094% |
| 4 | 13,786 | 0 | 0.000% |

Cluster 4 contains no training fraud, so its model can only predict "legitimate". In practice the ensemble has four active fraud models.

---

## 4. Currency note

The ULB data comes from European cardholders and Amount is not denominated in Rand. Values are shown with the "R" symbol to match the dissertation's South African framing and are not converted. The Savings Score is a ratio, so the symbol does not affect any percentage reported here.

---

## 5. Statistical test and its limits

McNemar's exact test compares Phase 1a and Phase 3a on the same test set.

|  | Phase 3a correct | Phase 3a wrong |
| :-- | --: | --: |
| **Phase 1a correct** | 56,920 | 9 |
| **Phase 1a wrong** | 12 | 21 |

The test uses only the 21 transactions on which the models disagree (9 vs 12), so p = 0.6636.

With 98 fraud cases there are very few disagreements, so the test has little power to detect a difference of this size.

---

## 6. Limitations

- **Small fraud population.** Every fraud-related result rests on 98 test frauds (394 training frauds).
- **Single split, single seed.** No cross-validation, repeated runs or confidence intervals were produced.
- **Zero-fraud cluster.** Unsupervised clustering on an imbalanced dataset can produce segments with no fraud, which cannot support a supervised model.
- **Dataset scope.** Two days of European card transactions from 2013.

---

## 7. Repository structure

```text
fraud-detection/
├── data/                           # (gitignored) creditcard.csv from Kaggle
├── dashboard/
│   └── index.html                  # Interactive Command Center Dashboard
├── models/                         # Serialised XGBoost and K-Means models (.pkl)
├── results/
│   ├── ConsoleOutput/              # Text logs of pipeline execution
│   ├── metrics/                    # JSON files capturing final KPIs
│   └── plots/                      # ROC curves and confusion matrices by phase
├── src/
│   ├── config.py                   # Hyperparameters and file paths
│   ├── utils.py                    # Logging and directory helpers
│   ├── ulb_data_prep.py            # Loading, 80:20 stratified split, SMOTE
│   ├── evaluation.py               # Statistical metrics and Savings Score
│   ├── ulb_base_xgboost.py         # Phase 1: cost-blind baseline
│   ├── ulb_xgboost_edcsl_phase2.py # Phase 2: global EDCSL
│   ├── ulb_xgboost_edcsl_phase3.py # Phase 3: category-specific EDCSL (ClusterEnsemble)
│   └── run_mcnemars.py             # McNemar's test
└── README.md
```

---

## 8. Reproducing the results & Interactive Dashboard

This repository offers two ways to explore the results: Execution Mode (running the Python pipeline) and Dashboard Mode (interactive exploration of results).

### Option A: The Interactive Dashboard (No setup required)

You do not need to install Python to view the results.

1. Navigate to the `dashboard/` folder.
2. Double-click `index.html` to open it in any modern web browser.

The dashboard functions as a simulated Command Center. Select any model phase from the sidebar and click **Execute Model**. It dynamically reads the generated `metrics.json` logs and loads the confusion matrices and ROC curves directly from the `results/` folder for interactive analysis.

### Option B: Code Execution (Running the Python Pipeline)

1. Install dependencies: `pip install -r requirements.txt`
2. Download `creditcard.csv` from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it in `data/`.
3. Run the phases sequentially from the repository root:

```bash
python src/ulb_base_xgboost.py
python src/ulb_xgboost_edcsl_phase2.py
python src/ulb_xgboost_edcsl_phase3.py
python src/run_mcnemars.py
```

(All random processes use `random_state=42`, ensuring exact reproducibility of the tables in section 2).

---

## 9. Dissertation Citation

Munyoro, T.M., 2026. Beyond Binary Classification: Latent Behavioural Segmentation and Cost-Sensitive Machine Learning for Credit Card Fraud Detection: source code [Online]. Available at: https://github.com/[username]/[repository] (release v1.0-dissertation).

---

## 10. References

Bahnsen, A.C., Aouada, D., Stojanovic, A. and Ottersten, B., 2014. Example-dependent cost-sensitive logistic regression for credit card fraud detection. Machine Learning, 97(1-2), pp.277-289.

Chawla, N.V. et al., 2002. SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, pp.321-357.

Chen, T. and Guestrin, C., 2016. XGBoost: A scalable tree boosting system. Proceedings of KDD 2016, pp.785-794.

Dal Pozzolo, A. et al., 2014. Learned lessons in credit card fraud detection from a practitioner perspective. Expert Systems with Applications, 41(10), pp.4915-4928.

Le Borgne, Y.A. et al., 2022. Reproducible Machine Learning for Credit Card Fraud Detection: Practical Handbook. Brussels: ULB Machine Learning Group.

McNemar, Q., 1947. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2), pp.153-157.