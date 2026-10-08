# Beyond Binary Classification 🛡️

**Latent Behavioural Segmentation and Cost-Sensitive Machine Learning for Credit Card Fraud Detection**

> *A quantitative machine learning architecture demonstrating that combining unsupervised categorical segmentation (K-Means) with Example-Dependent Cost-Sensitive Learning (EDCSL) maximizes financial savings in credit card fraud detection on privacy-anonymized datasets.*

---

## 📖 Abstract

Credit card fraud inflicts severe financial damage on digital payment ecosystems. Current machine learning architectures predominantly function as binary classifiers optimized via cost-blind statistical metrics (e.g., the F1-score). These metrics treat all misclassification errors as financially equivalent, frequently causing models to generate massive administrative overhead through false positive inflation when deployed on imbalanced datasets. 

This repository implements a three-phase experimental matrix to evaluate whether applying an Example-Dependent Cost-Sensitive Learning (EDCSL) framework to behaviourally segmented transactions produces greater financial savings than a globally applied cost-sensitive model. Because real-world financial datasets utilize Principal Component Analysis (PCA) to protect customer privacy, explicit categorical labels are obfuscated. To resolve this, this architecture utilizes unsupervised **K-Means clustering** on PCA features to discover latent behavioural archetypes prior to training independent EDCSL **XGBoost** models. 

## 📊 Core Findings & Experimental Matrix

The experiment was conducted on the European Credit Card Fraud Dataset (ULB). The baseline maximum financial fraud exposure in the test set was **R10,644.93**, with a simulated administrative investigation cost of **R50.00** per flagged transaction.

| Experimental Phase | Model Architecture | Frauds Caught | False Positives | Admin Cost | Savings Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1a** | Base XGBoost (Cost-Blind) | 76 | 11 | R550 | 49.74% |
| **Phase 1b** | XGBoost + SMOTE | 88 | 403 | R20,150 | -101.48% ⚠️ |
| **Phase 2a** | Global EDCSL | 71 | 11 | R550 | 55.39% |
| **Phase 2b** | Global EDCSL + SMOTE | 92 | 303 | R15,150 | -54.52% |
| **Phase 3a** | **Category EDCSL (Latent Clusters)** | **73** | **5** | **R250** | **57.81%** 🏆 |
| **Phase 3b** | Category EDCSL + SMOTE | 86 | 204 | R10,200 | -8.98% |

* **The SMOTE Crisis:** Cost-blind data augmentation (Phase 1b) bankrupted the system. By blindly attempting to maximize minority-class Recall, it generated 403 False Positives. The R20,150 administrative overhead exceeded the total value of the fraud itself, yielding a -101.48% Savings Score.
* **The Segmentation Triumph:** The Category-Specific EDCSL architecture (Phase 3a) achieved the highest absolute financial performance. By isolating latent behavioural channels before applying financial loss weights, the model achieved a **57.81% Savings Score** and reduced administrative false alarms to just 5 incidents.
* **Statistical Constraint:** McNemar’s Test between Phase 1a and Phase 3a returned a $p$-value of $0.6636$. Because the severe class imbalance yielded only 98 frauds in the test set, the 21-transaction disagreement between the models was mathematically insufficient to declare algorithmic supremacy with 95% statistical confidence, exposing the structural vulnerability of unsupervised clustering on highly imbalanced data.

---

## 🗂️ Repository Structure

```text
fraud-detection/
├── data/                           # (Gitignored) Raw Kaggle ULB dataset
├── dashboard/
│   └── index.html                  # Interactive HTML/JS Results Dashboard
├── models/                         # Serialized .pkl XGBoost and KMeans models
├── plots/                          # Evaluation visual artifacts (ROC, PR, Confusion Matrices)
├── src/
│   ├── config.py                   # Hyperparameters and global file paths
│   ├── utils.py                    # Console logging and directory management
│   ├── ulb_data_prep.py            # Data loading, 80:20 Stratified Split, SMOTE
│   ├── evaluation.py               # Statistical and Savings Score calculation logic
│   ├── ulb_base_xgboost.py         # Phase 1: Cost-Blind Baseline Training
│   ├── ulb_xgboost_edcsl_phase2.py # Phase 2: Global EDCSL Training
│   ├── ulb_xgboost_edcsl_phase3.py # Phase 3: Category EDCSL (ClusterEnsemble)
│   └── run_mcnemars.py       # McNemar's statistical significance test
└── README.md