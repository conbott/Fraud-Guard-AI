# Model Card: FraudGuard AI Imbalanced Fraud Classifier

## 1. Model Details
- **Model Name:** FraudGuard AI Credit Card Fraud Classifier
- **Model Version:** model_v20261017_164934 (Active Champion)
- **Model Architecture:** Random Forest Classifier (`n_estimators=100`, `max_depth=10`, `class_weight='balanced'`, `min_samples_split=10`, `min_samples_leaf=4`)
- **Compared Architectures:** Stratified Dummy Baseline, L2-Regularized Logistic Regression (`C=1.0`, `class_weight='balanced'`)
- **Framework & Libraries:** Scikit-Learn 1.6+, Python 3.11/3.13, NumPy, Pandas, Joblib
- **Release Date:** October 2026
- **Maintainer:** conbott / FraudGuard AI Team

---

## 2. Intended Use
- **Primary Use Case:** Real-time risk scoring, transaction triage, and automated prioritization of suspicious card-not-present and card-present payment transactions for fraud operations teams.
- **Intended Users:** Payment operations analysts, fraud investigators, risk management teams, and transaction scoring engines.
- **Out-of-Scope Uses:**
  - Automated unilateral card cancellation or permanent account termination without secondary human review or customer step-up challenge (e.g. 3D Secure / SMS OTP).
  - Cross-domain application to cryptocurrency settlement, anti-money laundering (AML) structuring, or loan default prediction without domain-specific features and retraining.
  - Deployment in production banking core systems without latency auditing, regulatory compliance clearance (e.g. Fair Credit Reporting Act), and regional data residency adherence.

---

## 3. Training Data & Preprocessing
- **Primary Dataset:** Kaggle MLG-ULB Credit Card Fraud Detection (Université Libre de Bruxelles).
- **Data Volume:** 284,807 total European cardholder transactions across two consecutive days in September 2013.
- **Label Distribution:** 492 fraudulent transactions (0.1727% positive prevalence; severe class imbalance of ~578 legitimate transactions per 1 fraud).
- **Feature Schema:**
  - `Time`: Elapsed seconds between the transaction and the first transaction in the dataset.
  - `V1` to `V28`: Numerical feature projections obtained via Principal Component Analysis (PCA) to preserve cardholder privacy.
  - `Amount`: Transaction transaction amount in local currency (EUR).
  - `Class`: Binary ground truth target (0 = Legitimate, 1 = Fraudulent).
- **Leakage-Safe Preprocessing Pipeline:**
  - Partitioning into 60% Train, 20% Validation, and 20% Held-Out Test BEFORE any feature transformations.
  - `RobustScaler` is fitted exclusively on the training partition for `Time` and `Amount` to handle extreme outliers without leaking distributional parameters.
  - Natural logarithmic transformation `log1p(Amount)` applied strictly after scaling.
  - PCA features `V1` through `V28` are preserved as pre-orthogonalized inputs with median imputation for missing values (0 missing values present in real dataset).

---

## 4. Evaluation Methodology & Baseline Comparisons
To ensure complete data integrity, models are evaluated across both **Stratified** and **Chronological** splits. The decision threshold is tuned strictly on the **Validation partition** to maximize F1-score across $[0.05, 0.95]$ in increments of $0.01$. The held-out **Test partition** is evaluated blindly at the chosen threshold.

### Comprehensive Benchmark Comparison (Real MLG-ULB Dataset, 284,807 Transactions)

#### A. Stratified Partitioning (60% Train: 170,883 | 20% Val: 56,962 | 20% Test: 56,962)
| Model / Algorithm | Fit Time (s) | Tuned Threshold | Val PR-AUC | Val ROC-AUC | Val F1 | Test PR-AUC | Test ROC-AUC | Test Precision | Test Recall | Test F1 | Test FPR | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Dummy Baseline (Stratified)** | 0.01 | 0.05 | 0.0017 | 0.4992 | 0.0000 | 0.0017 | 0.4992 | 0.0000 | 0.0000 | 0.0000 | 0.001635 | 0 | 93 | 56,771 | 98 |
| **Logistic Regression (`balanced`)** | 0.37 | 0.95 | 0.6761 | 0.9749 | 0.4233 | 0.7168 | 0.9724 | 0.3000 | 0.8878 | 0.4485 | 0.003570 | 87 | 203 | 56,661 | 11 |
| **Random Forest (Champion)** | 9.83 | 0.43 | **0.7693** | 0.9593 | **0.8280** | **0.8140** | 0.9717 | **0.8081** | 0.8163 | **0.8122** | **0.000334** | 80 | 19 | 56,845 | 18 |

#### B. Chronological Partitioning (Ordered by `Time`: First 60% Train | Middle 20% Val | Final 20% Test)
| Model / Algorithm | Fit Time (s) | Tuned Threshold | Val PR-AUC | Val ROC-AUC | Val F1 | Test PR-AUC | Test ROC-AUC | Test Precision | Test Recall | Test F1 | Test FPR | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Dummy Baseline (Stratified)** | 0.01 | 0.05 | 0.0010 | 0.4990 | 0.0000 | 0.0013 | 0.4990 | 0.0000 | 0.0000 | 0.0000 | 0.002039 | 0 | 116 | 56,770 | 75 |
| **Logistic Regression (`balanced`)** | 0.35 | 0.95 | 0.7511 | 0.9692 | 0.3498 | 0.7442 | 0.9817 | 0.3520 | 0.8400 | 0.4961 | 0.002039 | 63 | 116 | 56,770 | 12 |
| **Random Forest** | 9.81 | 0.34 | **0.7508** | 0.9637 | **0.8350** | **0.8151** | 0.9547 | **0.8906** | 0.7600 | **0.8201** | **0.000123** | 57 | 7 | 56,879 | 18 |

---

## 5. Why Precision-Recall AUC Governs Imbalanced Fraud Detection
In highly skewed environments (99.83% negative class), standard accuracy and ROC-AUC can provide dangerously misleading assessments:
1. **The Accuracy Paradox:** A trivial model predicting `Class = 0` for all transactions achieves 99.83% accuracy while detecting 0 frauds (Recall = 0.0). Accuracy evaluates true negatives equally with true positives, which masks complete minority-class failure.
2. **ROC-AUC Insensitivity:** ROC-AUC plots True Positive Rate vs False Positive Rate. Because the denominator for FPR is the colossal negative class ($N = 56,864$), even 200 false alarms yield an FPR of only $0.0035$, keeping the ROC curve near the top-left corner ($ROC-AUC > 0.97$).
3. **PR-AUC Dominance:** Precision-Recall AUC focuses exclusively on positive predictions and true positive captures. It directly measures how many genuine alerts are generated for each false positive. Random guessing yields a baseline PR-AUC equal to the positive prevalence ($0.0017$). Achieving $PR-AUC = 0.8140$ confirms that high precision is maintained even as recall is driven above 80%.

---

## 6. Ethical Considerations & Operational Economics
1. **Privacy Safeguards:** Feature sets $V_1$ through $V_{28}$ are anonymized PCA projections. No personally identifiable information (PII), names, card numbers, physical addresses, or merchant names are ingested or logged.
2. **False Positive Economics (Customer Friction):** Declining or challenging legitimate cardholder transactions damages card usage, brand loyalty, and merchant conversion rates. The Champion Random Forest limits test false alarms to 19 out of 56,864 transactions (False Positive Rate = 0.033%), preserving customer trust.
3. **False Negative Economics (Chargeback Loss):** Undetected fraud incurs direct merchant write-offs, dispute fees ($15-$50 per chargeback), and interchange penalization. The model successfully detects 80 out of 98 test fraud events ($81.63\%$ recall), mitigating over 80% of aggregate loss.
4. **Fairness & Demographic Neutrality:** Because demographic proxies (age, gender, ethnicity) are excluded, disparate impact across customer segments is structurally suppressed.

---

## 7. Limitations & Operational Risks
1. **Short Observation Window:** The ULB dataset captures transactions within a 48-hour continuous window. Long-term seasonal cycles (e.g. quarterly spending surges, Cyber Monday, holiday travel) are not represented in the baseline weights.
2. **Adversarial Adaptation:** Fraudsters continuously modify attack patterns (e.g., structuring charges under automated alert thresholds, switching merchant categories). Models require periodic retraining and drift monitoring.
3. **Extreme Outliers:** Unseen transaction amounts significantly exceeding training distributions can induce high model uncertainty. Input validation enforces bounded caps and alerts on extreme values.
4. **Hardware and Latency Profile:** Random Forest inference executes within 1.2 milliseconds per transaction on standard x86 CPU hardware, making it well-suited for lightweight API endpoints without requiring GPU accelerators.

---

## 8. Monitoring, Quality Gates & Rollback
1. **Automated Quality Gates:** Any retrained candidate model must pass stringent promotion criteria before replacing the Champion:
   - PR-AUC $\ge 0.70$
   - Recall $\ge 0.75$
   - Precision $\ge 0.60$
   - False Positive Rate $\le 0.02$
   - Non-Degradation Gate: Candidate PR-AUC must meet or exceed the active Champion.
2. **Drift Monitoring:** Production batches are monitored using the **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS) two-sample test** against the baseline reference distribution.
   - $PSI < 0.10$: Stable (No drift)
   - $0.10 \le PSI < 0.25$: Moderate shift (Warning)
   - $PSI \ge 0.25$: Severe drift (Automated retraining triggered)
3. **Disaster Recovery Rollback:** The file-based registry retains timestamped backups of preceding champions. If deployment anomalies occur, `scripts/retrain.py --rollback` or the Streamlit dashboard provides single-command rollback with zero downtime.
