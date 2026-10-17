# FraudGuard AI — Data Dictionary

## Overview
The feature space of FraudGuard AI mirrors the benchmark credit card fraud dataset collected by the Machine Learning Group (ULB) on European cardholders in September 2013. The dataset contains 284,807 transactions with 492 frauds, presenting an extreme class imbalance where frauds account for 0.172% of all transactions.

---

## Feature Attributes

| Column Name | Data Type | Value Range | Statistical Description & Semantic Meaning |
|---|---|---|---|
| `Time` | `float64` | `[0.0, 172792.0]` | Seconds elapsed between this transaction and the first transaction in the dataset (spans ~48 hours / 2 days). |
| `Amount` | `float64` | `[0.0, 25691.16]` | Transaction amount in currency ($/€). Highly right-skewed; transformed via $\ln(1 + \text{Amount})$ and scaled with `RobustScaler`. |
| `V1` | `float64` | `[-56.41, 2.45]` | 1st Principal Component derived from PCA. Anonymized bank/cardholder feature projection. |
| `V2` | `float64` | `[-72.72, 22.06]` | 2nd Principal Component derived from PCA. |
| `V3` | `float64` | `[-48.33, 9.38]` | 3rd Principal Component derived from PCA. |
| `V4` | `float64` | `[-5.68, 16.88]` | 4th Principal Component derived from PCA (frequently shifted positive in fraud). |
| `V5` | `float64` | `[-113.74, 34.80]` | 5th Principal Component derived from PCA. |
| `V6` | `float64` | `[-26.16, 73.30]` | 6th Principal Component derived from PCA. |
| `V7` | `float64` | `[-43.56, 120.59]` | 7th Principal Component derived from PCA. |
| `V8` | `float64` | `[-73.22, 20.01]` | 8th Principal Component derived from PCA. |
| `V9` | `float64` | `[-13.43, 15.60]` | 9th Principal Component derived from PCA. |
| `V10` | `float64` | `[-24.59, 23.75]` | 10th Principal Component derived from PCA (strongly negatively shifted in fraud). |
| `V11` | `float64` | `[-4.80, 12.02]` | 11th Principal Component derived from PCA (frequently shifted positive in fraud). |
| `V12` | `float64` | `[-18.68, 7.85]` | 12th Principal Component derived from PCA (strongly negatively shifted in fraud). |
| `V13` | `float64` | `[-5.79, 7.13]` | 13th Principal Component derived from PCA. |
| `V14` | `float64` | `[-19.21, 10.53]` | 14th Principal Component derived from PCA (primary discriminative component in literature). |
| `V15` | `float64` | `[-4.50, 8.88]` | 15th Principal Component derived from PCA. |
| `V16` | `float64` | `[-14.13, 17.32]` | 16th Principal Component derived from PCA. |
| `V17` | `float64` | `[-25.16, 9.25]` | 17th Principal Component derived from PCA (negatively shifted in fraud). |
| `V18` | `float64` | `[-9.50, 5.05]` | 18th Principal Component derived from PCA. |
| `V19` | `float64` | `[-7.21, 5.59]` | 19th Principal Component derived from PCA. |
| `V20` | `float64` | `[-54.50, 39.42]` | 20th Principal Component derived from PCA. |
| `V21` | `float64` | `[-34.83, 27.20]` | 21st Principal Component derived from PCA. |
| `V22` | `float64` | `[-10.93, 10.50]` | 22nd Principal Component derived from PCA. |
| `V23` | `float64` | `[-44.81, 22.53]` | 23rd Principal Component derived from PCA. |
| `V24` | `float64` | `[-2.84, 4.58]` | 24th Principal Component derived from PCA. |
| `V25` | `float64` | `[-10.30, 7.52]` | 25th Principal Component derived from PCA. |
| `V26` | `float64` | `[-2.60, 3.52]` | 26th Principal Component derived from PCA. |
| `V27` | `float64` | `[-22.57, 31.61]` | 27th Principal Component derived from PCA. |
| `V28` | `float64` | `[-15.43, 33.85]` | 28th Principal Component derived from PCA. |
| `Class` | `int64` | `{0, 1}` | Target ground truth label: `0` = Legitimate transaction, `1` = Fraudulent transaction. (Required for training; optional for inference). |

---

## Critical Engineering Notes
1. **PCA Anonymization Integrity**: Due to confidentiality requirements, original raw attributes (e.g., merchant category code, device fingerprint, geographic location, cardholder details) were transformed using PCA before public release.
2. **Never Interpret V1–V28 Arbitrarily**: Features V1 through V28 must never be assigned invented human-readable labels (e.g. "user age" or "zip code"). They are treated purely as mathematical projections.
3. **Duplicate Record Policy**: In credit card transaction streaming, duplicate rows can represent legitimate customer retries, automated micro-charges, or network replay anomalies. FraudGuard flags duplicate rows as telemetry warnings rather than silently discarding them.
