# FraudGuard AI - Data Directory

This directory stores transaction datasets for FraudGuard AI.

## Directory Layout
- `raw/`: Place the real Kaggle Credit Card Fraud Detection dataset here as `creditcard.csv`.
- `processed/`: Intermediate split datasets and reference features saved in Parquet/CSV formats.
- `synthetic/`: Synthetic transaction datasets generated locally for unit/integration testing and CI/CD pipelines.

## How to Download the Kaggle Dataset
1. Visit the Kaggle dataset page:
   [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Click **Download** (requires free Kaggle account).
3. Unzip the downloaded archive `creditcardfraud.zip`.
4. Place the extracted `creditcard.csv` into `data/raw/creditcard.csv`.
5. Run the validation script to verify its schema and checksum:
   ```powershell
   .\.venv\Scripts\python.exe scripts/validate_data.py --data-path data/raw/creditcard.csv
   ```

## Privacy & Security Safeguards
- **Zero Raw Data in Version Control**: `data/raw/*` and `data/processed/*` are strictly ignored by `.gitignore`.
- **No Sensitive Personal Information**: The Kaggle dataset features `V1` through `V28` are anonymized PCA components. `Time` and `Amount` contain transaction elapsed time and transaction amounts. No credit card numbers, CVVs, cardholder names, or bank credentials exist in this dataset or are processed by this project.
- **Synthetic Data for CI**: Automated tests and CI pipelines rely solely on reproducible synthetic fixtures in `data/synthetic/` and never require the raw 150MB Kaggle CSV.
