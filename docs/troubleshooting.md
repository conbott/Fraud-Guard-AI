# FraudGuard AI — Troubleshooting & FAQ Guide

## Common Issues & Solutions

### 1. PowerShell Script Execution Policy Error
**Symptom:** Running `.\.venv\Scripts\Activate.ps1` produces an error:
`File ... cannot be loaded because running scripts is disabled on this system.`

**Solution:** PowerShell restricts script execution by default. Run the following command in your current PowerShell session to enable local script execution without modifying system-wide settings:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```
Then re-run:
```powershell
.\.venv\Scripts\Activate.ps1
```

---

### 2. Missing Champion Model Artifact
**Symptom:** Starting FastAPI or Streamlit throws:
`FileNotFoundError: Model artifact not found at models/champion/model.joblib.`

**Solution:** Generate test fixtures and train the initial champion model:
```powershell
.\.venv\Scripts\python.exe scripts/generate_synthetic_data.py --create-fixtures
.\.venv\Scripts\python.exe scripts/train.py --champion
```

---

### 3. Port Conflicts (`8000` or `8501` already in use)
**Symptom:** `error: [Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)`

**Solution:** Run on alternative ports:
- For FastAPI:
  ```powershell
  .\.venv\Scripts\uvicorn.exe src.api.app:app --host 127.0.0.1 --port 8080
  ```
- For Streamlit:
  ```powershell
  .\.venv\Scripts\streamlit.exe run app.py --server.port 8502
  ```

---

### 4. Raw Kaggle Dataset Not Found
**Symptom:** `FileNotFoundError: Dataset not found at data/raw/creditcard.csv.`

**Solution:** 
1. Download `creditcardfraud.zip` from Kaggle.
2. Unzip and place `creditcard.csv` in `data/raw/creditcard.csv`.
3. Alternatively, for local testing and CI, the system will seamlessly use synthetic data in `data/synthetic/synthetic_transactions.csv`.

---

### 5. MLflow Tracking URI Windows Path Issues
**Symptom:** `mlflow.exceptions.MlflowException: The path 'C:\...' is not a valid URI.`

**Solution:** FraudGuard AI automatically formats local file URIs as standard POSIX paths (`file:///C:/Users/...`) in `src/training/tracking.py` to prevent Windows backslash escaping errors.
