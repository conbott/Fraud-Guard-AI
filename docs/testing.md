# FraudGuard AI — Testing & Quality Assurance Guide

## Testing Philosophy
FraudGuard AI enforces rigorous automated testing across all pipeline components. Tests are completely isolated, fast (running in < 10 seconds), and execute natively on Windows without requiring Docker or the 150MB raw Kaggle dataset.

---

## Test Suite Breakdown

### 1. Unit Tests (`tests/unit/`)
| Test File | Verified Functionality | Status |
|---|---|---|
| `test_validation.py` | Schema enforcement, missing columns, nulls, infs, negative amount/time, duplicate alerts, and target label validation. | **PASSED (8/8)** |
| `test_preprocessing.py` | RobustScaler & log1p transforms, transform consistency, single dict/list batch input handling, and missing feature error catching. | **PASSED (4/4)** |
| `test_evaluation.py` | Imbalanced classification metrics, confusion matrix elements, validation threshold tuning, PR/ROC curve calculation. | **PASSED (3/3)** |
| `test_pipeline_and_model.py` | End-to-end `ModelTrainer` pipeline, baseline comparison, Joblib serialization, safe loading, and single/batch inference. | **PASSED (2/2)** |
| `test_monitoring.py` | PSI algorithm correctness (stable vs shifted), KS 2-sample test, synthetic drift generator, and SQLite audit telemetry logging. | **PASSED (4/4)** |
| `test_retraining_and_promotion.py` | Candidate quality gate evaluation (pass/fail), champion promotion, automatic backup, and deterministic rollback. | **PASSED (2/2)** |
| `test_dashboard.py` | Streamlit dashboard module importability, model loading, and predictor initialization without syntax or import errors. | **PASSED (2/2)** |

### 2. API Integration Tests (`tests/integration/`)
| Test File | Verified Endpoints | Status |
|---|---|---|
| `test_api.py` | `GET /health`, `GET /model-info`, `POST /predict`, `POST /predict/batch`, `GET /metrics`, 422 validation error handling, header injection. | **PASSED (8/8)** |

---

## Running the Automated Tests

### 1. Run Complete Test Suite
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

### 2. Run Tests with Code Coverage Report
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v --cov=src --cov-report=term-missing
```

### 3. Run Ruff Code Linter
```powershell
.\.venv\Scripts\python.exe -m ruff check src scripts tests
```

### 4. Run Model Training Smoke Test
```powershell
.\.venv\Scripts\python.exe scripts/train.py --data-path data/synthetic/test_fixture.csv
```

### 5. Run Drift Monitoring Smoke Test
```powershell
.\.venv\Scripts\python.exe scripts/run_monitoring.py --data-path data/synthetic/test_fixture.csv --simulate-drift amount_spike
```

---

## Test Fixture Strategy
- **`data/synthetic/test_fixture.csv`**: A small, highly predictable 300-row fixture with 5% fraud prevalence generated with fixed seed `42`. Used for rapid CI smoke testing.
- **`data/synthetic/synthetic_transactions.csv`**: A 3,000-row dataset with 2% fraud prevalence used for drift simulation and retraining validation.
- **Isolation Guarantee**: Tests write temporary artifacts into `tmp_path` fixtures to ensure tests never mutate production model artifacts.
