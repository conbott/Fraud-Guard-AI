# FraudGuard AI — Security & Privacy Architecture

## Security Principles

FraudGuard AI is designed with strict data privacy, artifact security, and input hygiene policies appropriate for financial transaction processing.

---

## 1. Zero Personal Financial Information Policy
- **No Cardholder Data:** The system strictly refuses and forbids personal account numbers (PAN), credit card numbers, CVVs, expiration dates, cardholder names, or bank credentials.
- **Strict Pydantic Validation:** The API's `TransactionPayload` sets `extra = "forbid"`. Any payload containing unexpected fields (e.g. `card_number` or `cvv`) is instantly rejected with an HTTP 422 error before reaching preprocessing.
- **PCA Anonymization:** Features `V1` through `V28` are anonymized numerical projections. The underlying real-world features are non-recoverable.

---

## 2. Safe Deserialization & Model Artifact Security
- **The Joblib Vulnerability:** Standard Python pickle/joblib deserialization can execute arbitrary code (`os.system`) if fed malicious binary payloads.
- **Defensive Measures:**
  1. The API and Dashboard never accept user-uploaded pickle or joblib files.
  2. The `ModelLoader` validates that the loaded object is a dictionary matching strict schema keys: `{"model", "preprocessor", "algorithm", "version", "optimal_threshold", "feature_names"}`.
  3. Preprocessor instances are validated to confirm they are fitted before inference calls.
  4. Models are stored in local, read-controlled directories (`models/champion/`).

---

## 3. Data Leakage & Git Hygiene
- **Strict `.gitignore` Enforcement:**
  - Raw datasets (`data/raw/*`) and processed feature parquets (`data/processed/*`) are excluded from version control.
  - Large binary artifacts in `models/candidates/*` are excluded.
  - Local MLflow runs (`mlruns/`) and SQLite audit databases are excluded.
  - Environment variables (`.env`) and virtual environments (`.venv/`) are excluded.
- **Synthetic Data for CI:** Automated testing in CI runs strictly against mathematically simulated test fixtures in `data/synthetic/test_fixture.csv`.

---

## 4. API Hardening & Error Handling
- **No Stack Trace Exposure:** Unhandled exceptions are intercepted by FastAPI exception handlers to return sanitized JSON error messages rather than internal server traces.
- **Request Tracing:** Every prediction generates a unique UUID `X-Request-ID` returned in response headers for distributed tracing and audit correlation.
- **Batch Size Limits:** Batch scoring requests are capped at 1,000 transactions to prevent memory exhaustion (DoS).
- **Graceful Degradation:** Failure of telemetry logging (SQLite write locks) fails silently in background tasks and never blocks user prediction responses.
