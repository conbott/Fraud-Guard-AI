# FraudGuard AI — MLOps & ML Engineering Interview Guide

This guide compiles technical questions and rigorous engineering answers designed to help showcase this project in senior and entry-level ML / MLOps engineering interviews.

---

### Q1: Why did you choose PR-AUC over ROC-AUC and Accuracy as your primary evaluation metric?
**Answer:**
"In credit card fraud detection, the dataset exhibits severe class imbalance—fraud accounts for only ~0.17% of transactions (492 out of 284,807). 

- **Accuracy is actively deceptive:** A naive model that classifies every transaction as legitimate achieves 99.83% accuracy while detecting zero frauds (0% recall).
- **ROC-AUC is artificially inflated:** The false positive rate (FPR) denominator is $(FP + TN)$. Because true negatives (TN) are overwhelmingly large (~284,000), a model can make hundreds of false alarms while FPR remains near zero, leading to an artificially high ROC-AUC of 0.95+ despite terrible real-world precision.
- **PR-AUC (Average Precision) isolates the minority class:** Precision is $\frac{TP}{TP + FP}$ and Recall is $\frac{TP}{TP + FN}$. Neither metric includes true negatives. PR-AUC directly measures the trade-off between capturing true fraud and wasting investigator time on false alarms."

---

### Q2: How did you ensure zero data leakage in your preprocessing and evaluation pipeline?
**Answer:**
"Data leakage was systematically prevented at three critical checkpoints:
1. **Pre-split Isolation:** The stratified train/validation/test split is executed *before* any feature scaling or transformation.
2. **Train-Only Fitting:** The `FraudFeaturePreprocessor` (which applies `RobustScaler` and $\ln(1 + \text{Amount})$) is fitted *strictly* on the training partition. The validation and test sets are transformed using the parameters learned from training.
3. **Threshold Tuning on Validation Only:** In imbalanced classification, the default 0.50 threshold is rarely optimal. We tune the decision threshold across a search grid ($0.05$ to $0.95$) strictly on the validation set to maximize F1-score. The tuned threshold is then evaluated on the held-out test set to ensure realistic out-of-sample performance."

---

### Q3: Why did you omit Docker and local virtualization, and how does the system adapt to production?
**Answer:**
"The project was specifically architected to demonstrate Windows-native development and compliance with hardware undervolting constraints (such as ThrottleStop, where hardware virtualization must remain disabled). 

Rather than relying on Docker Desktop as a crutch, we demonstrated clean dependency isolation and platform portability using standard Python virtual environments, strictly pinned dependencies in `pyproject.toml`, and environment-agnostic relative pathing.

In a corporate production cloud migration, containerization is trivial: a two-stage Dockerfile can be added to package FastAPI and Streamlit into OCI containers running on AWS ECS Fargate or Google Cloud Run without altering a single line of application source code."

---

### Q4: How does your drift monitoring work, and what is the difference between Data Drift and Concept Drift?
**Answer:**
"Our drift engine implements:
1. **Population Stability Index (PSI):** Quantile-based metric measuring distribution divergence between baseline training features and incoming transaction batches. We apply standard operational thresholds: $\text{PSI} < 0.10$ is stable, $0.10 \le \text{PSI} < 0.25$ triggers moderate warning, and $\text{PSI} \ge 0.25$ indicates critical drift.
2. **Kolmogorov-Smirnov (KS) 2-Sample Test:** Non-parametric statistical test evaluating cumulative distribution equality with $p$-value threshold $0.05$.

**The Conceptual Distinction:**
- **Data Drift ($P(X)$ change):** The input features change (e.g. transaction amounts increase due to holiday shopping or inflation). The model might still predict accurately if the fraud boundary hasn't shifted.
- **Concept Drift ($P(Y|X)$ change):** The underlying relationship between features and the target label changes (e.g. fraudsters develop novel laundering techniques).
- **Performance Degradation:** The true decline in precision/recall when ground-truth labels are verified.
Data drift alone should never trigger automated model promotion without verified labels and quality gate validation."

---

### Q5: How does your Model Registry ensure safe candidate promotion and rollback?
**Answer:**
"Rather than assuming an expensive managed registry like Databricks or SageMaker, we engineered a deterministic, file-based model registry with explicit quality gates:
1. **Absolute Gates:** Any candidate model must meet $\text{PR-AUC} \ge 0.70$, $\text{Recall} \ge 0.75$, $\text{Precision} \ge 0.60$, and $\text{FPR} \le 0.02$ on held-out test data.
2. **Comparative Benchmark:** The candidate must not degrade PR-AUC compared to the active Champion.
3. **Automated Rollback Backup:** Before promoting a candidate to `models/champion/`, the active champion is automatically archived to `models/previous_champion/`. If production metrics degrade, running `python scripts/retrain.py --rollback` restores the previous champion instantly without downtime."

---

### Q6: How would you scale this architecture from 10 RPS to 10,000 RPS in production?
**Answer:**
"1. **Decouple Ingestion from Serving:** Introduce a distributed message streaming broker (Apache Kafka or AWS Kinesis) to buffer transactions.
2. **Stateless Autoscaling:** Deploy the FastAPI prediction service on AWS ECS Fargate or Kubernetes behind an Application Load Balancer with autoscaling driven by CPU/request queue depth.
3. **Caching Layer:** Cache risk scores for identical customer device fingerprints using Redis with TTLs.
4. **Model Quantization / ONNX Runtime:** Export the trained Scikit-learn model to ONNX runtime, reducing inference latency from ~5ms down to <1ms per transaction."
