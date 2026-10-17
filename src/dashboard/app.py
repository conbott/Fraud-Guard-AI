"""Streamlit interactive dashboard for FraudGuard AI."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_model_config
from src.data.validator import DataValidator
from src.inference.loader import ModelLoader
from src.inference.predictor import FraudPredictor
from src.monitoring.drift import DriftMonitor, simulate_synthetic_drift
from src.training.registry import ModelRegistry

# Representative benchmark samples verified against model weights
NORMAL_SAMPLE = {
    "Time": 30689.13,
    "Amount": 42.15,
    "V1": -0.2212,
    "V2": 0.8682,
    "V3": -1.3820,
    "V4": -1.0593,
    "V5": -1.2879,
    "V6": 0.0653,
    "V7": -0.5174,
    "V8": -0.4487,
    "V9": 0.6447,
    "V10": 0.2411,
    "V11": 0.0287,
    "V12": -0.3699,
    "V13": -1.5692,
    "V14": -0.0921,
    "V15": 0.5103,
    "V16": 1.7553,
    "V17": -1.0942,
    "V18": -0.3446,
    "V19": -0.0536,
    "V20": 0.0560,
    "V21": 0.5510,
    "V22": 0.4896,
    "V23": 1.5562,
    "V24": 0.2421,
    "V25": 1.6089,
    "V26": -0.2427,
    "V27": -0.8958,
    "V28": 0.0665,
}

FRAUD_SAMPLE = {
    "Time": 170785.65,
    "Amount": 19.77,
    "V1": 1.4028,
    "V2": 2.7834,
    "V3": -1.1178,
    "V4": 4.3958,
    "V5": -1.9659,
    "V6": -0.9549,
    "V7": -0.3647,
    "V8": 0.3952,
    "V9": -1.6138,
    "V10": -0.3692,
    "V11": 1.0590,
    "V12": -4.2741,
    "V13": -0.3656,
    "V14": -3.4621,
    "V15": -1.9267,
    "V16": 0.4420,
    "V17": -1.4720,
    "V18": -0.3474,
    "V19": 2.0083,
    "V20": -0.3250,
    "V21": -1.4320,
    "V22": 0.7044,
    "V23": -0.0663,
    "V24": 0.2332,
    "V25": -0.9632,
    "V26": -1.2486,
    "V27": -1.6551,
    "V28": -0.6798,
}

# Page configuration
st.set_page_config(
    page_title="FraudGuard AI - MLOps Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_model_bundle():
    """Load champion model artifact with caching."""
    try:
        return ModelLoader.load_champion()
    except Exception as e:
        st.error(f"Failed to load champion model: {e}")
        return None


@st.cache_resource
def get_predictor():
    """Instantiate fraud risk predictor."""
    bundle = get_model_bundle()
    if bundle:
        return FraudPredictor(loaded_model=bundle)
    return None


def main():
    config = load_model_config()
    bundle = get_model_bundle()
    predictor = get_predictor()

    # Prevent Streamlit metric value text truncation with ellipsis
    st.markdown(
        """
        <style>
        div[data-testid="stMetricValue"] {
            font-size: 1.45rem !important;
            white-space: normal !important;
            word-break: normal !important;
            line-height: 1.25 !important;
        }
        div[data-testid="stMetricValue"] > div {
            white-space: normal !important;
            text-overflow: unset !important;
            overflow: visible !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    algo_display = bundle.algorithm.replace("_", " ").title() if bundle else "None"

    st.sidebar.title("FraudGuard AI")
    st.sidebar.caption("End-to-End MLOps Platform")

    nav_option = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Transaction Prediction",
            "Model Evaluation",
            "Data Quality & Schema",
            "Drift Monitoring",
            "Model Registry & Gating",
            "System Architecture",
        ],
    )

    meta = bundle.metadata if bundle else {}
    prov = meta.get("provenance", {})
    is_synth = prov.get("is_synthetic", True)
    prov_label = "Kaggle MLG-ULB (Real Data)" if not is_synth else "Synthetic Benchmark"

    st.sidebar.divider()
    if bundle:
        st.sidebar.markdown(f"**Model Chosen:**\n\n`{algo_display}`")
        st.sidebar.markdown(f"**Version:**\n\n`{bundle.version}`")
        st.sidebar.markdown(f"**Dataset Provenance:**\n\n`{prov_label}`")
        st.sidebar.markdown(f"**Decision Threshold:**\n\n`{bundle.optimal_threshold:.2f}`")
    else:
        st.sidebar.warning("No Champion Model Found")

    # =========================================================================
    # 1. OVERVIEW
    # =========================================================================
    if nav_option == "Overview":
        st.title("FraudGuard AI — MLOps Overview")
        st.markdown(
            "> **Notice:** FraudGuard AI is an end-to-end credit card fraud detection and MLOps portfolio platform "
            "designed for risk scoring on extreme class imbalances (~0.17% fraud prevalence). "
            "Features `V1` through `V28` are anonymized PCA components. No personal financial data is stored."
        )

        if not bundle:
            st.error("Champion model artifact is missing. Please run `python scripts/train.py --champion`.")
            return

        meta = bundle.metadata
        test_m = meta.get("test_metrics", {})
        prov = meta.get("provenance", {})
        is_synth = prov.get("is_synthetic", True)

        if not is_synth:
            st.success(
                f"**Dataset Provenance Verified:** {prov.get('dataset_name', 'MLG-ULB Credit Card Fraud Detection')} "
                f"(Real Kaggle Dataset, {prov.get('total_samples', 284807):,} transactions, "
                f"{prov.get('fraud_samples', 492):,} fraud events, {prov.get('prevalence', 0.001727)*100:.3f}% prevalence, "
                f"Stratified Split: 60% Train / 20% Val / 20% Held-Out Test)."
            )
        else:
            st.info("**Dataset Provenance:** Synthetic benchmark demonstration fixture.")

        col1, col2, col3, col4, col5 = st.columns([1.6, 1, 1, 1, 1])
        with col1:
            st.markdown(
                f"""
                <div style="padding: 1px 0;">
                    <div style="font-size: 0.875rem; color: #9aa0a6; margin-bottom: 2px;">Model Algorithm</div>
                    <div style="font-size: 1.5rem; font-weight: 600; line-height: 1.2;">{algo_display}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        col2.metric("PR-AUC (Test)", f"{test_m.get('pr_auc', 0.0):.4f}")
        col3.metric("F1-Score (Test)", f"{test_m.get('f1_score', 0.0):.4f}")
        col4.metric("Recall (Test)", f"{test_m.get('recall', 0.0):.4f}")
        col5.metric("Decision Threshold", f"{bundle.optimal_threshold:.2f}")

        st.subheader("Key Platform Capabilities")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Leakage-Safe Pipeline")
            st.write(
                "- Stratified train/val/test splits prior to any transformation.\n"
                "- RobustScaler & Log1p fitted strictly on training partition.\n"
                "- Optimal decision threshold tuned strictly on validation set."
            )
        with c2:
            st.markdown("### Imbalanced Evaluation")
            st.write(
                "- Evaluates precision, recall, F1, PR-AUC, and FPR.\n"
                "- Accuracy is explicitly flagged as misleading on imbalanced data.\n"
                "- Compares Dummy Baseline, Logistic Regression, and Random Forest."
            )
        with c3:
            st.markdown("### Automated MLOps")
            st.write(
                "- File-based Model Registry with candidate promotion gates.\n"
                "- Instant rollback capability to previous champion.\n"
                "- Population Stability Index (PSI) and KS-test drift alarms."
            )

        st.divider()
        st.subheader("Model Artifact Metadata & Provenance")
        st.json(
            {
                "version": bundle.version,
                "algorithm": bundle.algorithm,
                "timestamp": meta.get("timestamp"),
                "dataset_fingerprint": meta.get("dataset_fingerprint"),
                "provenance": prov,
                "random_seed": meta.get("random_seed"),
                "feature_count": len(bundle.feature_names),
                "environment": meta.get("environment"),
            }
        )

    # =========================================================================
    # 2. TRANSACTION PREDICTION
    # =========================================================================
    elif nav_option == "Transaction Prediction":
        st.title("Transaction Fraud Prediction")
        st.write(
            "Perform single transaction risk scoring or upload a CSV batch for automated evaluation."
        )

        tab_single, tab_batch = st.tabs(["Single Transaction Scoring", "Batch CSV Scoring"])

        with tab_single:
            st.markdown("#### Enter Transaction Parameters")

            # Initialize session state with normal sample defaults if not present
            if "input_time" not in st.session_state:
                st.session_state["input_time"] = float(NORMAL_SAMPLE["Time"])
                st.session_state["input_amount"] = float(NORMAL_SAMPLE["Amount"])
                for i in range(1, 29):
                    st.session_state[f"input_V{i}"] = float(NORMAL_SAMPLE[f"V{i}"])

            # Presets with clean professional button styling
            st.caption(
                "Benchmark presets: Note that these sample profiles are synthetic demonstration fixtures "
                "modeled on MLG-ULB PCA feature distributions. They allow testing the risk scoring engine "
                "without exposing confidential banking records."
            )
            preset_col1, preset_col2, preset_col3 = st.columns(3)

            if preset_col1.button("Load Normal Sample (Synthetic Demo)", use_container_width=True):
                st.session_state["input_time"] = float(NORMAL_SAMPLE["Time"])
                st.session_state["input_amount"] = float(NORMAL_SAMPLE["Amount"])
                for i in range(1, 29):
                    st.session_state[f"input_V{i}"] = float(NORMAL_SAMPLE[f"V{i}"])
                st.rerun()

            if preset_col2.button("Load Fraud-like Sample (Synthetic Demo)", use_container_width=True):
                st.session_state["input_time"] = float(FRAUD_SAMPLE["Time"])
                st.session_state["input_amount"] = float(FRAUD_SAMPLE["Amount"])
                for i in range(1, 29):
                    st.session_state[f"input_V{i}"] = float(FRAUD_SAMPLE[f"V{i}"])
                st.rerun()

            if preset_col3.button("Generate Random Synthetic Sample", use_container_width=True):
                rng = np.random.default_rng()
                st.session_state["input_time"] = float(np.round(rng.uniform(100.0, 150000.0), 2))
                st.session_state["input_amount"] = float(np.round(rng.exponential(50.0), 2))
                for i in range(1, 29):
                    st.session_state[f"input_V{i}"] = float(np.round(rng.normal(0.0, 1.2), 4))
                st.rerun()

            c_t, c_a = st.columns(2)
            time_val = c_t.number_input(
                "Time (Seconds elapsed)",
                min_value=0.0,
                format="%.2f",
                key="input_time",
            )
            amount_val = c_a.number_input(
                "Amount ($ USD)",
                min_value=0.0,
                format="%.2f",
                key="input_amount",
            )

            with st.expander("PCA Anonymized Features (V1 to V28)", expanded=False):
                st.caption(
                    "Features V1–V28 are PCA transformed components from the original banking transactions. "
                    "Their numerical values represent variance projections."
                )
                pca_inputs = {}
                cols = st.columns(4)
                for idx, v_name in enumerate([f"V{i}" for i in range(1, 29)]):
                    col_target = cols[idx % 4]
                    pca_inputs[v_name] = col_target.number_input(
                        v_name,
                        step=0.1,
                        format="%.4f",
                        key=f"input_{v_name}",
                    )

            if st.button("Evaluate Transaction Risk", type="primary"):
                payload = {"Time": time_val, "Amount": amount_val, **pca_inputs}
                try:
                    res = predictor.predict_single(payload)
                    st.divider()

                    r1, r2, r3 = st.columns(3)
                    if res.is_fraud:
                        r1.error("**SUSPECTED FRAUD DETECTED**")
                    else:
                        r1.success("**LEGITIMATE TRANSACTION**")

                    r2.metric("Calibrated Risk Score", f"{res.risk_score:.4f}")
                    r3.metric("Decision Threshold", f"{res.threshold:.4f}")

                    st.progress(res.risk_score)

                    st.info(f"Risk Tier: `{res.risk_level}` | Model: `{res.algorithm.replace('_', ' ').title()}` ({res.model_version})")
                    st.warning(f"Notice: {res.cautionary_notice}")

                except Exception as e:
                    st.error(f"Inference error: {e}")

        with tab_batch:
            st.markdown("#### Batch CSV Evaluation (Max 1,000 records)")
            uploaded_file = st.file_uploader("Upload CSV matching schema", type=["csv"])

            if uploaded_file is not None:
                df_upload = pd.read_csv(uploaded_file)
                st.write(f"Uploaded `{len(df_upload)}` rows.")

                validator = DataValidator()
                val_res = validator.validate(df_upload, is_training=False, min_rows=1)

                if not val_res.is_valid:
                    st.error(f"Validation failed with {len(val_res.errors)} errors:")
                    for err in val_res.errors:
                        st.write(f"- {err.message}")
                else:
                    st.success("CSV Schema validated successfully.")
                    if len(df_upload) > 1000:
                        st.warning("Displaying and scoring first 1,000 records to respect memory limits.")
                        df_upload = df_upload.head(1000)

                    if st.button("Run Batch Scoring"):
                        with st.spinner("Processing batch predictions..."):
                            predictions = predictor.predict_batch(df_upload)
                            scores = [p.risk_score for p in predictions]
                            flags = [p.is_fraud for p in predictions]
                            tiers = [p.risk_level for p in predictions]

                            df_result = df_upload.copy()
                            df_result.insert(0, "is_fraud_suspected", flags)
                            df_result.insert(1, "risk_score", scores)
                            df_result.insert(2, "risk_level", tiers)

                            b_c1, b_c2, b_c3 = st.columns(3)
                            fraud_cnt = sum(flags)
                            b_c1.metric("Total Scored", f"{len(df_result):,}")
                            b_c2.metric("Suspected Frauds", f"{fraud_cnt} ({fraud_cnt/len(df_result)*100:.2f}%)")
                            b_c3.metric("Avg Risk Score", f"{np.mean(scores):.4f}")

                            st.dataframe(df_result.head(50), use_container_width=True)

                            csv_data = df_result.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Download Scored Batch Results (CSV)",
                                data=csv_data,
                                file_name="fraudguard_batch_predictions.csv",
                                mime="text/csv",
                            )

    # =========================================================================
    # 3. MODEL EVALUATION
    # =========================================================================
    elif nav_option == "Model Evaluation":
        st.title("Model Evaluation & Baseline Comparison")
        if not bundle:
            st.error("No champion model found.")
            return

        meta = bundle.metadata
        comparison = meta.get("model_comparison", {})
        test_m = meta.get("test_metrics", {})

        st.subheader("Model Benchmark Comparison")
        st.markdown(
            "Comparing baseline vs machine learning algorithms on identical partitions. "
            "**PR-AUC** is the primary criterion for imbalanced ranking, and decision thresholds are tuned strictly on validation data."
        )

        eval_view = st.radio(
            "Select Benchmark View:",
            [
                "MLG-ULB Stratified Split (60% Train / 20% Val / 20% Test)",
                "MLG-ULB Chronological Split (Ordered by Time)",
                "IEEE-CIS Fraud Detection (Advanced Offline Experiment)",
            ],
            horizontal=True,
        )

        csv_comp_path = PROJECT_ROOT / "models" / "reports" / "ulb_benchmark_comparison.csv"
        if eval_view == "MLG-ULB Stratified Split (60% Train / 20% Val / 20% Test)":
            if csv_comp_path.exists():
                df_bench = pd.read_csv(csv_comp_path)
                df_strat = df_bench[df_bench["split_strategy"] == "stratified"].copy()
                st.dataframe(df_strat, use_container_width=True)
            else:
                comp_data = []
                for name, metrics in comparison.items():
                    comp_data.append({
                        "Algorithm": name.replace("_", " ").title(),
                        "Val PR-AUC": metrics.get("val_pr_auc"),
                        "Val ROC-AUC": metrics.get("val_roc_auc"),
                        "Val F1 (Tuned)": metrics.get("val_f1_tuned"),
                        "Val Recall": metrics.get("val_recall_tuned"),
                        "Val Precision": metrics.get("val_precision_tuned"),
                        "Optimal Threshold": metrics.get("optimal_threshold"),
                    })
                if comp_data:
                    st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

        elif eval_view == "MLG-ULB Chronological Split (Ordered by Time)":
            if csv_comp_path.exists():
                df_bench = pd.read_csv(csv_comp_path)
                df_chrono = df_bench[df_bench["split_strategy"] == "chronological"].copy()
                st.dataframe(df_chrono, use_container_width=True)
                st.caption(
                    "Chronological Split Analysis: Models are trained on the first 60% of timeline, "
                    "threshold-tuned on the middle 20%, and evaluated on the final 20%. "
                    "This evaluates robustness against non-stationary temporal shifts in transaction patterns."
                )
            else:
                st.info("Chronological benchmark report not found at models/reports/ulb_benchmark_comparison.csv.")

        elif eval_view == "IEEE-CIS Fraud Detection (Advanced Offline Experiment)":
            ieee_report_path = PROJECT_ROOT / "models" / "reports" / "ieee_cis_evaluation.json"
            if ieee_report_path.exists():
                import json
                with open(ieee_report_path, "r", encoding="utf-8") as f:
                    ieee_data = json.load(f)
                st.markdown(f"**Dataset:** `{ieee_data.get('dataset')}` | **Split Method:** `{ieee_data.get('split_method')}`")
                ieee_rows = []
                for m_name, m_vals in ieee_data.get("models", {}).items():
                    ieee_rows.append({
                        "Model": m_name.replace("_", " ").title(),
                        "PR-AUC": m_vals.get("pr_auc"),
                        "ROC-AUC": m_vals.get("roc_auc"),
                        "F1-Score": m_vals.get("f1"),
                        "Recall": m_vals.get("recall"),
                        "Precision": m_vals.get("precision"),
                        "Optimal Threshold": m_vals.get("optimal_threshold", 0.5),
                    })
                st.dataframe(pd.DataFrame(ieee_rows), use_container_width=True)
                st.caption("Note: IEEE-CIS is evaluated as a separate offline experiment with memory safeguards and chronological splitting on TransactionDT.")
            else:
                st.info("IEEE-CIS experiment report not yet generated. Run: python scripts/run_ieee_cis_experiment.py --mock")

        st.divider()
        st.subheader("Held-Out Test Set Evaluation (Champion)")
        st.caption(f"Evaluated at optimal tuned decision threshold: **{bundle.optimal_threshold:.2f}**")

        tm1, tm2, tm3, tm4, tm5, tm6 = st.columns(6)
        tm1.metric("Test PR-AUC", f"{test_m.get('pr_auc', 0.0):.4f}")
        tm2.metric("Test ROC-AUC", f"{test_m.get('roc_auc', 0.0):.4f}")
        tm3.metric("Precision", f"{test_m.get('precision', 0.0):.4f}")
        tm4.metric("Recall", f"{test_m.get('recall', 0.0):.4f}")
        tm5.metric("F1-Score", f"{test_m.get('f1_score', 0.0):.4f}")
        tm6.metric("False Pos. Rate", f"{test_m.get('false_positive_rate', 0.0):.6f}")

        col_cm, col_curves = st.columns([1, 2])

        with col_cm:
            st.markdown("#### Confusion Matrix")
            tp = test_m.get("true_positives", 0)
            fp = test_m.get("false_positives", 0)
            tn = test_m.get("true_negatives", 0)
            fn = test_m.get("false_negatives", 0)

            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            cm_matrix = np.array([[tn, fp], [fn, tp]])
            sns.heatmap(
                cm_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Pred Normal", "Pred Fraud"],
                yticklabels=["Actual Normal", "Actual Fraud"],
                ax=ax_cm,
            )
            ax_cm.set_title("Test Confusion Matrix")
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with col_curves:
            st.markdown("#### Precision-Recall & ROC Curves")
            curve_points = meta.get("curve_points", {})
            pr_data = curve_points.get("pr_curve", {})
            roc_data = curve_points.get("roc_curve", {})

            if pr_data and roc_data:
                fig_c, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(8, 3.2))

                # PR Curve
                ax_pr.plot(pr_data["recall"], pr_data["precision"], color="#2b5c8f", lw=2)
                ax_pr.set_title(f"PR Curve (AUC = {test_m.get('pr_auc', 0.0):.3f})")
                ax_pr.set_xlabel("Recall")
                ax_pr.set_ylabel("Precision")
                ax_pr.grid(True, linestyle="--", alpha=0.5)

                # ROC Curve
                ax_roc.plot(roc_data["fpr"], roc_data["tpr"], color="#2ca02c", lw=2)
                ax_roc.plot([0, 1], [0, 1], color="gray", linestyle="--")
                ax_roc.set_title(f"ROC Curve (AUC = {test_m.get('roc_auc', 0.0):.3f})")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.grid(True, linestyle="--", alpha=0.5)

                st.pyplot(fig_c)
                plt.close(fig_c)
            else:
                st.info("Curve points not available in metadata bundle.")

        st.info(
            "**Evaluation Note: Why PR-AUC Matters More Than Accuracy:** In a transaction stream with 99.83% legitimate transactions, "
            "a model predicting all zeros achieves 99.83% accuracy and 0% fraud recall. Precision-Recall AUC isolates "
            "the minority class performance, showing exactly how many genuine alerts are generated per false positive."
        )

    # =========================================================================
    # 4. DATA QUALITY & SCHEMA
    # =========================================================================
    elif nav_option == "Data Quality & Schema":
        st.title("Data Quality & Schema Validation")
        st.write("Inspect schema health, duplicate transaction policy, and data distributions.")

        data_source = st.radio(
            "Select Data Source to Validate:",
            [
                "Upload Custom CSV Dataset",
                "Synthetic Benchmark Dataset",
                "Local Kaggle Dataset (creditcard.csv)",
            ],
            horizontal=True,
        )

        df_inspect = None

        if data_source == "Upload Custom CSV Dataset":
            uploaded_file = st.file_uploader(
                "Upload Transaction CSV Dataset",
                type=["csv"],
                help="Upload any credit card transaction CSV to validate schema, types, nulls, duplicates, and class distribution.",
            )
            if uploaded_file is not None:
                try:
                    df_inspect = pd.read_csv(uploaded_file)
                    st.success(f"File loaded successfully: `{uploaded_file.name}` ({len(df_inspect):,} rows, {len(df_inspect.columns)} columns)")
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")
            else:
                st.info("Upload your dataset CSV above to run automated schema checks, range validations, duplicate detection, and class balance analysis.")

        elif data_source == "Synthetic Benchmark Dataset":
            target_file = config.synthetic_path
            if not target_file.exists():
                st.info(f"Generating synthetic benchmark dataset at `{target_file}`...")
                from src.data.synthetic import create_test_fixtures
                create_test_fixtures()
            df_inspect = pd.read_csv(target_file)
            st.caption(f"Loaded synthetic benchmark: `{target_file}` ({len(df_inspect):,} rows)")

        elif data_source == "Local Kaggle Dataset (creditcard.csv)":
            target_file = config.raw_data_path
            if not target_file.exists():
                st.warning(
                    f"Local Kaggle dataset not found at `{target_file}`.\n\n"
                    "**How to supply the Kaggle dataset:**\n"
                    "1. Download `creditcard.csv` (~143 MB uncompressed) from Kaggle.\n"
                    f"2. Save it to `data/raw/creditcard.csv` in your project folder.\n"
                    "3. Alternatively, select **Upload Custom CSV Dataset** above to upload it directly from any folder on your machine without moving files!"
                )
            else:
                df_inspect = pd.read_csv(target_file)
                st.caption(f"Loaded local Kaggle dataset: `{target_file}` ({len(df_inspect):,} rows)")

        if df_inspect is not None:
            validator = DataValidator()
            has_class = "Class" in df_inspect.columns
            res = validator.validate(df_inspect, is_training=has_class)

            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Validation Status", "PASSED" if res.is_valid else "FAILED")
            q2.metric("Total Records", f"{res.row_count:,}")
            q3.metric("Feature Columns", f"{res.column_count}")
            q4.metric("Duplicates Flagged", f"{res.summary_stats.get('duplicate_count', 0):,}")

            if res.errors:
                st.error("Validation Errors Detected:")
                for e in res.errors:
                    st.write(f"- [{e.category}] {e.message}")

            if res.warnings:
                with st.expander("Validation Warnings", expanded=True):
                    for w in res.warnings:
                        st.write(f"- [{w.category}] {w.message}")

            st.subheader("Data Summary & Range Check")
            st.caption("First 10 records preview:")
            st.dataframe(df_inspect.head(10), use_container_width=True)

            sample_cols = [c for c in ["Time", "Amount", "V1", "V2", "V3", "V14", "Class"] if c in df_inspect.columns]
            if sample_cols:
                st.caption("Statistical Distribution of Key Features:")
                st.dataframe(df_inspect[sample_cols].describe(), use_container_width=True)

            if has_class:
                st.markdown("#### Class Imbalance Distribution")
                cd = res.summary_stats.get("class_distribution", {})
                c_norm, c_fraud = cd.get("normal_count", 0), cd.get("fraud_count", 0)
                st.write(f"- **Normal Transactions (Class=0):** `{c_norm:,}`")
                st.write(f"- **Fraudulent Transactions (Class=1):** `{c_fraud:,}` ({cd.get('fraud_rate', 0.0)*100:.3f}%)")
            else:
                st.info("Notice: No `Class` column detected in this dataset. Validated as inference/test transactions stream.")

    # =========================================================================
    # 5. DRIFT MONITORING
    # =========================================================================
    elif nav_option == "Drift Monitoring":
        st.title("Feature & Prediction Drift Monitoring")
        st.markdown(
            "Monitors statistical shift between incoming transaction batches and baseline features "
            "using **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS)** tests."
        )

        try:
            monitor = DriftMonitor()
            ref_data = monitor.reference_data
            st.success(f"Reference baseline loaded: `{len(ref_data):,}` training samples.")
        except Exception as e:
            st.error(f"Reference baseline unavailable: {e}")
            return

        st.subheader("Simulate Controlled Data Drift")
        sim_choice = st.selectbox(
            "Inject Synthetic Drift Scenario:",
            [
                "Baseline Stream (No Drift)",
                "Amount Spike (Inflation / Fraud Influx)",
                "PCA Component Shift (V14 Shift)",
                "Mixed Anomaly Shift",
            ],
        )

        if not config.synthetic_path.exists():
            from src.data.synthetic import create_test_fixtures
            create_test_fixtures()

        test_df = pd.read_csv(config.synthetic_path)
        if sim_choice == "Amount Spike (Inflation / Fraud Influx)":
            test_df = simulate_synthetic_drift(test_df, drift_type="amount_spike", severity=4.0)
        elif sim_choice == "PCA Component Shift (V14 Shift)":
            test_df = simulate_synthetic_drift(test_df, drift_type="feature_shift", severity=3.0)
        elif sim_choice == "Mixed Anomaly Shift":
            test_df = simulate_synthetic_drift(test_df, drift_type="mixed", severity=3.5)

        # Run drift calculation
        preds = predictor.predict_batch(test_df.drop(columns=["Class"], errors="ignore"))
        scores = np.array([p.risk_score for p in preds])
        report = monitor.evaluate_drift(test_df, risk_scores=scores)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Overall Drift Status", report.overall_drift_status)
        d2.metric("Drifted Features", f"{report.drifted_features_count} / {report.total_monitored_features}")
        d3.metric("Current Batch Size", f"{report.current_sample_count:,}")
        d4.metric("Mean Risk Score", f"{report.prediction_drift.get('mean_score', 0.0):.4f}")

        st.subheader("Feature-Level Stability Metrics")
        drift_rows = []
        for feat in report.feature_reports:
            drift_rows.append({
                "Feature": feat.feature_name,
                "PSI": feat.psi,
                "KS Statistic": feat.ks_statistic,
                "KS p-value": feat.ks_p_value,
                "Drift Status": feat.drift_status,
                "Ref Mean": feat.ref_mean,
                "Curr Mean": feat.curr_mean,
            })
        st.dataframe(pd.DataFrame(drift_rows), use_container_width=True)

        st.subheader("Prediction Distribution Shift")
        fig_dist, ax_dist = plt.subplots(figsize=(7, 2.5))
        sns.histplot(scores, bins=30, kde=True, color="#d9534f", ax=ax_dist)
        ax_dist.set_title("Current Batch Fraud Risk Score Distribution")
        ax_dist.set_xlabel("Predicted Fraud Probability")
        ax_dist.axvline(bundle.optimal_threshold, color="black", linestyle="--", label="Decision Threshold")
        ax_dist.legend()
        st.pyplot(fig_dist)
        plt.close(fig_dist)

        st.info(f"**Methodological Note:** {report.conceptual_notes}")

    # =========================================================================
    # 6. MODEL REGISTRY & GATING
    # =========================================================================
    elif nav_option == "Model Registry & Gating":
        st.title("Model Registry, Quality Gates & Rollback")
        st.write("Manage candidate models, verify deployment quality gates, and perform zero-downtime rollbacks.")

        registry = ModelRegistry()
        champ_meta = registry.get_champion_metadata()
        candidates = registry.list_candidates()

        st.subheader("Active Champion Model")
        if champ_meta:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Active Version", champ_meta.get("version"))
            c1.caption(f"Algorithm: `{champ_meta.get('algorithm')}`")
            test_m = champ_meta.get("test_metrics", {})
            c2.metric("Test PR-AUC", f"{test_m.get('pr_auc', 0.0):.4f}")
            c3.metric("Test Recall", f"{test_m.get('recall', 0.0):.4f}")
            c4.metric("Test Precision", f"{test_m.get('precision', 0.0):.4f}")
        else:
            st.warning("No champion metadata found.")

        st.divider()
        st.subheader("Automated Quality Gates")
        qg = config.quality_gates
        st.markdown(
            f"- **Minimum PR-AUC:** `>= {qg.min_pr_auc}`\n"
            f"- **Minimum Recall:** `>= {qg.min_recall}`\n"
            f"- **Minimum Precision:** `>= {qg.min_precision}`\n"
            f"- **Maximum False Positive Rate:** `<= {qg.max_fpr}`\n"
            "- **Benchmark Gate:** Must not degrade PR-AUC compared to current Champion."
        )

        st.subheader("Available Candidate Models")
        if candidates:
            cand_list = []
            for c in candidates:
                cand_list.append({
                    "Version": c.get("version"),
                    "Algorithm": c.get("algorithm"),
                    "Timestamp": c.get("timestamp"),
                    "Test PR-AUC": c.get("test_metrics", {}).get("pr_auc"),
                    "Test F1": c.get("test_metrics", {}).get("f1_score"),
                })
            st.dataframe(pd.DataFrame(cand_list), use_container_width=True)

            selected_cand = st.selectbox("Select Candidate to Evaluate / Promote:", [c["version"] for c in candidates])
            if st.button("Evaluate Promotion Gates"):
                cand_obj = next(c for c in candidates if c["version"] == selected_cand)
                passed, qg_eval = registry.evaluate_quality_gates(
                    cand_obj.get("test_metrics", {}), champion_metrics=champ_meta
                )
                if passed:
                    st.success(f"Candidate `{selected_cand}` PASSED all quality gates!")
                    if st.button(f"Confirm Promotion of {selected_cand} to Champion"):
                        registry.promote_candidate(selected_cand, reason="Dashboard manual promotion")
                        st.success("Candidate promoted! Please reload the page.")
                else:
                    st.error(f"Candidate `{selected_cand}` FAILED one or more quality gates:")
                    st.json(qg_eval["gates"])
        else:
            st.info("No candidate versions currently stored. Retrain models via `scripts/retrain.py`.")

        st.divider()
        st.subheader("Disaster Recovery: Champion Rollback")
        st.write("Restore the immediately preceding champion model from backup if deployment anomalies arise.")
        if st.button("Rollback to Previous Champion"):
            try:
                registry.rollback_to_previous_champion()
                st.success("Rollback executed successfully! Reloading active champion...")
                st.cache_resource.clear()
            except Exception as e:
                st.error(f"Rollback failed: {e}")

    # =========================================================================
    # 7. SYSTEM ARCHITECTURE
    # =========================================================================
    elif nav_option == "System Architecture":
        st.title("System Architecture & Engineering Specifications")

        st.markdown("### End-to-End Pipeline Workflow")
        st.markdown("""
```mermaid
flowchart TD
    A[Raw Kaggle / Synthetic Data] --> B[DataValidator Schema Check]
    B -->|Passed| C[Leakage-Safe Stratified Split]
    C --> D[Fit Preprocessor on Train Only]
    D --> E[Train Models: Baseline, LogReg, RF]
    E --> F[Tune Decision Threshold on Validation Set]
    F --> G[Evaluate Champion on Held-Out Test Set]
    G --> H[Quality Gates & Model Registry]
    H -->|Passed| I[Promote Champion Model]
    I --> J[FastAPI REST Service]
    I --> K[Streamlit Interactive Dashboard]
    J --> L[SQLite Telemetry & Latency Logging]
    K --> M[Drift Monitoring: PSI & KS-Test]
```
        """)

        st.subheader("Design Decisions & Constraints Compliance")
        st.markdown(
            """
| Dimension | Specification Requirement | FraudGuard AI Implementation |
|---|---|---|
| **Zero Spending** | No paid APIs, databases, or accounts | 100% open-source local stack (scikit-learn, SQLite, FastAPI, Streamlit) |
| **No Virtualization** | No Docker, WSL2, or Hyper-V | Windows-native execution via standard Python environment |
| **Leakage Safeguards** | Preprocessing fitted only on Train | `FraudFeaturePreprocessor` fit on train; applied identically at inference |
| **Imbalanced Classification** | Accuracy not primary metric | PR-AUC and F1-score optimize minority fraud class detection |
| **Threshold Calibration** | No tuning on test partition | Decision threshold tuned strictly on Validation set |
| **Artifact Security** | Reject arbitrary pickles | Validated dictionary schema containing versioned metadata and scaler |
| **Drift Detection** | Scientific shift metrics | Population Stability Index (PSI) & Kolmogorov-Smirnov 2-sample tests |
| **Model Registry** | Versioning, Gates, Rollback | Local file-based registry with automated backups and rollback |
            """
        )

        st.subheader("Environment Details")
        st.json({
            "python_version": sys.version,
            "platform": sys.platform,
            "project_root": str(PROJECT_ROOT),
        })


if __name__ == "__main__":
    main()
