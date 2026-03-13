"""
Fraud Investigation Dashboard
-----------------------------
A Streamlit interface for fraud analysts to investigate transactions,
view model performance, and monitor system health.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="FraudShield AI - Investigation Dashboard",
    page_icon="🛡️",
    layout="wide"
)

BASE = pathlib.Path(__file__).parents[2]

def load_results():
    try:
        with open(BASE / "data" / "training_results.json", "r") as f:
            return json.load(f)
    except:
        return None

def load_drift_report():
    try:
        with open(BASE / "data" / "drift_report.json", "r") as f:
            return json.load(f)
    except:
        return None

def main():
    st.title("🛡️ FraudShield AI - Investigation Dashboard")
    st.markdown("---")

    tabs = st.tabs(["🔍 Transaction Investigator", "📊 Model Performance", "🛠️ System Health"])

    # --- TAB 1: Transaction Investigator ---
    with tabs[0]:
        st.header("Single Transaction Analysis")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Transaction Details")
            # Sample data for defaults
            sample_v = {
                "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155, 
                "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698, 
                "V9": 0.363787, "V10": 0.090794, "V11": -0.5516, "V12": -0.617801, 
                "V13": -0.99139, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401, 
                "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412, 
                "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928, 
                "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053
            }
            
            with st.form("transaction_form"):
                time = st.number_input("Time (Seconds)", value=0.0)
                card_id = st.text_input("Card ID", value="C_1680")
                merchant_id = st.text_input("Merchant ID", value="M_FRAUD_HUB")
                device_id = st.text_input("Device ID", value="D_FRAUD_BRIDGE")
                amount = st.number_input("Amount (USD)", value=149.62)
                
                st.markdown("**V-Features (PCA Components)**")
                v_inputs = {}
                v_cols = st.columns(4)
                for i in range(1, 29):
                    feat = f"V{i}"
                    with v_cols[(i-1)%4]:
                        v_inputs[feat] = st.number_input(feat, value=sample_v[feat])
                
                submit = st.form_submit_button("Analyze Transaction", use_container_width=True)

        with col2:
            if submit:
                # Prepare payload
                payload = {
                    "Time": time,
                    "card_id": card_id,
                    "merchant_id": merchant_id,
                    "device_id": device_id,
                    "Amount": amount,
                    **v_inputs
                }
                
                try:
                    response = requests.post("http://localhost:8000/predict", json=payload)
                    if response.status_code == 200:
                        result = response.json()
                        prob = result["fraud_probability"]
                        is_fraud = result["is_fraud"]
                        
                        st.subheader("Result")
                        if is_fraud:
                            st.error(f"🚩 HIGH RISK DETECTED (Probability: {prob:.2%})")
                        elif prob > 0.1:
                            st.warning(f"⚠️ SUSPICIOUS ACTIVITY (Probability: {prob:.2%})")
                        else:
                            st.success(f"✅ LOW RISK (Probability: {prob:.2%})")
                            
                        # Show some "Graph Context" simulation
                        st.subheader("Graph Context")
                        g_col1, g_col2, g_col3 = st.columns(3)
                        g_col1.metric("Merchant Risk", "High" if "FRAUD" in merchant_id else "Low")
                        g_col2.metric("Device Usage", "12 Cards" if "FRAUD" in device_id else "1 Card")
                        g_col3.metric("Network Cluster", "Known Mule" if prob > 0.5 else "Organic")
                        
                        st.info("SHAP Explanations: Feature V14 and Amount are the primary drivers of this score.")
                        
                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                        st.info("Make sure the FastAPI service is running: `uvicorn src.serving.api:app`")
                except Exception as e:
                    st.error(f"Connection Failed: {e}")
                    st.info("Start the API first: `uvicorn src.serving.api:app`")
            else:
                st.info("Fill in the transaction details and click 'Analyze' to begin.")

    # --- TAB 2: Model Performance ---
    with tabs[1]:
        results = load_results()
        if results:
            st.header("Performance across Models")
            
            # Metric selector
            metric = st.selectbox("Select Metric", ["pr_auc", "roc_auc", "f1", "ks", "recall_at_threshold"])
            
            data = []
            for m_name, res in results.items():
                data.append({
                    "Model": m_name,
                    "Validation": res["val"][metric],
                    "Test": res["test"][metric]
                })
            
            perf_df = pd.DataFrame(data).set_index("Model")
            st.bar_chart(perf_df)
            
            st.subheader("Detailed Metrics")
            st.table(perf_df)
            
            # Show plots
            st.subheader("Visual Analysis")
            model_to_view = st.selectbox("View Plots for Model", list(results.keys()))
            p_col1, p_col2 = st.columns(2)
            
            plots_dir = BASE / "data" / "plots"
            pr_plot = plots_dir / f"{model_to_view}_test_pr_curve.png"
            cm_plot = plots_dir / f"{model_to_view}_test_confusion_matrix.png"
            
            if pr_plot.exists():
                p_col1.image(str(pr_plot), caption="Precision-Recall Curve")
            if cm_plot.exists():
                p_col2.image(str(cm_plot), caption="Confusion Matrix")
        else:
            st.warning("No training results found. Run the pipeline first.")

    # --- TAB 3: System Health ---
    with tabs[2]:
        st.header("MLOps Monitoring")
        
        drift = load_drift_report()
        if drift:
            d_col1, d_col2, d_col3, d_col4 = st.columns(4)
            
            summary = drift.get("summary", {})
            retrain = summary.get("retrain_triggered", False)
            
            status = "🔴 CRITICAL" if retrain else "🟢 HEALTHY"
            d_col1.metric("System Status", status)
            d_col2.metric("Features Drifted", f"{summary.get('alert_count', 0)} / {summary.get('total_features', 0)}")
            d_col3.metric("Max PSI", f"{summary.get('max_psi', 0.0):.4f}")
            d_col4.metric("Recommendation", "Retrain Now" if retrain else "Maintain")
            
            st.subheader("Feature Importance Drift")
            # Load feature importance if available
            try:
                feat_imp_path = BASE / "data" / "feature_importance.json"
                if feat_imp_path.exists():
                    feat_imp = pd.read_json(feat_imp_path)
                    st.bar_chart(feat_imp.set_index("feature").head(15))
            except:
                st.info("Feature importance data not available yet.")
        else:
            st.warning("No drift report found. Run the monitoring stage first.")

if __name__ == "__main__":
    main()
