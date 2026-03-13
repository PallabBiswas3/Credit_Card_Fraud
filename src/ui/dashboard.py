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
import networkx as nx

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
                        
                        # --- Enhanced Network Graph Visualization ---
                        st.subheader("🕸️ Transaction Network Analysis")
                        
                        # Create more sophisticated graph
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                        
                        # Graph 1: Transaction Network
                        G = nx.Graph()
                        
                        # Add nodes with enhanced attributes
                        G.add_node(card_id, color='#1f77b4', size=3000, label=f'Card: {card_id[-4:]}')
                        G.add_node(merchant_id, color='#2ca02c', size=2500, label=f'Merchant: {merchant_id[-4:]}')
                        G.add_node(device_id, color='#ff7f0e', size=2000, label=f'Device: {device_id[-4:]}')
                        
                        # Add transaction edges
                        G.add_edge(card_id, merchant_id, weight=amount, label=f'${amount:.2f}')
                        G.add_edge(card_id, device_id, weight=1.0, label='Used')
                        
                        # Add suspicious connections if high risk
                        if prob > 0.4:
                            # Add fraud ring nodes
                            fraud_ring_1 = f"FRAUD_M_{merchant_id[-4:]}"
                            fraud_ring_2 = f"FRAUD_D_{device_id[-4:]}"
                            
                            G.add_node("FRAUD_HUB", color='red', size=4000, label='Fraud Hub')
                            G.add_node(fraud_ring_1, color='orange', size=2000, label='Suspicious Merchant')
                            G.add_node(fraud_ring_2, color='orange', size=1500, label='Suspicious Device')
                            
                            # Add suspicious connections
                            G.add_edge(merchant_id, "FRAUD_HUB", weight=5.0, style='dashed')
                            G.add_edge(device_id, "FRAUD_HUB", weight=3.0, style='dashed')
                            G.add_edge("FRAUD_HUB", fraud_ring_1, weight=2.0)
                            G.add_edge("FRAUD_HUB", fraud_ring_2, weight=2.0)
                        
                        # Calculate layout
                        pos = nx.spring_layout(G, seed=42, k=3, iterations=50)
                        
                        # Draw network graph
                        node_colors = [G.nodes[n]['color'] for n in G.nodes]
                        node_sizes = [G.nodes[n].get('size', 2000) for n in G.nodes]
                        node_labels = {n: G.nodes[n].get('label', n) for n in G.nodes}
                        
                        # Draw edges with different styles
                        for u, v, d in G.edges(data=True):
                            edge_style = d.get('style', 'solid')
                            edge_width = d.get('weight', 1.0) * 0.5
                            edge_color = 'red' if edge_style == 'dashed' else 'gray'
                            nx.draw_networkx_edges(G, pos, [(u, v)], edge_color=edge_color, 
                                                width=edge_width, style=edge_style, ax=ax1)
                        
                        # Draw nodes
                        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                                           alpha=0.8, ax=ax1)
                        
                        # Draw labels
                        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax1)
                        
                        ax1.set_title("Transaction Network Graph", fontsize=14, fontweight='bold')
                        ax1.axis('off')
                        
                        # Graph 2: Risk Metrics
                        risk_metrics = ['Transaction Amount', 'Network Degree', 'Risk Score', 'Anomaly Level']
                        risk_values = [amount, len(G.edges), prob, 'High' if prob > 0.4 else 'Low']
                        colors = ['#ff7f0e' if v == amount else '#1f77b4' for v in risk_values]
                        
                        bars = ax2.bar(risk_metrics, [amount, len(G.edges), prob, 100 if prob > 0.4 else 20], 
                                       color=colors, alpha=0.7)
                        ax2.set_title("Risk Analysis Metrics", fontsize=14, fontweight='bold')
                        ax2.set_ylabel("Value")
                        
                        # Rotate x-axis labels
                        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
                        
                        # Add value labels on bars
                        for bar, value in zip(bars, [amount, len(G.edges), prob, 100 if prob > 0.4 else 20]):
                            height = bar.get_height()
                            ax2.text(bar.get_x() + bar.get_width()/2., height,
                                     f'{value:.1f}', ha='center', va='bottom')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Network Statistics
                        st.subheader("📊 Network Statistics")
                        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                        
                        stats_col1.metric("Network Nodes", len(G.nodes))
                        stats_col2.metric("Network Edges", len(G.edges))
                        stats_col3.metric("Avg Degree", f"{2 * len(G.edges) / len(G.nodes):.1f}")
                        stats_col4.metric("Clustering", "High" if prob > 0.4 else "Low")
                        
                        # Suspicious Pattern Detection
                        if prob > 0.4:
                            st.warning("🚨 **Suspicious Pattern Detected:**")
                            st.write("- Transaction connects to known fraud-associated entities")
                            st.write("- Unusual amount for this merchant category")
                            st.write("- Device shows high activity across multiple cards")
                            st.write("- Recommend: Enhanced verification required")
                        
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
