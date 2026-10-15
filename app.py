"""
DL-IDS: AI-Based Intrusion Detection System - Streamlit Dashboard

A professional dashboard for testing the trained DL-IDS model,
visualizing performance metrics, and making predictions.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.utils import load_config, get_project_root
from src.inference import IDSPredictor

# ========== Page Configuration ==========
st.set_page_config(
    page_title="DL-IDS | Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== Custom CSS ==========
st.markdown("""
<style>
    /* Main theme */
    .main { background-color: #0e1117; }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid rgba(83, 127, 231, 0.3);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 { color: #e94560; margin-bottom: 0.5rem; }
    .main-header p { color: #a8b2d1; }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid rgba(83, 127, 231, 0.2);
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value { font-size: 2rem; font-weight: bold; color: #e94560; }
    .metric-label { color: #8892b0; font-size: 0.85rem; }
    
    /* Alert boxes */
    .alert-normal {
        background: linear-gradient(135deg, #0a3d0a, #1a5a1a);
        border: 1px solid #4caf50;
        padding: 1.5rem; border-radius: 10px;
        color: #a5d6a7; text-align: center;
    }
    .alert-attack {
        background: linear-gradient(135deg, #3d0a0a, #5a1a1a);
        border: 1px solid #f44336;
        padding: 1.5rem; border-radius: 10px;
        color: #ef9a9a; text-align: center;
    }
    
    /* Status indicator */
    .status-dot { display: inline-block; width: 10px; height: 10px;
        border-radius: 50%; margin-right: 6px; }
    .status-online { background: #4caf50; box-shadow: 0 0 6px #4caf50; }
    .status-offline { background: #f44336; }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0e1117, #1a1a2e); }
    
    /* Table styling */
    .dataframe { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ========== Session State ==========
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "predictor" not in st.session_state:
    st.session_state.predictor = None
    st.session_state.model_loaded = False


# ========== Helper Functions ==========
@st.cache_resource
def load_predictor():
    """Load the trained model (cached)."""
    try:
        config = load_config(os.path.join(project_root, "config.yaml"))
        predictor = IDSPredictor(config, model_name="DL-IDS")
        predictor.load()
        return predictor, True
    except Exception as e:
        return None, str(e)


@st.cache_data
def load_metrics():
    """Load all evaluation metrics."""
    metrics = {}
    reports_dir = os.path.join(project_root, "reports")
    
    for filename in ["dl_ids_metrics.json", "rf_metrics.json", "cnn_metrics.json"]:
        filepath = os.path.join(reports_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                key = filename.replace("_metrics.json", "").upper().replace("DL_IDS", "DL-IDS").replace("RF", "Random Forest").replace("CNN", "CNN Only")
                metrics[key] = json.load(f)
    
    return metrics


@st.cache_data
def load_training_history():
    """Load training history."""
    filepath = os.path.join(project_root, "reports", "dl_ids_history.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_dataset_report():
    """Load dataset inspection report."""
    filepath = os.path.join(project_root, "reports", "dataset_inspection.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_comparison():
    """Load model comparison."""
    filepath = os.path.join(project_root, "reports", "model_comparison.json")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return None


def add_to_history(prediction_result, input_type="single"):
    """Add a prediction to the session history."""
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "input_type": input_type,
        "predicted_class": prediction_result.get("predicted_class", "N/A"),
        "confidence": prediction_result.get("confidence", 0),
        "is_attack": prediction_result.get("is_attack", False),
    }
    st.session_state.prediction_history.insert(0, entry)
    # Keep last 50 predictions
    st.session_state.prediction_history = st.session_state.prediction_history[:50]


# ========== Sidebar ==========
with st.sidebar:
    st.markdown("# 🛡️ DL-IDS")
    st.markdown("**AI-Based Intrusion Detection**")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔍 Single Prediction", "📊 Batch Prediction",
         "📈 Dataset Explorer", "🏆 Model Evaluation", "📋 Prediction History"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Model status
    predictor, load_status = load_predictor()
    if load_status is True:
        st.markdown('<span class="status-dot status-online"></span> Model Loaded', unsafe_allow_html=True)
        st.session_state.predictor = predictor
        st.session_state.model_loaded = True
    else:
        st.markdown('<span class="status-dot status-offline"></span> Model Not Loaded', unsafe_allow_html=True)
        st.error(f"Error: {load_status}")
    
    st.markdown("---")
    st.markdown("*Based on DL-IDS paper*")
    st.markdown("*Sun et al., 2020*")


# ========== Pages ==========

if page == "🏠 Overview":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ DL-IDS: AI-Based Intrusion Detection System</h1>
        <p>Deep Learning-based Network Intrusion Detection using CNN-LSTM Hybrid Architecture</p>
        <p><em>Based on: "DL-IDS: Extracting Features Using CNN-LSTM Hybrid Network for Intrusion Detection System" — Sun et al., 2020</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    metrics = load_metrics()
    ds_report = load_dataset_report()
    
    col1, col2, col3, col4 = st.columns(4)
    
    dl_ids_m = metrics.get("DL-IDS", {})
    
    with col1:
        st.metric("🎯 Accuracy", f"{dl_ids_m.get('accuracy', 0):.2%}")
    with col2:
        st.metric("📊 F1 Score (weighted)", f"{dl_ids_m.get('f1_weighted', 0):.2%}")
    with col3:
        total_samples = ds_report.get("shape", {}).get("rows", 0) if ds_report else 0
        st.metric("📁 Total Records", f"{total_samples:,}")
    with col4:
        num_classes = ds_report.get("num_classes", 0) if ds_report else 0
        st.metric("🏷️ Classes", str(num_classes))
    
    st.markdown("---")
    
    # Project description
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📖 About This Project")
        st.markdown("""
        This system implements an **AI-based Intrusion Detection System (IDS)** 
        using a **CNN-LSTM hybrid deep learning architecture** to classify network 
        traffic as normal or belonging to various attack categories.
        
        **Key Features:**
        - 🧠 CNN extracts local feature patterns from network flow data
        - 🔄 LSTM captures sequential dependencies across feature groups
        - ⚖️ Class-weighted training handles the severe class imbalance
        - 📊 Trained on the CICIDS2017 benchmark dataset
        - 🎯 Multi-class classification (Normal + multiple attack types)
        """)
    
    with col2:
        st.markdown("### 🗃️ Dataset: CICIDS2017")
        st.markdown("""
        The **Canadian Institute for Cybersecurity IDS 2017** dataset covers:
        
        - **Normal traffic** and **14 attack types** across 5 days
        - Generated using CICFlowMeter with 78 network flow features
        - Attack types include: DoS, DDoS, Brute Force, Port Scan, 
          Botnet, Infiltration, Web Attacks, and Heartbleed
        - **~2.8 million** total flow records across 8 CSV files
        """)
    
    # Model comparison summary
    comparison = load_comparison()
    if comparison:
        st.markdown("### 🏆 Model Performance Comparison")
        comp_df = pd.DataFrame(comparison).T
        st.dataframe(comp_df, use_container_width=True)


elif page == "🔍 Single Prediction":
    st.markdown("## 🔍 Single Record Prediction")
    st.markdown("Enter network traffic features to classify as normal or attack.")
    
    if not st.session_state.model_loaded:
        st.error("⚠️ Model not loaded. Please check the artifacts directory.")
    else:
        predictor = st.session_state.predictor
        feature_names = predictor.get_feature_names()
        
        st.info(f"📋 This model uses **{len(feature_names)} features**. Enter values below or use defaults (0).")
        
        # Create input form with columns
        with st.form("single_prediction_form"):
            features = {}
            
            # Organize into columns
            cols = st.columns(3)
            for i, fname in enumerate(feature_names):
                with cols[i % 3]:
                    features[fname] = st.number_input(
                        fname, value=0.0, format="%.4f", key=f"feat_{i}"
                    )
            
            submitted = st.form_submit_button("🔍 Predict", use_container_width=True)
        
        if submitted:
            with st.spinner("Analyzing traffic..."):
                try:
                    result = predictor.predict_single(features)
                    
                    # Display result
                    st.markdown("### 📊 Prediction Result")
                    
                    if result["is_attack"]:
                        st.markdown(f"""
                        <div class="alert-attack">
                            <h2>⚠️ ATTACK DETECTED</h2>
                            <h3>{result['predicted_class']}</h3>
                            <p>Confidence: {result['confidence']:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="alert-normal">
                            <h2>✅ NORMAL TRAFFIC</h2>
                            <h3>{result['predicted_class']}</h3>
                            <p>Confidence: {result['confidence']:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Class probabilities
                    st.markdown("#### Class Probabilities")
                    probs_df = pd.DataFrame(
                        list(result["class_probabilities"].items()),
                        columns=["Class", "Probability"]
                    ).sort_values("Probability", ascending=False)
                    
                    # Bar chart
                    fig, ax = plt.subplots(figsize=(10, 4))
                    colors = ['#e94560' if c != 'Normal' else '#4caf50' for c in probs_df["Class"]]
                    ax.barh(probs_df["Class"], probs_df["Probability"], color=colors, alpha=0.8)
                    ax.set_xlabel("Probability")
                    ax.set_title("Class Probabilities")
                    ax.set_xlim(0, 1)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Add to history
                    add_to_history(result, "single")
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")


elif page == "📊 Batch Prediction":
    st.markdown("## 📊 Batch CSV Prediction")
    st.markdown("Upload a CSV file with network traffic records for batch classification.")
    
    if not st.session_state.model_loaded:
        st.error("⚠️ Model not loaded. Please check the artifacts directory.")
    else:
        predictor = st.session_state.predictor
        
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.info(f"📄 Loaded {len(df):,} records with {len(df.columns)} columns")
                
                # Show preview
                st.markdown("#### Input Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                if st.button("🔍 Run Batch Prediction", use_container_width=True):
                    with st.spinner(f"Classifying {len(df):,} records..."):
                        start_time = time.time()
                        result_df = predictor.predict_batch(df)
                        elapsed = time.time() - start_time
                    
                    st.success(f"✅ Classified {len(result_df):,} records in {elapsed:.2f}s")
                    
                    # Summary
                    col1, col2, col3 = st.columns(3)
                    attack_count = result_df["Is_Attack"].sum()
                    normal_count = len(result_df) - attack_count
                    
                    with col1:
                        st.metric("Total Records", f"{len(result_df):,}")
                    with col2:
                        st.metric("Normal", f"{normal_count:,}", delta_color="normal")
                    with col3:
                        st.metric("Attacks", f"{attack_count:,}", delta_color="inverse")
                    
                    # Class distribution
                    st.markdown("#### Prediction Distribution")
                    pred_counts = result_df["Predicted_Class"].value_counts()
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    colors = ['#4caf50' if c == 'Normal' else '#e94560' for c in pred_counts.index]
                    pred_counts.plot(kind='barh', ax=ax, color=colors, alpha=0.8)
                    ax.set_xlabel("Count")
                    ax.set_title("Predicted Class Distribution")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Show results
                    st.markdown("#### Prediction Results")
                    st.dataframe(result_df.head(50), use_container_width=True)
                    
                    # Download
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Predictions CSV",
                        csv,
                        file_name="ids_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                    
                    # Add summary to history
                    add_to_history({
                        "predicted_class": f"Batch ({len(result_df)} records)",
                        "confidence": float(result_df["Confidence"].mean()),
                        "is_attack": attack_count > 0,
                    }, "batch")
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")


elif page == "📈 Dataset Explorer":
    st.markdown("## 📈 Dataset Explorer")
    
    ds_report = load_dataset_report()
    
    if ds_report:
        # Dataset shape
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{ds_report['shape']['rows']:,}")
        with col2:
            st.metric("Features", f"{ds_report['shape']['columns'] - 1}")
        with col3:
            st.metric("Classes", str(ds_report.get('num_classes', 'N/A')))
        
        st.markdown("---")
        
        # Class distribution
        if "label_distribution" in ds_report:
            st.markdown("### 📊 Class Distribution")
            
            label_dist = ds_report["label_distribution"]
            dist_df = pd.DataFrame(
                [(k, v, f"{v/ds_report['shape']['rows']*100:.2f}%")
                 for k, v in sorted(label_dist.items(), key=lambda x: -x[1])],
                columns=["Class", "Count", "Percentage"]
            )
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.dataframe(dist_df, use_container_width=True)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = plt.cm.Set3(np.linspace(0, 1, len(label_dist)))
                counts = [label_dist[k] for k in sorted(label_dist.keys())]
                labels = sorted(label_dist.keys())
                bars = ax.barh(labels, counts, color=colors, alpha=0.85)
                ax.set_xlabel("Count (log scale)")
                ax.set_xscale('log')
                ax.set_title("Class Distribution")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        
        st.markdown("---")
        
        # Data quality
        st.markdown("### 🔧 Data Quality Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Missing Values", f"{ds_report.get('total_missing', 0):,}")
        with col2:
            st.metric("Duplicate Rows", f"{ds_report.get('duplicate_rows', 0):,}")
        with col3:
            st.metric("Constant Columns", str(len(ds_report.get('constant_columns', []))))
        
        if ds_report.get("constant_columns"):
            with st.expander("View Constant Columns"):
                st.write(ds_report["constant_columns"])
        
        if ds_report.get("infinite_values"):
            with st.expander("View Columns with Infinite Values"):
                st.json(ds_report["infinite_values"])
        
        # Feature list
        with st.expander("View All Features"):
            st.write(ds_report.get("columns", []))
    else:
        st.warning("Dataset inspection report not found. Run training first.")


elif page == "🏆 Model Evaluation":
    st.markdown("## 🏆 Model Evaluation Results")
    
    metrics = load_metrics()
    
    if not metrics:
        st.warning("No evaluation results found. Run training first.")
    else:
        # Tabs for different models
        model_tabs = st.tabs(list(metrics.keys()) + ["📊 Comparison"])
        
        for i, (model_name, m) in enumerate(metrics.items()):
            with model_tabs[i]:
                st.markdown(f"### {model_name}")
                
                # Overall metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Accuracy", f"{m.get('accuracy', 0):.4f}")
                with col2:
                    st.metric("Precision (macro)", f"{m.get('precision_macro', 0):.4f}")
                with col3:
                    st.metric("Recall (macro)", f"{m.get('recall_macro', 0):.4f}")
                with col4:
                    st.metric("F1 (macro)", f"{m.get('f1_macro', 0):.4f}")
                with col5:
                    st.metric("F1 (weighted)", f"{m.get('f1_weighted', 0):.4f}")
                
                st.markdown("---")
                
                # Per-class metrics
                if "per_class" in m:
                    st.markdown("#### Per-Class Metrics")
                    per_class_data = []
                    for cls_name, cls_m in m["per_class"].items():
                        per_class_data.append({
                            "Class": cls_name,
                            "Accuracy": f"{cls_m.get('accuracy', 0):.4f}",
                            "Precision": f"{cls_m.get('precision', 0):.4f}",
                            "Recall (TPR)": f"{cls_m.get('recall', 0):.4f}",
                            "F1-Score": f"{cls_m.get('f1_score', 0):.4f}",
                            "FPR": f"{cls_m.get('fpr', 0):.4f}",
                            "Support": cls_m.get("support", 0),
                        })
                    st.dataframe(pd.DataFrame(per_class_data), use_container_width=True)
                
                # Confusion matrix
                if "confusion_matrix" in m:
                    st.markdown("#### Confusion Matrix")
                    cm = np.array(m["confusion_matrix"])
                    
                    # Get class names from per_class keys
                    class_names = list(m.get("per_class", {}).keys())
                    if not class_names:
                        class_names = [str(i) for i in range(len(cm))]
                    
                    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # Raw counts
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                               xticklabels=class_names, yticklabels=class_names,
                               ax=axes[0], linewidths=0.5)
                    axes[0].set_title("Confusion Matrix (Counts)")
                    axes[0].set_xlabel("Predicted")
                    axes[0].set_ylabel("Actual")
                    axes[0].tick_params(axis='x', rotation=45)
                    
                    # Normalized
                    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    cm_norm = np.nan_to_num(cm_norm)
                    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                               xticklabels=class_names, yticklabels=class_names,
                               ax=axes[1], linewidths=0.5)
                    axes[1].set_title("Confusion Matrix (Normalized)")
                    axes[1].set_xlabel("Predicted")
                    axes[1].set_ylabel("Actual")
                    axes[1].tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
        
        # Comparison tab
        with model_tabs[-1]:
            st.markdown("### 📊 Model Comparison")
            
            comparison = load_comparison()
            if comparison:
                comp_df = pd.DataFrame(comparison).T
                st.dataframe(comp_df, use_container_width=True)
            
            # Comparison plot
            metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
            display_names = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "F1 (weighted)"]
            
            fig, ax = plt.subplots(figsize=(12, 5))
            x = np.arange(len(metric_names))
            width = 0.8 / len(metrics)
            colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
            
            for i, (model_name, m) in enumerate(metrics.items()):
                values = [m.get(mn, 0) for mn in metric_names]
                offset = (i - len(metrics) / 2 + 0.5) * width
                ax.bar(x + offset, values, width, label=model_name, color=colors[i % len(colors)], alpha=0.85)
            
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(display_names, rotation=15)
            ax.legend()
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Training curves
        history = load_training_history()
        if history:
            st.markdown("---")
            st.markdown("### 📈 DL-IDS Training Curves")
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            epochs = range(1, len(history["train_loss"]) + 1)
            
            axes[0].plot(epochs, history["train_loss"], 'b-o', label='Train', ms=3)
            axes[0].plot(epochs, history["val_loss"], 'r-o', label='Validation', ms=3)
            axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(epochs, history["train_acc"], 'b-o', label='Train', ms=3)
            axes[1].plot(epochs, history["val_acc"], 'r-o', label='Validation', ms=3)
            axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
            
            if "val_f1" in history:
                axes[2].plot(epochs, history["val_f1"], 'g-o', label='Val F1', ms=3)
                axes[2].set_title('F1 Score (Weighted)'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
            
            plt.suptitle("DL-IDS Training History", fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


elif page == "📋 Prediction History":
    st.markdown("## 📋 Prediction History")
    st.markdown("Predictions made during this session.")
    
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", len(history_df))
        with col2:
            attacks = history_df["is_attack"].sum()
            st.metric("Attacks Detected", int(attacks))
        with col3:
            avg_conf = history_df["confidence"].mean()
            st.metric("Avg Confidence", f"{avg_conf:.2%}")
        
        st.markdown("---")
        
        # Format for display
        display_df = history_df.copy()
        display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.2%}")
        display_df["is_attack"] = display_df["is_attack"].apply(lambda x: "🚨 Yes" if x else "✅ No")
        display_df.columns = ["Time", "Type", "Predicted Class", "Confidence", "Attack?"]
        
        st.dataframe(display_df, use_container_width=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()
    else:
        st.info("No predictions made yet. Use the Single or Batch prediction pages to get started.")
