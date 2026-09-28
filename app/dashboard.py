import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Retail Demand & Inventory Dashboard", layout="wide")

DATA_DIR = Path("data")

st.title("📦 Retail Demand Forecasting & Inventory Optimization")
st.markdown("Interactive Operational Dashboard — M5 Demand Insights & Reorder Policies")

# Load Datasets
metrics_file = DATA_DIR / "evaluation_metrics.csv"
inventory_file = DATA_DIR / "inventory_plan.csv"
elasticity_file = DATA_DIR / "price_elasticity_results.csv"

# Row 1: KPI Summary Cards
st.subheader("1. Key Performance Indicators")
col1, col2, col3 = st.columns(3)

if metrics_file.exists():
    metrics_df = pd.read_csv(metrics_file)
    col1.metric("Validation RMSE", f"{metrics_df['RMSE'].iloc[0]:.4f}")
    col2.metric("Validation MAE", f"{metrics_df['MAE'].iloc[0]:.4f}")
    col3.metric("Validation WAPE", f"{metrics_df['WAPE_%'].iloc[0]:.2f}%")
else:
    st.info("Evaluation metrics file not found. Run pipeline first.")

st.markdown("---")

# Row 2: Inventory Optimization & Reorder Alerts
st.subheader("2. Inventory Policy & Safety Stock Overview")

if inventory_file.exists():
    inv_df = pd.read_csv(inventory_file)
    
    # Line chart comparing actual sales, predicted demand, and reorder point
    st.line_chart(inv_df.set_index("day_num")[["sales", "predicted_demand", "reorder_point"]])

    st.subheader("Detailed Reorder Schedule")
    st.dataframe(inv_df, use_container_width=True)
else:
    st.info("Inventory plan file not found. Run pipeline first.")

st.markdown("---")

# Row 3: Price Elasticity Summary
st.subheader("3. Price Elasticity of Demand")

if elasticity_file.exists():
    elast_df = pd.read_csv(elasticity_file)
    coeff = elast_df["Elasticity_Coefficient"].iloc[0]
    cls = elast_df["Classification"].iloc[0]

    st.write(f"**Elasticity Coefficient:** `{coeff:.4f}`")
    st.write(f"**Demand Sensitivity:** `{cls}`")
else:
    st.info("Price elasticity results file not found. Run pipeline first.")