# Zaalima Project 3: Retail Demand Forecasting & Inventory Optimization

Production-level time-series forecasting and inventory management dashboard powered by dbt, LightGBM/Prophet, and Streamlit.
# 📦 Retail Demand Forecasting & Inventory Optimization System

An end-to-end, production-grade demand forecasting and inventory optimization engine built on the M5 dataset. The system features automated data validation, feature engineering (lag/rolling/EMA), hyperparameter-tuned LightGBM modeling, safety stock/reorder point policies, price elasticity analysis, Streamlit visualization, and automated CI/CD with drift detection.

---

## 📐 System Architecture
---

## 🛠️ Modular Pipeline Architecture

* **Data Quality & Ingestion**: Validates schema integrity, null ratios, and data types (`scripts/validate_data_quality.py`).
* **Feature Engineering**: Generates multi-window lag, rolling mean/std, and exponential moving average (EMA) features (`scripts/feature_engineering.py`).
* **Predictive Modeling**: LightGBM model trained using Optuna automated hyperparameter optimization (`scripts/tune_lightgbm.py`).
* **Inventory Policy**: Computes dynamic **Safety Stock (SS)** and **Reorder Points (ROP)** assuming a 95% service level (`scripts/inventory_optimization.py`).
* **Pricing Dynamics**: Estimates log-log price elasticity to evaluate demand sensitivity (`scripts/price_elasticity.py`).
* **Monitoring & Retraining**: Evaluates Population Stability Index (PSI) and WAPE drift to trigger retraining alerts (`scripts/drift_detection.py`).
* **Visualization**: Interactive Streamlit dashboard for real-time KPI inspection (`app/dashboard.py`).

---

## 🚀 Quickstart & Setup

### 1. Environment Setup
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt