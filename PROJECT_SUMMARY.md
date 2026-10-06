# 🚀 Executive Project Summary: Retail Demand Forecasting & Inventory Optimization

## 📌 Project Overview
An end-to-end, production-grade demand forecasting and inventory optimization system built over a 20-day engineering cycle. The platform transforms raw M5 retail time-series data into actionable supply chain reorder policies and price elasticity insights.

---

## 🏗️ Architecture & Modules
1. **Data Quality & Validation**: Automated schema and null checks (`scripts/validate_data_quality.py`).
2. **Feature Pipeline**: Multi-window lag (7/28 days), 7-day rolling statistics, and Exponential Moving Averages (`scripts/feature_engineering.py`).
3. **Predictive Modeling**: LightGBM model tuned via Optuna hyperparameter optimization (`scripts/tune_lightgbm.py`).
4. **Inventory Policy**: Dynamic Safety Stock and Reorder Point (ROP) calculation at a 95% service level (`scripts/inventory_optimization.py`).
5. **Pricing Dynamics**: Log-log demand price elasticity analysis (`scripts/price_elasticity.py`).
6. **MLOps & Quality Assurance**:
   * Automated CI/CD workflow via GitHub Actions (`.github/workflows/ci-pipeline.yml`).
   * Population Stability Index (PSI) drift monitoring & retraining alerts (`scripts/drift_detection.py`).
   * Automated integration & unit testing suite (`scripts/test_pipeline.py`).
7. **Operational Dashboard**: Interactive Streamlit UI (`app/dashboard.py`).

---

## 📊 Key Performance Indicators
* **Validation WAPE**: ~22% (outperforming naive baseline WAPE of ~34.5%)
* **Target Service Level**: 95%
* **Production Status**: Production Ready (Release Tag: `v1.0.0`)