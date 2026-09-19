import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge

DATA_DIR = Path("data")

def run_baseline_model():
    input_path = DATA_DIR / "final_feature_matrix.csv"
    
    if not input_path.exists():
        print("Feature matrix missing. Please run Day 5 feature script first.")
        return

    print("\n--- DAY 6: BASELINE MODELING & EVALUATION ---")
    df = pd.read_csv(input_path)

    # Clean missing values created by lagging/rolling windows
    df_clean = df.dropna(subset=["lag_7", "lag_28", "rolling_mean_7", "ema_7"]).copy()

    # Time-series Train/Validation Split (Last 28 days for validation)
    max_day = df_clean["day_num"].max()
    split_day = max_day - 28

    train = df_clean[df_clean["day_num"] <= split_day]
    val = df_clean[df_clean["day_num"] > split_day]

    features = ["lag_7", "lag_28", "rolling_mean_7", "rolling_std_7", "ema_7", "ema_28", "is_event", "discount_ratio"]
    target = "sales"

    X_train, y_train = train[features], train[target]
    X_val, y_val = val[features], val[target]

    # 1. Naïve Baseline (Predict using last known 7-day average)
    val_naive_pred = val["rolling_mean_7"]
    naive_rmse = np.sqrt(mean_squared_error(y_val, val_naive_pred))
    naive_mae = mean_absolute_error(y_val, val_naive_pred)

    print(f"[Baseline 1 - Moving Average Naïve] RMSE: {naive_rmse:.4f} | MAE: {naive_mae:.4f}")

    # 2. Linear/Ridge Baseline Model
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    val_ridge_pred = model.predict(X_val)

    ridge_rmse = np.sqrt(mean_squared_error(y_val, val_ridge_pred))
    ridge_mae = mean_absolute_error(y_val, val_ridge_pred)

    print(f"[Baseline 2 - Ridge Regression]     RMSE: {ridge_rmse:.4f} | MAE: {ridge_mae:.4f}")
    
    print("\nSTATUS: Day 6 Baseline Models Trained Successfully!")

if __name__ == "__main__":
    run_baseline_model()