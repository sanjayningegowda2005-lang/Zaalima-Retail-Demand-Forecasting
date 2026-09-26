import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from scipy.stats import norm

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

def optimize_inventory():
    input_path = DATA_DIR / "final_feature_matrix.csv"
    model_path = MODEL_DIR / "lgb_tuned.txt"

    if not input_path.exists() or not model_path.exists():
        print("Required dataset or trained model file missing.")
        return

    print("\n--- DAY 10: INVENTORY OPTIMIZATION & SAFETY STOCK ---")
    df = pd.read_csv(input_path)
    df_clean = df.dropna(subset=["lag_7", "lag_28", "rolling_mean_7", "ema_7"]).copy()

    max_day = df_clean["day_num"].max()
    split_day = max_day - 28

    val = df_clean[df_clean["day_num"] > split_day].copy()

    features = [
        "lag_7", "lag_28", "rolling_mean_7", "rolling_std_7", 
        "ema_7", "ema_28", "is_event", "discount_ratio", 
        "day_of_week", "month", "is_weekend"
    ]

    # Load Model & Predict Demand
    model = lgb.Booster(model_file=str(model_path))
    val["predicted_demand"] = model.predict(val[features])

    # Parameters for Inventory Policy
    LEAD_TIME_DAYS = 7        # Supplier lead time
    SERVICE_LEVEL = 0.95      # 95% Target Service Level
    Z_SCORE = norm.ppf(SERVICE_LEVEL) # ~1.645

    # Group demand stats per item (assuming item_id column exists or global calculation)
    # Estimate forecast standard error (RMSE of predictions)
    forecast_errors = val["sales"] - val["predicted_demand"]
    sigma_demand = np.std(forecast_errors)

    # Calculate Safety Stock (SS) and Reorder Point (ROP)
    val["safety_stock"] = np.ceil(Z_SCORE * sigma_demand * np.sqrt(LEAD_TIME_DAYS)).astype(int)
    val["lead_time_demand"] = val["predicted_demand"] * LEAD_TIME_DAYS
    val["reorder_point"] = np.ceil(val["lead_time_demand"] + val["safety_stock"]).astype(int)

    summary = val[["day_num", "sales", "predicted_demand", "safety_stock", "reorder_point"]].head(10)
    print("\nSample Inventory Optimization Policy Output:")
    print(summary.to_string(index=False))

    # Export Inventory Recommendations
    output_path = DATA_DIR / "inventory_plan.csv"
    val[["day_num", "sales", "predicted_demand", "safety_stock", "reorder_point"]].to_csv(output_path, index=False)
    print(f"\nInventory plan saved to {output_path}")

if __name__ == "__main__":
    optimize_inventory()