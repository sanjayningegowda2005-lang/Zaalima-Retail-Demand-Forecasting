import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from datetime import datetime

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

def run_batch_forecasting():
    input_path = DATA_DIR / "final_feature_matrix.csv"
    model_path = MODEL_DIR / "lgb_tuned.txt"

    if not input_path.exists() or not model_path.exists():
        print("Required dataset or trained model file is missing.")
        return

    print("\n--- DAY 15: BATCH FORECASTING PIPELINE ---")
    df = pd.read_csv(input_path)
    df_clean = df.dropna(subset=["lag_7", "lag_28", "rolling_mean_7", "ema_7"]).copy()

    # Select the most recent 28 days for batch production scoring
    max_day = df_clean["day_num"].max()
    latest_batch = df_clean[df_clean["day_num"] > (max_day - 28)].copy()

    features = [
        "lag_7", "lag_28", "rolling_mean_7", "rolling_std_7", 
        "ema_7", "ema_28", "is_event", "discount_ratio", 
        "day_of_week", "month", "is_weekend"
    ]

    # Load Tuned Booster Model
    model = lgb.Booster(model_file=str(model_path))
    
    # Generate Batch Predictions
    latest_batch["batch_forecast"] = model.predict(latest_batch[features])
    latest_batch["batch_forecast"] = np.maximum(0, np.round(latest_batch["batch_forecast"])) # Floor at 0 demand
    latest_batch["forecast_generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format Batch Output
    batch_output = latest_batch[["day_num", "sales", "batch_forecast", "forecast_generated_at"]]
    
    output_path = DATA_DIR / "batch_demand_forecast.csv"
    batch_output.to_csv(output_path, index=False)
    
    print(f"Batch predictions successfully generated for {len(batch_output)} records.")
    print(f"Production forecast saved to {output_path}")

    print("\nSample Batch Forecast Output:")
    print(batch_output.head(10).to_string(index=False))

if __name__ == "__main__":
    run_batch_forecasting()