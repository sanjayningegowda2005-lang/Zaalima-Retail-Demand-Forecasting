import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

def build_features():
    sales_path = DATA_DIR / "sales_train_validation.csv"
    calendar_path = DATA_DIR / "calendar.csv"
    
    if not sales_path.exists() or not calendar_path.exists():
        print("Data missing. Please run ingest_m5.py first.")
        return

    print("\n--- DAY 4: FEATURE ENGINEERING PIPELINE ---")
    sales_df = pd.read_csv(sales_path)
    calendar_df = pd.read_csv(calendar_path)

    # Melt sales into long format
    d_cols = [c for c in sales_df.columns if c.startswith("d_")]
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    
    sales_long = pd.melt(sales_df, id_vars=id_vars, value_vars=d_cols, var_name="d", value_name="sales")
    
    # Merge calendar event indicators
    df = pd.merge(sales_long, calendar_df[["d", "date", "wday", "month", "year", "event_name_1"]], on="d")
    df["is_event"] = df["event_name_1"].notnull().astype(int)
    
    # Sort for time-series feature calculation
    df["day_num"] = df["d"].str.replace("d_", "").astype(int)
    df = df.sort_values(by=["id", "day_num"]).reset_index(drop=True)

    # 1. Lag Features (7-day and 28-day demand history)
    print("Generating demand lag features (lag_7, lag_28)...")
    df["lag_7"] = df.groupby("id")["sales"].shift(7)
    df["lag_28"] = df.groupby("id")["sales"].shift(28)

    # 2. Rolling Window Statistics
    print("Calculating rolling mean and standard deviation (7-day, 28-day)...")
    df["rolling_mean_7"] = df.groupby("id")["sales"].transform(lambda x: x.shift(1).rolling(7).mean())
    df["rolling_std_7"] = df.groupby("id")["sales"].transform(lambda x: x.shift(1).rolling(7).std())
    df["rolling_mean_28"] = df.groupby("id")["sales"].transform(lambda x: x.shift(1).rolling(28).mean())

    # Save engineered feature dataset
    output_path = DATA_DIR / "engineered_features.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\nSTATUS: Feature matrix created successfully!")
    print(f"Features Dataset Shape: {df.shape}")
    print(f"Sample columns generated: {[c for c in df.columns if 'lag' in c or 'rolling' in c or 'is_event' in c]}")

if __name__ == "__main__":
    build_features()