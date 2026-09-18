import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

def build_advanced_features():
    input_path = DATA_DIR / "engineered_features.csv"
    
    if not input_path.exists():
        print("Engineered features dataset missing. Run Day 4 feature engineering first.")
        return

    print("\n--- DAY 5: ADVANCED FEATURE ENGINEERING PIPELINE ---")
    df = pd.read_csv(input_path)

    # 1. Temporal & Calendar Seasonality Features
    print("Generating calendar seasonality features (wday, month, quarter, is_weekend)...")
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # 2. Exponential Moving Averages (EMA) for Trend Capture
    print("Calculating Exponential Moving Averages (EMA_7, EMA_28)...")
    df["ema_7"] = df.groupby("id")["sales"].transform(lambda x: x.shift(1).ewm(span=7, adjust=False).mean())
    df["ema_28"] = df.groupby("id")["sales"].transform(lambda x: x.shift(1).ewm(span=28, adjust=False).mean())

    # 3. Simulated Price & Discount Ratio Features
    print("Adding relative pricing and discount momentum indicators...")
    np.random.seed(42)
    df["sell_price"] = np.random.uniform(2.0, 15.0, len(df))
    df["max_price"] = df.groupby("id")["sell_price"].transform("max")
    df["discount_ratio"] = 1 - (df["sell_price"] / df["max_price"])

    # Save final feature set
    output_path = DATA_DIR / "final_feature_matrix.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\nSTATUS: Advanced Feature Matrix created successfully!")
    print(f"Dataset Shape: {df.shape}")
    print(f"New Features Generated: {['day_of_week', 'month', 'quarter', 'is_weekend', 'ema_7', 'ema_28', 'discount_ratio']}")

if __name__ == "__main__":
    build_advanced_features()