import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

def run_eda():
    sales_path = DATA_DIR / "sales_train_validation.csv"
    calendar_path = DATA_DIR / "calendar.csv"
    
    if not sales_path.exists():
        print("Data files missing. Please run ingest_m5.py first.")
        return

    sales_df = pd.read_csv(sales_path)
    calendar_df = pd.read_csv(calendar_path)

    # Melt sales data into long time-series format
    d_cols = [c for c in sales_df.columns if c.startswith("d_")]
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    
    sales_long = pd.melt(sales_df, id_vars=id_vars, value_vars=d_cols, var_name="d", value_name="sales")
    
    # Merge with calendar info
    merged_df = pd.merge(sales_long, calendar_df[["d", "date", "weekday", "event_name_1"]], on="d")
    merged_df["date"] = pd.to_datetime(merged_df["date"])
    
    print("\n--- Exploratory Data Analysis Report ---")
    
    # Store-level aggregations
    store_sales = merged_df.groupby("store_id")["sales"].agg(["sum", "mean", "std"]).reset_index()
    print("\n1. Store Sales Performance:")
    print(store_sales.to_string(index=False))
    
    # Day of week seasonality
    day_sales = merged_df.groupby("weekday")["sales"].mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ).reset_index()
    print("\n2. Weekly Sales Seasonality (Avg Units Sold):")
    print(day_sales.to_string(index=False))
    
    # High volatility items
    item_vol = merged_df.groupby("item_id")["sales"].agg(["mean", "std"]).reset_index()
    item_vol["cv"] = item_vol["std"] / (item_vol["mean"] + 1e-5) # Coefficient of variation
    print("\n3. Top 3 Most Volatile Items (Highest Coefficient of Variation):")
    print(item_vol.sort_values(by="cv", ascending=False).head(3).to_string(index=False))

if __name__ == "__main__":
    run_eda()