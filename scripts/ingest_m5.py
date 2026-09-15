import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

def create_synthetic_m5_data():
    """Generates sample M5-structured datasets for development if raw files are absent."""
    DATA_DIR.mkdir(exist_ok=True)
    
    print("Generating sample M5 sales and calendar data...")
    dates = pd.date_range(start="2024-01-01", periods=180, freq="D")
    
    calendar = pd.DataFrame({
        "date": dates,
        "wm_yr_wk": (dates.isocalendar().year * 100) + dates.isocalendar().week,
        "weekday": dates.day_name(),
        "wday": dates.dayofweek + 1,
        "month": dates.month,
        "year": dates.year,
        "d": [f"d_{i+1}" for i in range(len(dates))],
        "event_name_1": [np.random.choice(["SuperBowl", "LaborDay", None], p=[0.02, 0.02, 0.96]) for _ in range(len(dates))]
    })
    
    items = [f"FOODS_3_{i:03d}" for i in range(1, 11)]
    stores = ["CA_1", "CA_2", "TX_1"]
    
    sales_list = []
    for store in stores:
        for item in items:
            row = {
                "id": f"{item}_{store}_validation",
                "item_id": item,
                "dept_id": "FOODS_3",
                "cat_id": "FOODS",
                "store_id": store,
                "state_id": store[:2]
            }
            # Simulate daily sales quantities
            for i in range(len(dates)):
                row[f"d_{i+1}"] = np.random.poisson(lam=np.random.randint(5, 25))
            sales_list.append(row)
            
    sales = pd.DataFrame(sales_list)
    
    calendar.to_csv(DATA_DIR / "calendar.csv", index=False)
    sales.to_csv(DATA_DIR / "sales_train_validation.csv", index=False)
    print("Sample datasets saved to data/ directory.")

def validate_and_load_data():
    """Loads raw sales/calendar data and prints basic quality check summaries."""
    sales_path = DATA_DIR / "sales_train_validation.csv"
    calendar_path = DATA_DIR / "calendar.csv"
    
    if not sales_path.exists() or not calendar_path.exists():
        create_synthetic_m5_data()
        
    sales_df = pd.read_csv(sales_path)
    calendar_df = pd.read_csv(calendar_path)
    
    print("\n--- Data Ingestion Quality Summary ---")
    print(f"Sales Data Shape: {sales_df.shape}")
    print(f"Calendar Data Shape: {calendar_df.shape}")
    print(f"Null Values in Sales: {sales_df.isnull().sum().sum()}")
    print(f"Unique Items Tracked: {sales_df['item_id'].nunique()}")
    print(f"Unique Stores Tracked: {sales_df['store_id'].nunique()}")

if __name__ == "__main__":
    validate_and_load_data()