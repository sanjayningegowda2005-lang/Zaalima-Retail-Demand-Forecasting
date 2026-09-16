import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

def run_quality_checks():
    sales_path = DATA_DIR / "sales_train_validation.csv"
    calendar_path = DATA_DIR / "calendar.csv"
    
    if not sales_path.exists() or not calendar_path.exists():
        print("Data missing. Please run ingest_m5.py first.")
        return

    sales_df = pd.read_csv(sales_path)
    calendar_df = pd.read_csv(calendar_path)
    
    print("\n--- DAY 3: DATA QUALITY CHECK REPORT ---")
    
    # 1. Check null values
    sales_nulls = sales_df.isnull().sum().sum()
    calendar_nulls = calendar_df.isnull().sum().sum()
    print(f"[CHECK 1] Null Values -> Sales: {sales_nulls} | Calendar: {calendar_nulls}")
    
    # 2. Check duplicates
    sales_dupes = sales_df.duplicated(subset=['id']).sum()
    print(f"[CHECK 2] Duplicate Record IDs -> {sales_dupes}")
    
    # 3. Check negative sales values
    d_cols = [c for c in sales_df.columns if c.startswith("d_")]
    negative_sales = (sales_df[d_cols] < 0).sum().sum()
    print(f"[CHECK 3] Negative Sales Quantities -> {negative_sales}")
    
    # Validation status summary
    if sales_nulls == 0 and sales_dupes == 0 and negative_sales == 0:
        print("\nSTATUS: ALL DATA QUALITY CHECKS PASSED SUCCESSFULLY!")
    else:
        print("\nSTATUS: DATA QUALITY ISSUES DETECTED - REVIEW LOGS.")

if __name__ == "__main__":
    run_quality_checks()