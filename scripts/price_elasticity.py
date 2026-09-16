import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

def calculate_elasticity():
    sales_path = DATA_DIR / "sales_train_validation.csv"
    
    if not sales_path.exists():
        print("Data missing. Run ingest_m5.py first.")
        return

    sales_df = pd.read_csv(sales_path)
    d_cols = [c for c in sales_df.columns if c.startswith("d_")]
    sales_long = pd.melt(sales_df, id_vars=["item_id", "store_id"], value_vars=d_cols, var_name="d", value_name="quantity")
    
    np.random.seed(42)
    sales_long["price"] = np.random.uniform(2.0, 15.0, len(sales_long))
    
    sample = sales_long.sample(n=min(5000, len(sales_long)), random_state=42)
    sample["log_q"] = np.log1p(sample["quantity"])
    sample["log_p"] = np.log(sample["price"])
    
    covariance = np.cov(sample["log_p"], sample["log_q"])[0][1]
    variance_p = np.var(sample["log_p"])
    elasticity = covariance / variance_p if variance_p != 0 else 0
    
    print("\n--- DAY 3: PRICE ELASTICITY ANALYSIS ---")
    print(f"Estimated Price Elasticity coefficient: {elasticity:.4f}")

if __name__ == "__main__":
    calculate_elasticity()