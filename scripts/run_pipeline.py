import sys
import time
from pathlib import Path

# Add project root to path for modular imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_model import evaluate
from scripts.inventory_optimization import optimize_inventory
from scripts.price_elasticity import calculate_elasticity

def execute_pipeline():
    start_time = time.time()
    print("\n==================================================")
    print("      RETAIL DEMAND FORECASTING END-TO-END        ")
    print("==================================================")

    # 1. Run Evaluation & Feature Metrics
    print("\n[STEP 1/3] Running Model Evaluation...")
    evaluate()

    # 2. Run Inventory Optimization Policy
    print("\n[STEP 2/3] Executing Inventory Optimization...")
    optimize_inventory()

    # 3. Run Price Elasticity Analysis
    print("\n[STEP 3/3] Computing Price Elasticity...")
    calculate_elasticity()

    elapsed = time.time() - start_time
    print("\n==================================================")
    print(f" PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
    print("==================================================")

if __name__ == "__main__":
    execute_pipeline()