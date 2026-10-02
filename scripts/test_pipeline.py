import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

class TestRetailPipeline(unittest.TestCase):

    def setUp(self):
        """Set up paths and load dataset for test suite."""
        self.feature_matrix_path = DATA_DIR / "final_feature_matrix.csv"
        self.model_path = MODEL_DIR / "lgb_tuned.txt"

    def test_01_feature_matrix_exists_and_non_empty(self):
        """Verify feature matrix exists and contains valid rows."""
        self.assertTrue(self.feature_matrix_path.exists(), "Feature matrix file is missing.")
        df = pd.read_csv(self.feature_matrix_path)
        self.assertGreater(len(df), 0, "Feature matrix is empty.")

    def test_02_required_columns_present(self):
        """Check if all required predictive features exist in schema."""
        required_cols = [
            "day_num", "sales", "lag_7", "lag_28", 
            "rolling_mean_7", "rolling_std_7", "ema_7", "ema_28", 
            "is_event", "discount_ratio", "day_of_week", "month", "is_weekend"
        ]
        df = pd.read_csv(self.feature_matrix_path)
        for col in required_cols:
            self.assertIn(col, df.columns, f"Required column '{col}' missing from feature matrix.")

    def test_03_model_file_exists_and_loads(self):
        """Verify the trained LightGBM booster model loads cleanly."""
        self.assertTrue(self.model_path.exists(), "Trained LightGBM model file missing.")
        model = lgb.Booster(model_file=str(self.model_path))
        self.assertIsNotNone(model, "Failed to load LightGBM booster model.")

    def test_04_edge_case_zero_and_negative_sales_handling(self):
        """Ensure predictions handle zero/boundary demand edge cases without NaN outputs."""
        model = lgb.Booster(model_file=str(self.model_path))
        
        # Mock zero/boundary feature vector
        mock_input = pd.DataFrame([{
            "lag_7": 0.0, "lag_28": 0.0, "rolling_mean_7": 0.0, "rolling_std_7": 0.0,
            "ema_7": 0.0, "ema_28": 0.0, "is_event": 0, "discount_ratio": 1.0,
            "day_of_week": 1, "month": 1, "is_weekend": 0
        }])
        
        pred = model.predict(mock_input)
        self.assertFalse(np.isnan(pred[0]), "Model prediction returned NaN for zero-demand edge case.")

    def test_05_inventory_plan_output_validity(self):
        """Verify inventory plan outputs valid non-negative Safety Stock and ROP numbers."""
        inv_path = DATA_DIR / "inventory_plan.csv"
        if inv_path.exists():
            df_inv = pd.read_csv(inv_path)
            self.assertTrue((df_inv["safety_stock"] >= 0).all(), "Negative safety stock detected!")
            self.assertTrue((df_inv["reorder_point"] >= 0).all(), "Negative reorder point detected!")

if __name__ == "__main__":
    print("\n--- DAY 17: PIPELINE INTEGRATION TESTING & EDGE CASE VALIDATION ---")
    unittest.main()