import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

def evaluate():
    input_path = DATA_DIR / "final_feature_matrix.csv"
    model_path = MODEL_DIR / "lgb_tuned.txt"

    if not input_path.exists() or not model_path.exists():
        print("Required dataset or trained model file is missing.")
        return

    print("\n--- DAY 9: MODEL EVALUATION & FEATURE IMPORTANCE ---")
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
    target = "sales"

    X_val = val[features]
    y_val = val[target]

    # Load Model & Predict
    model = lgb.Booster(model_file=str(model_path))
    val["predictions"] = model.predict(X_val)

    # Evaluation Metrics
    rmse = np.sqrt(mean_squared_error(y_val, val["predictions"]))
    mae = mean_absolute_error(y_val, val["predictions"])

    # WAPE (Weighted Absolute Percentage Error)
    wape = (np.abs(y_val - val["predictions"]).sum() / y_val.sum()) * 100

    print(f"Validation RMSE : {rmse:.4f}")
    print(f"Validation MAE  : {mae:.4f}")
    print(f"Validation WAPE : {wape:.2f}%")

    # Feature Importance Breakdown
    importance = model.feature_importance(importance_type="gain")
    feature_imp = pd.DataFrame({
        "Feature": features,
        "Importance_Gain": importance
    }).sort_values(by="Importance_Gain", ascending=False)

    print("\nTop Feature Importances (Gain):")
    print(feature_imp.to_string(index=False))

    # Save summary metrics
    metrics_df = pd.DataFrame([{
        "RMSE": rmse, "MAE": mae, "WAPE_%": wape
    }])
    metrics_df.to_csv(DATA_DIR / "evaluation_metrics.csv", index=False)
    print(f"\nSaved evaluation metrics to {DATA_DIR / 'evaluation_metrics.csv'}")

if __name__ == "__main__":
    evaluate()