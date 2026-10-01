import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

def calculate_psi(expected, actual, num_buckets=10):
    """Calculates Population Stability Index (PSI) between baseline and new feature distributions."""
    def scale_range(data, min_val, max_val):
        return (data - min_val) / (max_val - min_val + 1e-6)

    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())

    exp_scaled = scale_range(expected, min_val, max_val)
    act_scaled = scale_range(actual, min_val, max_val)

    buckets = np.linspace(0, 1, num_buckets + 1)
    
    exp_counts, _ = np.histogram(exp_scaled, bins=buckets)
    act_counts, _ = np.histogram(act_scaled, bins=buckets)

    exp_pct = np.where(exp_counts == 0, 0.0001, exp_counts) / len(expected)
    act_pct = np.where(act_counts == 0, 0.0001, act_counts) / len(actual)

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return psi_val

def check_model_drift():
    input_path = DATA_DIR / "final_feature_matrix.csv"
    model_path = MODEL_DIR / "lgb_tuned.txt"

    if not input_path.exists() or not model_path.exists():
        print("Required dataset or trained model file is missing.")
        return

    print("\n--- DAY 16: AUTOMATED DRIFT DETECTION & RETRAINING TRIGGER ---")
    df = pd.read_csv(input_path)
    df_clean = df.dropna(subset=["lag_7", "lag_28", "rolling_mean_7", "ema_7"]).copy()

    max_day = df_clean["day_num"].max()

    # Split historical reference data vs recent operational data (last 28 days)
    baseline_data = df_clean[df_clean["day_num"] <= (max_day - 28)]
    recent_data = df_clean[df_clean["day_num"] > (max_day - 28)].copy()

    features = [
        "lag_7", "lag_28", "rolling_mean_7", "rolling_std_7", 
        "ema_7", "ema_28", "is_event", "discount_ratio", 
        "day_of_week", "month", "is_weekend"
    ]

    # 1. Feature Data Drift Check (PSI)
    print("\nCalculating Feature PSI (Population Stability Index)...")
    psi_results = {}
    for col in ["lag_7", "rolling_mean_7", "discount_ratio"]:
        psi_score = calculate_psi(baseline_data[col], recent_data[col])
        psi_results[col] = psi_score
        print(f"  Feature '{col}' PSI: {psi_score:.4f}")

    avg_psi = np.mean(list(psi_results.values()))

    # 2. Concept Drift / Performance Check
    model = lgb.Booster(model_file=str(model_path))
    recent_preds = model.predict(recent_data[features])
    recent_wape = (np.sum(np.abs(recent_data["sales"] - recent_preds)) / np.sum(recent_data["sales"])) * 100

    print(f"\nRecent Batch WAPE: {recent_wape:.2f}%")

    # 3. Decision Logic for Retraining Trigger
    PSI_THRESHOLD = 0.25      # >0.25 indicates significant population shift
    WAPE_THRESHOLD = 25.0     # >25% WAPE indicates unacceptable error drift

    retrain_needed = (avg_psi > PSI_THRESHOLD) or (recent_wape > WAPE_THRESHOLD)

    print("\n--------------------------------------------------")
    if retrain_needed:
        print("ALERT: Model Drift Detected! Retraining Trigger Activated [STATUS: YES]")
    else:
        print("STATUS: Model Performance & Data Distribution Stable [STATUS: NO RETRAIN NEEDED]")
    print("--------------------------------------------------")

    # Export Drift Log
    drift_report = pd.DataFrame([{
        "Average_PSI": avg_psi,
        "Recent_WAPE_%": recent_wape,
        "Retrain_Triggered": retrain_needed
    }])
    output_path = DATA_DIR / "drift_monitoring_report.csv"
    drift_report.to_csv(output_path, index=False)
    print(f"Drift evaluation log exported to {output_path}")

if __name__ == "__main__":
    check_model_drift()