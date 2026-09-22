import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATA_DIR = Path("data")

def train_lgb_model():
    input_path = DATA_DIR / "final_feature_matrix.csv"
    
    if not input_path.exists():
        print("Feature matrix missing.")
        return

    print("\n--- DAY 7: LIGHTGBM DEMAND FORECASTING ---")
    df = pd.read_csv(input_path)

    # Clean lag/rolling NaNs
    df_clean = df.dropna(subset=["lag_7", "lag_28", "rolling_mean_7", "ema_7"]).copy()

    # Time-based split (last 28 days for validation)
    max_day = df_clean["day_num"].max()
    split_day = max_day - 28

    train = df_clean[df_clean["day_num"] <= split_day]
    val = df_clean[df_clean["day_num"] > split_day]

    features = [
        "lag_7", "lag_28", "rolling_mean_7", "rolling_std_7", 
        "ema_7", "ema_28", "is_event", "discount_ratio", 
        "day_of_week", "month", "is_weekend"
    ]
    target = "sales"

    X_train, y_train = train[features], train[target]
    X_val, y_val = val[features], val[target]

    # Convert to LightGBM Datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42,
        'verbose': -1
    }

    # Train model
    print("Training LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[train_data, val_data]
    )

    # Evaluate
    val_preds = model.predict(X_val)
    lgb_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    lgb_mae = mean_absolute_error(y_val, val_preds)

    print(f"\n[Day 7 - LightGBM] Validation RMSE: {lgb_rmse:.4f} | MAE: {lgb_mae:.4f}")

    # Save trained model inside models directory
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model.save_model(model_dir / "lgb_baseline.txt")
    print(f"Model saved to {model_dir / 'lgb_baseline.txt'}")

if __name__ == "__main__":
    train_lgb_model()