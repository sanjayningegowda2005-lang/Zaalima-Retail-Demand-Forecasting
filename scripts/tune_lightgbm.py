import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error

DATA_DIR = Path("data")

def tune_lgb():
    input_path = DATA_DIR / "final_feature_matrix.csv"
    if not input_path.exists():
        print("Feature matrix missing.")
        return

    print("\n--- DAY 8: LIGHTGBM HYPERPARAMETER TUNING ---")
    df = pd.read_csv(input_path)
    df_clean = df.dropna(subset=["lag_7", "lag_28", "rolling_mean_7", "ema_7"]).copy()

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

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'random_state': 42,
            'verbose': -1
        }

        model = lgb.train(params, train_data, num_boost_round=150, valid_sets=[val_data])
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    print("Running Optuna hyperparameter trials...")
    study.optimize(objective, n_trials=10)

    print(f"\nBest Trial RMSE: {study.best_value:.4f}")
    print("Best Parameters Found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Retrain tuned model
    best_params = study.best_params
    best_params.update({'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'random_state': 42})
    
    tuned_model = lgb.train(best_params, train_data, num_boost_round=200, valid_sets=[val_data])
    
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    tuned_model.save_model(model_dir / "lgb_tuned.txt")
    print(f"\nTuned model successfully saved to {model_dir / 'lgb_tuned.txt'}")

if __name__ == "__main__":
    tune_lgb()