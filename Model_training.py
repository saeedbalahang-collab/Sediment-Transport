!pip install -q xgboost optuna shap matplotlib pandas scikit-learn joblib seaborn
import os
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import optuna
import xgboost as xgb
# ---------------- SETTINGS ---------------- #
OUTPUT_DIR = "/content/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_SEED = 42
N_TRIALS = 50
N_SPLITS = 5
# ---------------- LOAD DATA ---------------- #
# Load dataset from GitHub repository
url = 'https://raw.githubusercontent.com/saeedbalahang-collab/Sediment-Transport/main/data.csv'  # Update with your GitHub URL
df = pd.read_csv(url)
# ---------------- FEATURES ---------------- #
feature_cols = ["d*", "D/d50", "sheilds parameter", 'Frd', 'W/D']
target_col = "measured log(Phi)"
# Drop any missing values in key columns
df = df.dropna(subset=feature_cols + [target_col])
# ---------------- TRAIN/TEST SPLIT ---------------- #
X = df[feature_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
# Save train and test sets to CSV files
X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)
# ---------------- SCALING ---------------- #
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=feature_cols)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))
# ---------------- OPTUNA TUNING ---------------- #
def objective(trial):
    param = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
    }
    
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    rmses = []
    
    for tr_idx, val_idx in cv.split(X_train_scaled):
        X_tr, X_val = X_train_scaled.iloc[tr_idx], X_train_scaled.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        model = xgb.XGBRegressor(**param, tree_method="hist", random_state=RANDOM_SEED)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        rmses.append(rmse)
        
        # Print each fold's RMSE
        print(f"Fold RMSE: {rmse:.4f}")
    return np.mean(rmses)
print("\n🔹 Starting Optuna hyperparameter optimization...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS)
best_params = study.best_trial.params
joblib.dump(study, os.path.join(OUTPUT_DIR, "hydraulics_optuna.pkl"))
print("Best parameters:", best_params)
# ---------------- TRAIN FINAL MODEL ---------------- #
model = xgb.XGBRegressor(**best_params, tree_method="hist", random_state=RANDOM_SEED)
model.fit(X_train_scaled, y_train, verbose=False)
joblib.dump(model, os.path.join(OUTPUT_DIR, "XGBoost.pkl"))
# ---------------- EVALUATE ---------------- #
pred_train = model.predict(X_train_scaled)
pred_test  = model.predict(X_test_scaled)

metrics = {
    "Train_RMSE": np.sqrt(mean_squared_error(y_train, pred_train)),
    "Test_RMSE": np.sqrt(mean_squared_error(y_test, pred_test)),
    "Train_R2": r2_score(y_train, pred_train),
    "Test_R2": r2_score(y_test, pred_test)
}
print("\n✅ Model performance:")
print(metrics)
pd.DataFrame([metrics]).to_csv(os.path.join(OUTPUT_DIR, "metrics_summary.csv"), index=False)
