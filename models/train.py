import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# Configure local MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow/mlflow.db")
mlflow.set_experiment("Predictive_Maintenance_CMAPSS")

def load_and_preprocess_data(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 1. Compute Remaining Useful Life (RUL)
    # RUL for each row = max_cycle_for_unit - current_cycle
    max_cycles = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.rename(columns={'time_in_cycles': 'max_cycle'}, inplace=True)
    df = df.merge(max_cycles, on='unit_number', how='left')
    df['RUL'] = df['max_cycle'] - df['time_in_cycles']
    df.drop(columns=['max_cycle'], inplace=True)
    
    # 2. Create Target Variable (Failure within 30 cycles)
    df['failure'] = (df['RUL'] <= 30).astype(int)
    
    return df

def engineer_features(df):
    print("Engineering features...")
    # List of sensors to keep based on standard CMAPSS variance analysis
    # Constant sensors (1, 5, 6, 10, 16, 18, 19) are dropped
    critical_sensors = [
        'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 
        'sensor_11', 'sensor_12', 'sensor_13', 'sensor_15', 'sensor_17', 
        'sensor_20', 'sensor_21'
    ]
    
    features_list = ['time_in_cycles', 'op_setting_1', 'op_setting_2'] + critical_sensors
    
    # Let's create rolling window features for critical sensors (window size = 5)
    # Group by unit_number to avoid leakage across units
    rolled_features = []
    
    for sensor in critical_sensors:
        df[f'{sensor}_roll_mean_5'] = df.groupby('unit_number')[sensor].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
        df[f'{sensor}_roll_std_5'] = df.groupby('unit_number')[sensor].rolling(window=5, min_periods=1).std().reset_index(0, drop=True).fillna(0)
        rolled_features.extend([f'{sensor}_roll_mean_5', f'{sensor}_roll_std_5'])
        
    all_features = features_list + rolled_features
    return df, all_features

def train_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'train_FD001.csv')
    models_dir = os.path.join(base_dir, 'models')
    
    os.makedirs(models_dir, exist_ok=True)
    
    # Load and Preprocess
    df = load_and_preprocess_data(data_path)
    df, feature_cols = engineer_features(df)
    
    X = df[feature_cols]
    y = df['failure']
    
    # Split by unit_number to prevent data leakage (engines in training and validation should be completely disjoint)
    units = df['unit_number'].unique()
    train_units, val_units = train_test_split(units, test_size=0.2, random_state=42)
    
    train_idx = df['unit_number'].isin(train_units)
    val_idx = df['unit_number'].isin(val_units)
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save the scaler
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Save feature names list for reference during prediction
    joblib.dump(feature_cols, os.path.join(models_dir, 'feature_cols.joblib'))
    
    # --- Start MLflow Run ---
    with mlflow.start_run(run_name="Predictive_Maintenance_Run") as run:
        print("Training Random Forest Classifier...")
        rf_params = {"n_estimators": 100, "max_depth": 8, "random_state": 42}
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train_scaled, y_train)
        
        # Validation
        rf_preds = rf.predict(X_val_scaled)
        rf_probs = rf.predict_proba(X_val_scaled)[:, 1]
        
        # Calculate Metrics
        rf_acc = accuracy_score(y_val, rf_preds)
        rf_prec = precision_score(y_val, rf_preds)
        rf_rec = recall_score(y_val, rf_preds)
        rf_f1 = f1_score(y_val, rf_preds)
        rf_auc = roc_auc_score(y_val, rf_probs)
        
        print(f"RF Validation - Acc: {rf_acc:.4f}, Prec: {rf_prec:.4f}, Rec: {rf_rec:.4f}, F1: {rf_f1:.4f}, AUC: {rf_auc:.4f}")
        
        # Log RF to MLflow
        mlflow.log_params({f"rf_{k}": v for k, v in rf_params.items()})
        mlflow.log_metrics({
            "rf_accuracy": rf_acc,
            "rf_precision": rf_prec,
            "rf_recall": rf_rec,
            "rf_f1_score": rf_f1,
            "rf_auc": rf_auc
        })
        
        # --- Train XGBoost Classifier ---
        print("Training XGBoost Classifier...")
        xgb_params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1, "random_state": 42}
        # XGBoost works better if feature names are standard, we can pass raw numpy array
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train_scaled, y_train)
        
        # Validation
        xgb_preds = xgb_model.predict(X_val_scaled)
        xgb_probs = xgb_model.predict_proba(X_val_scaled)[:, 1]
        
        xgb_acc = accuracy_score(y_val, xgb_preds)
        xgb_prec = precision_score(y_val, xgb_preds)
        xgb_rec = recall_score(y_val, xgb_preds)
        xgb_f1 = f1_score(y_val, xgb_preds)
        xgb_auc = roc_auc_score(y_val, xgb_probs)
        
        print(f"XGB Validation - Acc: {xgb_acc:.4f}, Prec: {xgb_prec:.4f}, Rec: {xgb_rec:.4f}, F1: {xgb_f1:.4f}, AUC: {xgb_auc:.4f}")
        
        # Log XGB to MLflow
        mlflow.log_params({f"xgb_{k}": v for k, v in xgb_params.items()})
        mlflow.log_metrics({
            "xgb_accuracy": xgb_acc,
            "xgb_precision": xgb_prec,
            "xgb_recall": xgb_rec,
            "xgb_f1_score": xgb_f1,
            "xgb_auc": xgb_auc
        })
        
        # Log models in MLflow registry
        mlflow.sklearn.log_model(rf, "random_forest_model")
        mlflow.xgboost.log_model(xgb_model, "xgboost_model")
        
        # Choose XGBoost as our primary model for API deployment since it usually has slightly better metrics
        best_model = xgb_model if xgb_f1 >= rf_f1 else rf
        best_model_name = "XGBoost" if xgb_f1 >= rf_f1 else "Random Forest"
        print(f"Selecting {best_model_name} as the deployment model based on validation F1 score.")
        
        # Save primary model binary
        model_path = os.path.join(models_dir, 'model.joblib')
        joblib.dump(best_model, model_path)
        print(f"Saved deployment model binary to {model_path}")
        
        # Write metadata
        with open(os.path.join(models_dir, 'model_meta.txt'), 'w') as f:
            f.write(f"Model Type: {best_model_name}\n")
            f.write(f"F1 Score: {max(xgb_f1, rf_f1):.4f}\n")
            f.write(f"Accuracy: {xgb_acc if xgb_f1 >= rf_f1 else rf_acc:.4f}\n")
            f.write(f"Recall: {xgb_rec if xgb_f1 >= rf_f1 else rf_rec:.4f}\n")
            f.write(f"Precision: {xgb_prec if xgb_f1 >= rf_f1 else rf_prec:.4f}\n")
            f.write(f"Run ID: {run.info.run_id}\n")
            
        print("MLflow Run completed successfully.")

if __name__ == '__main__':
    train_pipeline()
