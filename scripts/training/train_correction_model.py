import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib

def train_model():
    # Load data
    with open("/data/gait/paired_waveforms.json", 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples.")
    
    # Prepare features
    # X: MP Cycle (101) + Joint One-Hot (3) + Side One-Hot (2)
    # y: GT Cycle (101)
    
    X_list = []
    y_list = []
    groups = []
    
    for _, row in df.iterrows():
        mp_cycle = np.array(row['mp_cycle'])
        gt_cycle = np.array(row['gt_cycle'])
        
        # Normalize/Standardize? 
        # MP angles are in degrees. GT in degrees.
        # Random Forest is scale invariant, so raw degrees are fine.
        
        # Joint One-Hot
        joint = row['joint']
        is_hip = 1 if joint == 'hip' else 0
        is_knee = 1 if joint == 'knee' else 0
        is_ankle = 1 if joint == 'ankle' else 0
        
        # Side One-Hot
        side = row['side']
        is_left = 1 if side == 'left' else 0
        is_right = 1 if side == 'right' else 0
        
        features = np.concatenate([
            mp_cycle,
            [is_hip, is_knee, is_ankle],
            [is_left, is_right]
        ])
        
        X_list.append(features)
        y_list.append(gt_cycle)
        groups.append(row['subject_id'])
        
    X = np.array(X_list)
    y = np.array(y_list)
    groups = np.array(groups)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Leave-One-Group-Out CV
    logo = LeaveOneGroupOut()
    
    rmses = []
    corrs = []
    
    print("\nStarting Cross-Validation...")
    
    # To save time, we can just do a few folds or train on all.
    # But for validation, we need CV.
    
    fold = 0
    for train_index, test_index in logo.split(X, y, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Train
        regr = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
        regr.fit(X_train, y_train)
        
        # Predict
        y_pred = regr.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Correlation per sample
        fold_corrs = []
        for i in range(len(y_test)):
            c = np.corrcoef(y_test[i], y_pred[i])[0, 1]
            fold_corrs.append(c)
        
        mean_corr = np.mean(fold_corrs)
        
        rmses.append(rmse)
        corrs.append(mean_corr)
        
        print(f"Fold {fold+1} (Subject {groups[test_index][0]}): RMSE={rmse:.2f}, Corr={mean_corr:.3f}")
        fold += 1
        
    print(f"\nAverage RMSE: {np.mean(rmses):.2f}")
    print(f"Average Correlation: {np.mean(corrs):.3f}")
    
    # Train Final Model on All Data
    print("\nTraining Final Model on All Data...")
    final_model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    final_model.fit(X, y)
    
    # Save Model
    joblib.dump(final_model, "/data/gait/omcs_correction_model.joblib")
    print("Model saved to /data/gait/omcs_correction_model.joblib")
    
    # Save Example Plot
    # Pick a random sample from the last fold
    idx = 0
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[idx], label='Ground Truth', color='black', linewidth=2)
    plt.plot(X_test[idx][:101], label='Input (MP)', color='red', linestyle='--')
    plt.plot(y_pred[idx], label='Predicted (Corrected)', color='green', linewidth=2)
    plt.title(f"Prediction Example (Subject {groups[test_index][0]})")
    plt.legend()
    plt.grid(True)
    plt.savefig("/data/gait/correction_example.png")
    print("Example plot saved to /data/gait/correction_example.png")

if __name__ == "__main__":
    train_model()
