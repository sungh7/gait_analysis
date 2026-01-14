import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib

# --- MLP Model Definition ---
class GaitMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GaitMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def train_mlp_fold(X_train, y_train, X_test, input_dim, output_dim, epochs=200):
    model = GaitMLP(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).numpy()
        
    return y_pred

def compare_models():
    # Load data
    with open("/data/gait/paired_waveforms.json", 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples.")
    
    X_list = []
    y_list = []
    groups = []
    
    for _, row in df.iterrows():
        mp_cycle = np.array(row['mp_cycle'])
        gt_cycle = np.array(row['gt_cycle'])
        
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
    
    # Scaling for MLP/Linear
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # LOGO CV
    logo = LeaveOneGroupOut()
    
    results = {
        'Linear': {'rmse': [], 'corr': []},
        'RandomForest': {'rmse': [], 'corr': []},
        'MLP': {'rmse': [], 'corr': []}
    }
    
    print("\nStarting Comparison (Leave-One-Group-Out)...")
    
    fold = 0
    for train_index, test_index in logo.split(X, y, groups):
        subject = groups[test_index][0]
        
        # Split Data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        X_train_s, X_test_s = X_scaled[train_index], X_scaled[test_index]
        
        # 1. Linear Regression
        lin = LinearRegression()
        lin.fit(X_train_s, y_train)
        y_pred_lin = lin.predict(X_test_s)
        
        # 2. Random Forest (Unscaled is fine, but scaled works too)
        rf = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        
        # 3. MLP
        y_pred_mlp = train_mlp_fold(X_train_s, y_train, X_test_s, X.shape[1], y.shape[1])
        
        # Evaluate
        for name, pred in [('Linear', y_pred_lin), ('RandomForest', y_pred_rf), ('MLP', y_pred_mlp)]:
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            
            fold_corrs = []
            for i in range(len(y_test)):
                c = np.corrcoef(y_test[i], pred[i])[0, 1]
                fold_corrs.append(c)
            mean_corr = np.mean(fold_corrs)
            
            results[name]['rmse'].append(rmse)
            results[name]['corr'].append(mean_corr)
            
        print(f"Fold {fold+1} ({subject}): Lin RMSE={results['Linear']['rmse'][-1]:.2f}, RF RMSE={results['RandomForest']['rmse'][-1]:.2f}, MLP RMSE={results['MLP']['rmse'][-1]:.2f}")
        fold += 1
        
    # Summary
    print("\n" + "="*40)
    print("FINAL COMPARISON RESULTS")
    print("="*40)
    print(f"{'Model':<15} {'Avg RMSE':<10} {'Avg Corr':<10}")
    print("-" * 40)
    
    summary_data = []
    for name in results:
        avg_rmse = np.mean(results[name]['rmse'])
        avg_corr = np.mean(results[name]['corr'])
        print(f"{name:<15} {avg_rmse:<10.2f} {avg_corr:<10.3f}")
        summary_data.append({'Model': name, 'RMSE': avg_rmse, 'Correlation': avg_corr})
        
    # Plot Comparison
    models = [d['Model'] for d in summary_data]
    rmses = [d['RMSE'] for d in summary_data]
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, rmses, color=['skyblue', 'lightgreen', 'salmon'])
    plt.title("Model Comparison (RMSE)")
    plt.ylabel("RMSE (degrees)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("/data/gait/model_comparison_rmse.png")
    print("\nComparison plot saved to /data/gait/model_comparison_rmse.png")

if __name__ == "__main__":
    compare_models()
