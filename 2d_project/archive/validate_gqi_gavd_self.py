
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.decomposition import PCA
from scipy.signal import resample

# --- HARCODED PATHS ---
DATA_DIR = "/data/gait/data"
GAVD_ROOT = "/data/datasets/GAVD"
MANIFEST_PATH = f"{GAVD_ROOT}/analysis_results/gavd_manifest_with_paths.csv"
CYCLES_DIR = f"{GAVD_ROOT}/mediapipe_cycles"
OUTPUT_DIR = "/data/gait/2d_project/research_metrics"

# --- DATA LOADING UTILS ---

def load_vicon_database():
    """Load all available Gait cycles from Vicon"""
    db = []
    # Using range 1-30
    for i in range(1, 40):
        gt_csv = f"/data/gait/data/processed_new/S1_{int(i):02d}_gait_long.csv"
        if os.path.exists(gt_csv):
            try:
                df = pd.read_csv(gt_csv)
                # Check column names
                # Vicon CSV usually has 'joint', 'plane', 'condition1_avg'
                subset = df[(df['joint'] == 'r.kn.angle') & (df['plane'] == 'sagittal')]
                if not subset.empty:
                    arr = resample(subset['condition1_avg'].values, 101)
                    db.append(arr)
            except: pass
    return np.array(db)

def load_gavd_data(manifest_df, group_label, max_samples=20):
    """Load waveforms for a specific group of GAVD subjects"""
    cycles = []
    
    # Filter by label (checking both columns)
    subset = manifest_df[
        (manifest_df['gait_pat'] == group_label) | 
        (manifest_df['primary_label'] == group_label)
    ]
        
    print(f"Loading {group_label}: Found {len(subset)} sequences. Sampling {max_samples}...")
    
    count = 0
    # Randomized sampling if many
    if len(subset) > max_samples:
        subset = subset.sample(max_samples, random_state=42)
        
    for _, row in subset.iterrows():
        seq = row['seq']
        vid = row['video_id']
        view = row['primary_view'] # 'left side' or 'right side'
        
        if pd.isna(view): continue
        
        # Construct path
        side_folder = view.replace(" ", "_")
        side_suffix = "left" if "left" in view.lower() else "right"
        
        fname = f"{seq}_{vid}_{side_suffix}_cycle.csv"
        fpath = f"{CYCLES_DIR}/{side_folder}/{fname}"
        
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                
                if count == 0 and len(cycles) == 0:
                    print(f"DEBUG: Processing {fpath} for {group_label}")
                    if 'joint' in df.columns:
                        print(f"DEBUG: Unique joints found: {df['joint'].unique()}")
                    if 'plane' in df.columns:
                        print(f"DEBUG: Unique planes found: {df['plane'].unique()}")
                
                # GAVD format check
                if 'joint' in df.columns and 'angle_mean' in df.columns:
                    # Filter for Knee (kn) + Sagittal
                    # Unique joints typically: ['l.hi.angle' 'l.kn.angle' 'l.an.angle']
                    knee_rows = df[df['joint'].astype(str).str.contains('kn', case=False, na=False)]
                    
                    if 'plane' in df.columns:
                        # Case insensitive check, allow 'sagittal' or 'y'
                        # GAVD seems to use 'y' for flexion/sagittal
                        valid_planes = ['sagittal', 'y']
                        knee_rows = knee_rows[knee_rows['plane'].astype(str).str.lower().isin(valid_planes)]
                    
                    if not knee_rows.empty:
                        arr = knee_rows['angle_mean'].values
                        if len(arr) == 101: 
                            cycles.append(arr)
                            count += 1
                        elif len(arr) > 10:
                            arr_res = resample(arr, 101)
                            cycles.append(arr_res)
                            count += 1
                    else:
                        if count == 0 and len(cycles) == 0:
                            print("DEBUG: knee_rows empty after filter checking 'kn' and 'sagittal/y'.")

                else:
                    # Fallback for simple format
                    col_name = f"{side_suffix}_knee_angle"
                    if col_name not in df.columns: col_name = 'knee_angle'
                    if col_name in df.columns:
                        arr = df[col_name].values
                        if len(arr) > 10:
                            arr_res = resample(arr, 101)
                            cycles.append(arr_res)
                            count += 1
                            
            except Exception as e:
                pass
                
    return np.array(cycles)

# --- CLASS DEFINITION ---

class GaitQualityIndex:
    def __init__(self, n_components=5):
        self.pca = PCA(n_components=n_components)
        self.mean_ = None
        self.std_ = None
        self.n_components = n_components
        self.q_limit = None
        self.t2_limit = None
        
    def fit(self, X):
        """Fit PCA and establish statistical limits (95% CI) based on Normal Data X"""
        # Z-Normalize Train Data
        self.mean_ = np.mean(X, axis=1, keepdims=True)
        self.std_ = np.std(X, axis=1, keepdims=True)
        X_z = (X - self.mean_) / self.std_
        
        self.pca.fit(X_z)
        
        # Calculate Q and T2 for training data (to set limits)
        scores = self.pca.transform(X_z)
        X_recon = self.pca.inverse_transform(scores)
        residuals = X_z - X_recon
        
        # Q-statistic (SPE)
        q_stats = np.sum(residuals**2, axis=1)
        # 95th percentile limit
        self.q_limit = np.percentile(q_stats, 95)
        
        # Hotelling's T2
        # T2 = sum(score^2 / lambda)
        lambdas = self.pca.explained_variance_
        t2_stats = np.sum(scores**2 / lambdas, axis=1)
        self.t2_limit = np.percentile(t2_stats, 95)
        
        print(f"Model Fitted. Q_limit(95%): {self.q_limit:.2f}, T2_limit(95%): {self.t2_limit:.2f}")
        return q_stats, t2_stats
        
    def evaluate(self, cycle):
        """Evaluate a single cycle"""
        # Z-Norm single cycle
        c_z = (cycle - np.mean(cycle)) / np.std(cycle)
        c_z_reshaped = c_z.reshape(1, -1)
        
        score = self.pca.transform(c_z_reshaped)
        recon = self.pca.inverse_transform(score)
        resid = c_z_reshaped - recon
        
        q = np.sum(resid**2)
        
        lambdas = self.pca.explained_variance_
        t2 = np.sum(score**2 / lambdas)
        
        return q, t2, recon.flatten()

# --- MAIN RUN ---

def run_validation():
    # 1. Train GQI on Vicon (Normal Baseline)
    print("Training GQI on Vicon...")
    vicon_db = load_vicon_database()
    if len(vicon_db) == 0:
        print("Error: Vicon DB empty.")
        
    gqi = GaitQualityIndex(n_components=5)
    gqi.fit(vicon_db)
    
    # 2. Load GAVD Manifest
    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return
        
    manifest = pd.read_csv(MANIFEST_PATH)
    
    # 3. Load Groups
    # Normal (MP)
    normal_mp = load_gavd_data(manifest, 'Normal Gait', max_samples=25)
    
    # Pathologies
    parkinsons = load_gavd_data(manifest, 'parkinsons', max_samples=15)
    cp = load_gavd_data(manifest, 'cerebral palsy', max_samples=15)
    
    results = []
    
    def eval_group(data, label):
        if len(data) == 0: return
        for c in data:
            q, t2, _ = gqi.evaluate(c)
            # Normalize index
            q_idx = q / gqi.q_limit
            results.append({'Group': label, 'Q_stat': q, 'T2_stat': t2, 'Q_Index': q_idx})

    eval_group(vicon_db, 'Vicon (GT)')
    eval_group(normal_mp, 'GAVD Normal')
    eval_group(parkinsons, 'Parkinsons')
    eval_group(cp, 'Cerebral Palsy')
    
    if not results:
        print("No results generated.")
        return

    df = pd.DataFrame(results)
    
    # 4. Visualization
    plt.figure(figsize=(10, 6))
    
    sns.boxplot(x='Group', y='Q_stat', data=df)
    plt.axhline(gqi.q_limit, color='k', linestyle='--', label='Normal Limit (Vicon)')
    plt.yscale('log') # Log scale because Q can be huge
    plt.title("GQI (Q-statistic) Distribution by Pathology")
    plt.ylabel("Q-statistic (Residual Error) - Log Scale")
    plt.xticks(rotation=45)
    
    output_png = f"{OUTPUT_DIR}/gqi_gavd_validation_standalone.png"
    plt.tight_layout()
    # Ensure dir exists
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plt.savefig(output_png)
    print(f"Saved plot to {output_png}")
    
    print("\n--- GQI Validation Results (Median Q) ---")
    print(df.groupby('Group')['Q_stat'].median())
    
    print("\n--- GQI Validation Results (Median T2) ---")
    print(df.groupby('Group')['T2_stat'].median())
    
    # Save CSV
    df.to_csv(f"{OUTPUT_DIR}/gqi_gavd_results.csv", index=False)

if __name__ == "__main__":
    run_validation()
