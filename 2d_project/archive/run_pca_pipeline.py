
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.signal import resample
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import derive_self_template, find_dtw_matches_euclidean

DATA_DIR = "/data/gait/data"
OUTPUT_DIR = "/data/gait/2d_project/research_metrics"

# Cache for loaded signals to avoid re-processing 20 times (LOOCV)
SIGNAL_CACHE = {}

def get_data(sid):
    if sid in SIGNAL_CACHE: return SIGNAL_CACHE[sid]
    
    # GT
    gt_csv = f"/data/gait/data/processed_new/S1_{int(sid):02d}_gait_long.csv"
    gt_cycle = None
    gt_rom = np.nan
    if os.path.exists(gt_csv):
        try:
            df = pd.read_csv(gt_csv)
            subset = df[(df['joint'] == 'r.kn.angle') & (df['plane'] == 'sagittal')]
            if not subset.empty:
                arr = resample(subset['condition1_avg'].values, 101)
                gt_cycle = arr
                gt_rom = np.max(arr) - np.min(arr)
        except: pass
        
    # MP
    sid_int = int(sid)
    sid_dir = str(sid_int)
    video_candidates = [
        f"{DATA_DIR}/{sid_dir}/{sid_dir}-2.mp4",
        f"{DATA_DIR}/{sid_dir}/{sid_dir}.mp4"
    ]
    mp_cycle = None
    mp_rom = np.nan
    
    for p in video_candidates:
        if os.path.exists(p):
            try:
                extractor = MediaPipeSagittalExtractor() 
                landmarks, _ = extractor.extract_pose_landmarks(p)
                if landmarks is not None and not landmarks.empty:
                    angles = extractor.calculate_joint_angles(landmarks)
                    signal = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
                    
                    template, _ = derive_self_template(signal)
                    if template is not None:
                        starts = find_dtw_matches_euclidean(signal, template)
                        cycles = []
                        for s in starts:
                            c = signal[s:s+35] 
                            if len(c) > 10: cycles.append(resample(c, 101))
                        if cycles:
                            avg_c = np.mean(cycles, axis=0)
                            mp_cycle = avg_c
                            mp_rom = np.max(avg_c) - np.min(avg_c)
            except: pass
            break
            
    SIGNAL_CACHE[sid] = {'gt': gt_cycle, 'mp': mp_cycle, 'gt_rom': gt_rom, 'mp_rom': mp_rom}
    return SIGNAL_CACHE[sid]

def run_pca_pipeline():
    print("Initializing Data Cache (N=21)...")
    # We need to know which subjects are valid. Using the list from previous analysis.
    # From debug script: 1, 3, 18, 25, 26, 30 are valid r>0.5.
    # But we want to test on ALL who have data.
    # Let's verify who has GT.
    valid_sids = []
    
    # Limit to representative subset for speed
    target_sids = [1, 10, 13, 18, 25, 26, 2, 3] # Add 2,3 since they were loading
    
    # 1. Load All Data (Pre-fetch)
    for i in target_sids:
        d = get_data(i)
        if d['gt'] is not None and d['mp'] is not None:
            valid_sids.append(i)
            print(f"Loaded S{i}")
            
    print(f"Total Valid Subjects: {len(valid_sids)}")
    
    results = []
    
    # 2. LOOCV
    for test_sid in valid_sids:
        # Train Set
        train_sids = [s for s in valid_sids if s != test_sid]
        
        train_gt_matrix = np.array([SIGNAL_CACHE[s]['gt'] for s in train_sids])
        
        # Z-Norm Training Data for PCA
        train_means = np.mean(train_gt_matrix, axis=1, keepdims=True)
        train_stds = np.std(train_gt_matrix, axis=1, keepdims=True)
        train_z = (train_gt_matrix - train_means) / train_stds
        
        # Fit PCA
        pca = PCA(n_components=5)
        pca.fit(train_z)
        
        # Test Data (MP)
        test_mp = SIGNAL_CACHE[test_sid]['mp']
        # Z-Norm Test MP
        mp_mean = np.mean(test_mp)
        mp_std = np.std(test_mp)
        mp_z = (test_mp - mp_mean) / mp_std
        
        # Project & Reconstruct
        coeffs = pca.transform(mp_z.reshape(1, -1))
        recon_z = pca.inverse_transform(coeffs).flatten()
        
        # Rescale Strategy: "Population Mean Scaling"
        # We replace the subject's MP Scale (which is unreliable) with the Training Population's Mean Scale.
        # This assumes "Everyone walks with Average Amplitude".
        pop_mean_amp = np.mean(train_stds) # Average standard deviation of GT
        pop_mean_offset = np.mean(train_means)
        
        recon_scaled = recon_z * pop_mean_amp + pop_mean_offset
        recon_rom = np.max(recon_scaled) - np.min(recon_scaled)
        
        gt_rom = SIGNAL_CACHE[test_sid]['gt_rom']
        mp_rom = SIGNAL_CACHE[test_sid]['mp_rom']
        
        results.append({
            'subject': test_sid,
            'gt_rom': gt_rom,
            'mp_rom': mp_rom,
            'recon_rom': recon_rom,
            'mp_error': abs(mp_rom - gt_rom),
            'recon_error': abs(recon_rom - gt_rom)
        })
        
    df = pd.DataFrame(results)
    
    # 3. Evaluation
    print("\n--- Pipeline Evaluation ---")
    mse_mp = np.mean((df['mp_rom'] - df['gt_rom'])**2)
    mse_recon = np.mean((df['recon_rom'] - df['gt_rom'])**2)
    r2_mp = r2_score(df['gt_rom'], df['mp_rom']) # Likely negative or bad
    r2_recon = r2_score(df['gt_rom'], df['recon_rom'])
    
    print(f"Baseline (Raw MP) RMSE: {np.sqrt(mse_mp):.2f}")
    print(f"Restored (PCA+PopMean) RMSE: {np.sqrt(mse_recon):.2f}")
    print(f"Restored R2: {r2_recon:.4f}")
    
    # 4. Regression on Restored?
    # Can we learn a correction factor for the restored signal?
    # Regress GT ~ Recon_ROM
    reg = LinearRegression()
    X = df[['recon_rom']]
    y = df['gt_rom']
    reg.fit(X, y)
    df['pred_rom'] = reg.predict(X)
    r2_final = r2_score(y, df['pred_rom'])
    print(f"Regression on Restored (GT ~ Recon) R2: {r2_final:.4f}")
    
    # Save Results
    df.to_csv(f"{OUTPUT_DIR}/pca_pipeline_results.csv", index=False)
    
    # Plot
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x='gt_rom', y='mp_rom', data=df, label='Raw MP', alpha=0.5)
    sns.scatterplot(x='gt_rom', y='recon_rom', data=df, label='Restored (PopMean)', color='r')
    plt.plot([20, 80], [20, 80], 'k--')
    plt.title("Impact of PCA Restoration on ROM Accuracy")
    plt.savefig(f"{OUTPUT_DIR}/pca_pipeline_plot.png")

if __name__ == "__main__":
    run_pca_pipeline()
