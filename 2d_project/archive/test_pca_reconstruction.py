
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from scipy.signal import resample
from sklearn.decomposition import PCA
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import derive_self_template, find_dtw_matches_euclidean

DATA_DIR = "/data/gait/data"
OUTPUT_DIR = "/data/gait/2d_project/research_metrics"

def load_gt_signal(sid):
    gt_csv = f"/data/gait/data/processed_new/S1_{int(sid):02d}_gait_long.csv"
    if not os.path.exists(gt_csv): return None
    try:
        df = pd.read_csv(gt_csv)
        subset = df[(df['joint'] == 'r.kn.angle') & (df['plane'] == 'sagittal')]
        if subset.empty: return None
        return resample(subset['condition1_avg'].values, 101)
    except: return None

def extract_mp_signal(sid):
    sid_int = int(sid)
    sid_dir = str(sid_int)
    # Try paths
    video_candidates = [
        f"{DATA_DIR}/{sid_dir}/{sid_dir}-2.mp4",
        f"{DATA_DIR}/{sid_dir}/{sid_dir}.mp4"
    ]
    video_path = None
    for p in video_candidates:
        if os.path.exists(p):
            video_path = p
            break
    if not video_path: return None

    extractor = MediaPipeSagittalExtractor() # Assuming efficient extraction
    # Note: Full extraction is slow. For N=21 this might take 10 mins.
    # To be fast, knowing user wants result, I should check if I have cached MP cycles.
    # I don't. But I can assume 'bad' signals for test.
    # Let's run for ALL valid subjects to get statistics.
    try:
        landmarks, _ = extractor.extract_pose_landmarks(video_path)
    except: return None
    
    if landmarks is None or landmarks.empty: return None
    angles = extractor.calculate_joint_angles(landmarks)
    signal = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
    
    template, _ = derive_self_template(signal)
    if template is None: return None
    starts = find_dtw_matches_euclidean(signal, template)
    
    cycles = []
    for s in starts:
        c = signal[s:s+35] 
        if len(c) > 10: cycles.append(resample(c, 101))
    
    if not cycles: return None
    return np.mean(cycles, axis=0)

def run_pca_reconstruction():
    print("Loading GT Database for PCA Basis...")
    gt_db = []
    subject_ids = []
    
    # 1. Build PCA Basis from GT (All available)
    for i in range(1, 31):
        sig = load_gt_signal(i)
        if sig is not None:
            gt_db.append(sig)
            subject_ids.append(i)
            
    gt_matrix = np.array(gt_db) # Shape (N, 101)
    print(f"GT Database: {gt_matrix.shape}")
    
    # Train PCA
    pca = PCA(n_components=0.95) # Keep 95% variance
    pca.fit(gt_matrix)
    print(f"PCA Components: {pca.n_components_} (Explains 95% variance)")
    
    # 2. Reconstruct MP Signals
    # Filter for N=21 Valid subjects from benchmarks to be consistent
    # Or just iterate all GT subjects
    
    results = []
    
    # Optimization: To avoid re-extracting video which takes forever, 
    # I will create synthetic "Distorted" signals from GT to prove the concept first?
    # No, user wants real data results.
    # I will process a small subset representing Good, Bad, Ugly.
    # S25 (Good Scale), S1 (Good All), S10 (Bad Pattern), S13 (Bad).
    subset_sids = [1, 25, 10, 13, 26, 18] 
    
    print("\nProcessing Subset for Validation...")
    plt.figure(figsize=(15, 10))
    
    for idx, sid in enumerate(subset_sids):
        print(f"Processing S{sid}...")
        gt = load_gt_signal(sid)
        mp = extract_mp_signal(sid)
        
        if gt is None or mp is None: continue
        
        # Normalize MP to match GT Range before Projection? 
        # PCA captures shape and amplitude. If MP is 0.5x amplitude, it acts like a "Low Amplitude Walker".
        # But if it's distorted, shape is wrong.
        # Strict reconstruction: Project Raw MP.
        # But scaling offset matters. PCA assumes mean=0 usually (centered).
        
        # Try 1: Z-Norm Projection (Shape Only Restoration)
        mp_z = (mp - np.mean(mp)) / np.std(mp)
        # We need to project into Z-Normed PCA space?
        # Let's train PCA on Z-Normed GT to learn "Pure Shapes".
        
        # Refit PCA on Z-Normed GT
        gt_z_matrix = (gt_matrix - np.mean(gt_matrix, axis=1, keepdims=True)) / np.std(gt_matrix, axis=1, keepdims=True)
        pca_z = PCA(n_components=5) # Fix to top 5 components
        pca_z.fit(gt_z_matrix)
        
        # Project MP
        coords = pca_z.transform(mp_z.reshape(1, -1))
        recon_z = pca_z.inverse_transform(coords).flatten()
        
        # Rescale Reconstruction to GT Range for visualization (Ideal Scaling)
        # recon_scaled = recon_z * np.std(gt) + np.mean(gt)
        
        # Compare Correlation
        corr_raw = np.corrcoef(gt, mp)[0,1]
        corr_recon = np.corrcoef(gt, recon_z)[0,1] # shape vs shape
        
        # Store
        results.append({'Subject': sid, 'R_Raw': corr_raw, 'R_Recon': corr_recon})
        
        # Plot
        plt.subplot(2, 3, idx+1)
        plt.plot(gt, 'k-', linewidth=2, label='Vicon (GT)')
        # Scale MP for visual overlay
        mp_viz = (mp_z * np.std(gt)) + np.mean(gt)
        plt.plot(mp_viz, 'r--', alpha=0.5, label=f'Raw MP (r={corr_raw:.2f})')
        
        recon_viz = (recon_z * np.std(gt)) + np.mean(gt)
        plt.plot(recon_viz, 'b-', linewidth=2, label=f'PCA Recon (r={corr_recon:.2f})')
        
        plt.title(f"S{sid}: Restoration {corr_raw:.2f} -> {corr_recon:.2f}")
        plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pca_reconstruction_subset.png")
    
    print("\n--- PCA Restoration Results ---")
    print(pd.DataFrame(results))

if __name__ == "__main__":
    run_pca_reconstruction()
