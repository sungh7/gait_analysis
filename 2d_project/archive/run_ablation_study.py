
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from scipy.signal import resample, butter, filtfilt, savgol_filter
from gait_augmentation import GaitAugmentor
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import derive_self_template, find_dtw_matches_euclidean

DATA_DIR = "/data/gait/data"
OUTPUT_DIR = "/data/gait/2d_project/research_metrics"

def load_vicon_database():
    """Load all available Gait cycles from Vicon (S1-S30)"""
    db = []
    sids = []
    for i in range(1, 31):
        gt_csv = f"/data/gait/data/processed_new/S1_{int(i):02d}_gait_long.csv"
        if os.path.exists(gt_csv):
            try:
                df = pd.read_csv(gt_csv)
                subset = df[(df['joint'] == 'r.kn.angle') & (df['plane'] == 'sagittal')]
                if not subset.empty:
                    # Note: Using the 'condition1_avg' which is a single cycle.
                    # Ideally we want raw cycles to capture variance, but we only have avg processed.
                    # We will augment this to create "Variance".
                    arr = resample(subset['condition1_avg'].values, 101)
                    db.append(arr)
                    sids.append(i)
            except: pass
    return np.array(db), sids

def load_mp_subset():
    """Load MP cycles for specific test subjects"""
    target_sids = [1, 10, 13, 18, 25, 26]
    signals = {}
    gt_refs = {}
    
    for sid in target_sids:
        # Load GT Reference
        gt_csv = f"/data/gait/data/processed_new/S1_{int(sid):02d}_gait_long.csv"
        if os.path.exists(gt_csv):
            df = pd.read_csv(gt_csv)
            subset = df[(df['joint'] == 'r.kn.angle') & (df['plane'] == 'sagittal')]
            if not subset.empty:
                gt_refs[sid] = resample(subset['condition1_avg'].values, 101)
        
        # Load MP Signal (Slow part)
        sid_str = str(sid)
        vid_path = None
        for p in [f"{DATA_DIR}/{sid_str}/{sid_str}-2.mp4", f"{DATA_DIR}/{sid_str}/{sid_str}.mp4"]:
            if os.path.exists(p): vid_path = p; break
            
        if vid_path:
            try:
                extractor = MediaPipeSagittalExtractor()
                landmarks, _ = extractor.extract_pose_landmarks(vid_path)
                if landmarks is not None and not landmarks.empty:
                    angles = extractor.calculate_joint_angles(landmarks)
                    signal = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
                    temp, _ = derive_self_template(signal)
                    if temp is not None:
                        starts = find_dtw_matches_euclidean(signal, temp)
                        cycles = []
                        for s in starts:
                            c = signal[s:s+35]
                            if len(c) > 10: cycles.append(resample(c, 101))
                        if cycles:
                            signals[sid] = np.mean(cycles, axis=0) # Average MP Cycle
            except: pass
            
    return signals, gt_refs

def train_pca_models(vicon_db, mp_data):
    """
    Train 3 PCA variants:
    1. Vicon-PCA: Trained on Vicon GT.
    2. MP-PCA: Trained on MP Data (Self-Supervised domain).
    3. Vicon+Aug: Trained on Synthetically Distorted Vicon.
    """
    # 1. Vicon PCA
    # Z-Norm
    vicon_z = (vicon_db - np.mean(vicon_db, axis=1, keepdims=True)) / np.std(vicon_db, axis=1, keepdims=True)
    pca_vicon = PCA(n_components=5)
    pca_vicon.fit(vicon_z)
    
    # 2. MP PCA
    # Gather all MP cycles available
    if mp_data:
        mp_matrix = np.array(list(mp_data.values()))
        # Augment MP data to stabilize covariance?
        # Let's just use what we have first.
        mp_z = (mp_matrix - np.mean(mp_matrix, axis=1, keepdims=True)) / np.std(mp_matrix, axis=1, keepdims=True)
        n_comp = min(len(mp_matrix), 5)
        pca_mp = PCA(n_components=n_comp)
        pca_mp.fit(mp_z)
    else:
        pca_mp = None
        
    # 3. Vicon + Aug PCA
    # Generate 1000 augmented samples from 20 Vicon subjects
    aug_db = []
    for _ in range(50): # 50 augments per subject
        for v in vicon_db:
            aug_db.append(GaitAugmentor.apply_camera_distortion(v))
    aug_matrix = np.array(aug_db)
    aug_z = (aug_matrix - np.mean(aug_matrix, axis=1, keepdims=True)) / np.std(aug_matrix, axis=1, keepdims=True)
    pca_aug = PCA(n_components=5)
    pca_aug.fit(aug_z)
    
    return pca_vicon, pca_mp, pca_aug

def apply_smoothing(signal, method):
    if method == 'raw': return signal
    if method == 'sg':
        return savgol_filter(signal, window_length=21, polyorder=3)
    if method == 'butter':
        b, a = butter(2, 0.1) # low pass
        return filtfilt(b, a, signal)
    return signal

def run_ablation():
    print("Loading Vicon DB...")
    vicon_db, _ = load_vicon_database()
    print(f"Vicon DB: {vicon_db.shape}")
    
    print("Loading MP Subset (This takes time)...")
    mp_signals, gt_refs = load_mp_subset()
    print(f"MP Subset: {len(mp_signals)} subjects")
    
    print("Training PCA Models...")
    pca_vicon, pca_mp, pca_aug = train_pca_models(vicon_db, mp_signals)
    
    results = []
    
    for sid, mp_raw in mp_signals.items():
        if sid not in gt_refs: continue
        gt = gt_refs[sid]
        
        # Normalize Input for PCA
        mp_z = (mp_raw - np.mean(mp_raw)) / np.std(mp_raw)
        
        # Method 1: Vicon PCA
        c_v = pca_vicon.transform(mp_z.reshape(1, -1))
        rec_v = pca_vicon.inverse_transform(c_v).flatten()
        # Rescale for viz (match GT mean/std roughly to check shape)
        # Actually correlation is invariant to scale.
        
        # Method 2: MP PCA
        c_m = pca_mp.transform(mp_z.reshape(1, -1))
        rec_m = pca_mp.inverse_transform(c_m).flatten()
        
        # Method 3: Vicon+Aug PCA
        c_a = pca_aug.transform(mp_z.reshape(1, -1))
        rec_a = pca_aug.inverse_transform(c_a).flatten()
        
        # Method 4 & 5: Smoothing
        rec_sg = apply_smoothing(mp_z, 'sg')
        rec_but = apply_smoothing(mp_z, 'butter')
        
        # Metrics (Correlation with GT)
        # Raw
        r_raw = np.corrcoef(gt, mp_raw)[0,1]
        
        r_vicon = np.corrcoef(gt, rec_v)[0,1]
        r_mp = np.corrcoef(gt, rec_m)[0,1]
        r_aug = np.corrcoef(gt, rec_a)[0,1]
        r_sg = np.corrcoef(gt, rec_sg)[0,1]
        r_but = np.corrcoef(gt, rec_but)[0,1]
        
        results.append({
            'Subject': sid,
            'Raw': r_raw,
            'Vicon_PCA': r_vicon,
            'MP_PCA': r_mp,
            'Vicon_Aug_PCA': r_aug,
            'SavitzkyGolay': r_sg,
            'Butterworth': r_but
        })
        
    df = pd.DataFrame(results)
    print("\n--- Ablation Results (Correlation) ---")
    print(df)
    df.to_csv(f"{OUTPUT_DIR}/ablation_comparison.csv", index=False)
    
    # Plot Average Improvements
    plt.figure(figsize=(10, 6))
    melted = df.melt(id_vars='Subject', var_name='Method', value_name='Correlation')
    sns.barplot(x='Method', y='Correlation', data=melted)
    plt.title("Method Comparison: PCA Bases vs Smoothing")
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0) # Assume positive
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ablation_barchart.png")

if __name__ == "__main__":
    run_ablation()
