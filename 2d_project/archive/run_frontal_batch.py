import pandas as pd
import numpy as np
from scipy.signal import resample, find_peaks
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import glob
import os
from self_driven_segmentation import find_dtw_matches_euclidean

DATA_DIR = "/data/gait/data"
GT_DIR = "/data/gait/data/processed_new"

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba, axis=1)
    norm_bc = np.linalg.norm(bc, axis=1)
    
    # Avoid division by zero
    norm_ba[norm_ba == 0] = 1e-6
    norm_bc[norm_bc == 0] = 1e-6
    
    cosine_angle = np.sum(ba*bc, axis=1) / (norm_ba * norm_bc)
    
    # Clip to valid range
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def load_mp_frontal(sid):
    try:
        sid_str = str(int(sid))
        # Robust CSV search
        search_pattern = f'{DATA_DIR}/{sid_str}/{sid_str}-1_front_pose_*.csv'
        files = glob.glob(search_pattern)
        
        if not files:
            # Try integer fallback
            search_pattern = f'{DATA_DIR}/{int(sid)}/{int(sid)}-1_front_pose_*.csv'
            files = glob.glob(search_pattern)
            
        if not files: return None, None
        
        csv_path = files[0]
        df = pd.read_csv(csv_path)
        
        hip = df[['RIGHT_HIP_x', 'RIGHT_HIP_y']].values
        knee = df[['RIGHT_KNEE_x', 'RIGHT_KNEE_y']].values
        ankle = df[['RIGHT_ANKLE_x', 'RIGHT_ANKLE_y']].values
        heel_y = df['RIGHT_HEEL_y'].values
        
        # Valgus Proxy: 180 - Angle (to match Adduction sign convention roughly)
        # Actually in pilot we found we had to invert it.
        # Let's use -(Angle - 180) -> 180 - Angle.
        raw_angle = calculate_angle(hip, knee, ankle)
        valgus_proxy = 180 - raw_angle
        
        # Signal for Seg: Heel Y (Inverted so PEAKS = Ground Contact)
        # In MP Y coordinates, larger Y is lower (feet). 
        # So Peaks in Raw Y are feet on ground.
        # Wait, in standard graph, Y=0 is top. 
        # So "Feet on ground" is MAX Y.
        # If we want peaks, we just use Raw Y?
        # Let's re-verify S3 logic.
        # In test_frontal_multi.py we used -heel_y. This implies we wanted Minima of Y (which corresponds to Top of frame? No).
        # Ah, if Y=1000 is floor, Y=0 is head.
        # We want Step events.
        # Usually "Heel Strike" is when Heel Y is Lowest? No, Highest value (lowest on screen).
        # We used -heel_y, so we looked for "Minima of Y" -> "Maxima of Height" (Head)?
        # That's Swing Phase. Steps are cyclic, so it works either way.
        # I will stick to what worked for S3 in pilot: `-heel_y`.
        
        signal_for_seg = -heel_y 
        
        return valgus_proxy, signal_for_seg
        
    except Exception as e:
        # print(f"MP Load Error {sid}: {e}")
        return None, None

def load_gt_frontal(sid):
    try:
        gt_sid = f"{int(sid):02d}"
        path = f'{GT_DIR}/S1_{gt_sid}_gait_long.csv'
        if not os.path.exists(path): return None
        
        gt_df = pd.read_csv(path)
        gt_mean = gt_df[
            (gt_df['joint'] == 'r.kn.angle') & 
            (gt_df['plane'] == 'frontal')
        ].sort_values('gait_cycle')['condition1_avg'].values
        
        return resample(gt_mean, 101)
    except:
        return None

def main():
    print("Starting Full N=26 Frontal Analysis...")
    
    # 1. Derive Prior Template from Subject 03
    print("Deriving Prior Template from Subject 03...")
    _, s3_seg_signal = load_mp_frontal('3')
    if s3_seg_signal is None:
        print("CRITICAL: S3 Missing! Cannot run Prior-Based analysis.")
        return

    # Manual segmentation for S3 to get template
    # Pilot params: distance=20, prominence=0.02
    period_guess = 25
    peaks, _ = find_peaks(s3_seg_signal, distance=20, prominence=0.02)
    s3_candidates = []
    for p in peaks:
        if p + period_guess < len(s3_seg_signal):
            s3_candidates.append(resample(s3_seg_signal[p:p+period_guess], 101))
            
    if not s3_candidates:
        print("CRITICAL: S3 Segmentation failed.")
        return
        
    PRIOR_TEMPLATE = np.median(s3_candidates, axis=0)
    print("Prior Template Created.")
    
    # 2. Batch Process
    subject_dirs = sorted(glob.glob(f"{DATA_DIR}/[0-9]*"))
    subject_ids = [os.path.basename(d) for d in subject_dirs if os.path.basename(d).isdigit()]
    
    results = []
    
    # Prepare Grid Plot
    n_subs = len(subject_ids)
    cols = 5
    rows = (n_subs // cols) + 1
    plt.figure(figsize=(20, 4*rows))
    
    for i, sid in enumerate(subject_ids):
        # Load Data
        mp_valgus, mp_seg_signal = load_mp_frontal(sid)
        gt_curve = load_gt_frontal(sid)
        
        if mp_valgus is None or gt_curve is None:
            # print(f"Skipping S{sid} (Missing Data)")
            continue
            
        # Segment using PRIOR TEMPLATE
        try:
            starts = find_dtw_matches_euclidean(mp_seg_signal, PRIOR_TEMPLATE)
        except:
            continue
            
        cycles = []
        for s in starts:
            # Fixed window 25 frames
            w = mp_valgus[s : s+25]
            if len(w) > 10:
                cycles.append(resample(w, 101))
        
        if not cycles:
            continue
            
        mp_mean = np.mean(cycles, axis=0)
        
        # Compare (Align Means)
        mp_mean_centered = mp_mean - np.mean(mp_mean)
        gt_mean_centered = gt_curve - np.mean(gt_curve)
        
        corr, _ = pearsonr(mp_mean_centered, gt_mean_centered)
        rmse = np.sqrt(np.mean((mp_mean_centered - gt_mean_centered)**2))
        
        results.append({'Subject': sid, 'R': corr, 'RMSE': rmse})
        print(f"S{sid}: R={corr:.4f}")
        
        # Plot
        plt.subplot(rows, cols, i+1)
        plt.plot(mp_mean_centered, 'b', label='MP')
        plt.plot(gt_mean_centered, 'r--', label='GT')
        plt.title(f"S{sid} (R={corr:.2f})")
        if i == 0: plt.legend()
        
    plt.tight_layout()
    plt.savefig('/data/gait/2d_project/frontal_N26_grid.png')
    
    # Save Results
    df_res = pd.DataFrame(results)
    df_res.to_csv('/data/gait/2d_project/frontal_N26_results.csv', index=False)
    
    print("\n=== Final Results ===")
    print(f"N = {len(df_res)}")
    print(f"Mean Correlation: {df_res['R'].mean():.4f}")
    print(f"Mean RMSE: {df_res['RMSE'].mean():.4f}")
    print(f"R > 0.5 Count: {len(df_res[df_res['R']>0.5])}")

if __name__ == "__main__":
    main()
