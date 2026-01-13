import pandas as pd
import numpy as np
from scipy.signal import resample, find_peaks
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from self_driven_segmentation import find_dtw_matches_euclidean

def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.sum(ba*bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def process_subject(sid):
    try:
        # Load MP
        sid_str = str(int(sid))
        csv_path = f'/data/gait/data/{sid_str}/{sid_str}-1_front_pose_fps23.csv'
        df = pd.read_csv(csv_path)
        
        hip = df[['RIGHT_HIP_x', 'RIGHT_HIP_y']].values
        knee = df[['RIGHT_KNEE_x', 'RIGHT_KNEE_y']].values
        ankle = df[['RIGHT_ANKLE_x', 'RIGHT_ANKLE_y']].values
        heel_y = df['RIGHT_HEEL_y'].values
        
        # Valgus Proxy (Angle - 180)
        # Sign Inversion applied: -(Angle - 180) = 180 - Angle.
        # This matches GT Adduction sign convention (usually).
        raw_angle = calculate_angle(hip, knee, ankle)
        # valgus_proxy = raw_angle - 180 
        valgus_proxy_inverted = -(raw_angle - 180) 
        
        # Segmentation Signal: Heel Y (Inverted)
        signal_for_seg = -heel_y 

        # Manual Segmentation
        period_guess = 25
        peaks, _ = find_peaks(signal_for_seg, distance=20, prominence=0.02)
        
        candidates = []
        for p in peaks:
            if p + period_guess < len(signal_for_seg):
                c = signal_for_seg[p : p+period_guess]
                candidates.append(resample(c, 101))
                
        if not candidates: return None, None
        
        template = np.median(candidates, axis=0)
        starts = find_dtw_matches_euclidean(signal_for_seg, template)
        
        cycles = []
        for s in starts:
            w = valgus_proxy_inverted[s:s+period_guess]
            if len(w) > 10:
                cycles.append(resample(w, 101))
                
        if not cycles: return None, None
        mp_mean = np.mean(cycles, axis=0)
        
        # Load GT
        gt_sid = f"{int(sid):02d}"
        gt_df = pd.read_csv(f'/data/gait/data/processed_new/S1_{gt_sid}_gait_long.csv')
        gt_mean = gt_df[
            (gt_df['joint'] == 'r.kn.angle') & 
            (gt_df['plane'] == 'frontal')
        ].sort_values('gait_cycle')['condition1_avg'].values
        gt_mean = resample(gt_mean, 101)
        
        # Center both
        mp_mean = mp_mean - np.mean(mp_mean)
        gt_mean = gt_mean - np.mean(gt_mean)
        
        return mp_mean, gt_mean
        
    except Exception as e:
        print(f"Error {sid}: {e}")
        return None, None

def main():
    subjects = ['1', '2', '3']
    results = []
    
    plt.figure(figsize=(15, 5))
    
    for i, sid in enumerate(subjects):
        mp, gt = process_subject(sid)
        if mp is not None:
            corr, _ = pearsonr(mp, gt)
            rmse = np.sqrt(np.mean((mp - gt)**2))
            
            print(f"Subject {sid}: r={corr:.4f}, RMSE={rmse:.4f}")
            results.append(corr)
            
            plt.subplot(1, 3, i+1)
            plt.plot(mp, label='MP (Valgus Proxy)')
            plt.plot(gt, label='GT (Adduction)')
            plt.title(f"S{sid} Frontal (r={corr:.2f})")
            plt.legend()
            
    plt.tight_layout()
    plt.savefig('/data/gait/2d_project/frontal_multi_pilot.png')
    print(f"Mean Correlation (N={len(results)}): {np.mean(results):.4f}")

if __name__ == "__main__":
    main()
