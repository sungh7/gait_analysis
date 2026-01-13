import pandas as pd
import numpy as np
from scipy.signal import resample, find_peaks
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from self_driven_segmentation import find_dtw_matches_euclidean

def calculate_angle(a, b, c):
    # a, b, c are (x,y) arrays
    ba = a - b
    bc = c - b
    cosine_angle = np.sum(ba*bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def main():
    print("Processing Frontal Pilot (S1) with Manual Parameters...")
    
    # 1. Load MP Frontal Data
    try:
        df = pd.read_csv('/data/gait/data/1/1-1_front_pose_fps23.csv')
        
        hip = df[['RIGHT_HIP_x', 'RIGHT_HIP_y']].values
        knee = df[['RIGHT_KNEE_x', 'RIGHT_KNEE_y']].values
        ankle = df[['RIGHT_ANKLE_x', 'RIGHT_ANKLE_y']].values
        heel_y = df['RIGHT_HEEL_y'].values
        
        # Valgus Proxy
        # Normal straight = 180.
        raw_angle = calculate_angle(hip, knee, ankle)
        valgus_proxy = raw_angle - 180 # Deviation from straight
        
        # Segmentation Signal: Heel Y
        # Frontal view heel might not have clear peaks?
        # Let's try Ankle X (Lateral movement) which is very periodic in frontal walk.
        # Actually, let's stick to Heel Y but force simple peak detection.
        signal_for_seg = -heel_y # Invert so peaks = ground contact/minima
        
    except Exception as e:
        print(f"Failed to load MP: {e}")
        return

    # 2. Segment (Manual Override)
    # Force Period = 25 frames (~1 sec at 23fps)
    period_guess = 25
    candidates = []
    
    # Simple Peak Detection
    peaks, _ = find_peaks(signal_for_seg, distance=20, prominence=0.02)
    
    # Create simple median template from these peaks
    for p in peaks:
        if p + period_guess < len(signal_for_seg):
            c = signal_for_seg[p : p+period_guess]
            candidates.append(resample(c, 101))
            
    if not candidates:
        print("No candidates found via manual peak detection.")
        return
        
    template = np.median(candidates, axis=0) # Simple median
    
    # DTW Search
    starts = find_dtw_matches_euclidean(signal_for_seg, template)
    
    cycles = []
    for s in starts:
        w = valgus_proxy[s:s+period_guess] # Use period guess
        if len(w) > 10:
            cycles.append(resample(w, 101))
            
    if not cycles:
        print("No cycles found after DTW.")
        return
        
    mp_mean = np.mean(cycles, axis=0)
    
    # 3. Load GT Frontal Data
    try:
        gt_df = pd.read_csv('/data/gait/data/processed_new/S1_01_gait_long.csv')
        gt_mean = gt_df[
            (gt_df['joint'] == 'r.kn.angle') & 
            (gt_df['plane'] == 'frontal')
        ].sort_values('gait_cycle')['condition1_avg'].values
        
        gt_mean = resample(gt_mean, 101)
        
        # Vicon Frontal is often Adduction/Abduction.
        # Check sign? Usually Adduction is positive?
        # MP Valgus Proxy (Angle - 180).
        # We might need to invert or offset.
        # Let's align means.
        
        mp_mean = mp_mean - np.mean(mp_mean)
        gt_mean = gt_mean - np.mean(gt_mean)
        
    except Exception as e:
        print(f"Failed to load GT: {e}")
        return
        
    # 4. Compare
    corr, _ = pearsonr(mp_mean, gt_mean)
    rmse = np.sqrt(np.mean((mp_mean - gt_mean)**2))
    
    print(f"Frontal Pilot Results (Manual Seg):")
    print(f"Correlation: {corr:.4f}")
    print(f"RMSE (Shape): {rmse:.4f}")
    
    # Save Plot
    plt.figure()
    plt.plot(mp_mean, label='MP Valgus Proxy (Centered)')
    plt.plot(gt_mean, label='GT Adduction (Centered)')
    plt.legend()
    plt.title(f"Frontal Pilot (r={corr:.2f})")
    plt.savefig('/data/gait/2d_project/frontal_pilot_check.png')
    print("Saved plot.")

if __name__ == "__main__":
    main()
