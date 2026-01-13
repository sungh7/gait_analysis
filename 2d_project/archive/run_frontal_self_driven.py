import pandas as pd
import numpy as np
from scipy.signal import resample, find_peaks, correlate, savgol_filter
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
    norm_ba[norm_ba == 0] = 1e-6
    norm_bc[norm_bc == 0] = 1e-6
    cosine_angle = np.sum(ba*bc, axis=1) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def derive_frontal_self_template(signal, fps=24):
    # 1. Preprocessing
    # Frontal signal is noisy. Smooth it.
    try:
        sig_smooth = savgol_filter(signal, window_length=9, polyorder=2)
    except:
        sig_smooth = signal
        
    sig_detrend = sig_smooth - np.mean(sig_smooth)
    
    # 2. Autocorrelation with constraints
    corr = correlate(sig_detrend, sig_detrend, mode='full')
    corr = corr[len(corr)//2:]
    
    # Expect Walking Period: 0.8s to 1.4s
    min_lag = int(0.8 * fps) # ~19
    max_lag = int(1.4 * fps) # ~34
    
    # Find peaks in valid range
    peaks, props = find_peaks(corr, distance=min_lag)
    
    # Filter peaks by range
    valid_peaks = [p for p in peaks if min_lag <= p <= max_lag]
    
    if not valid_peaks:
        # Fallback to strongest peak in wider range? or just default 25
        # print("Auto-corr failed. Using default 25.")
        period = 25
    else:
        # Pick highest peak in range
        period = valid_peaks[np.argmax(corr[valid_peaks])]
        # print(f"Est Period: {period}")
        
    # 3. Candidate Segmentation
    # Flip signal if using Heel Y (Max Height = Min Y)
    # If we passed -HeelY, we want Peaks.
    # Inverted Distance Strategy
    inverted = -sig_smooth 
    # Use prominence relative to signal range
    sig_range = np.max(sig_smooth) - np.min(sig_smooth)
    prom = sig_range * 0.1 # 10% of ROM (Sagittal used fixed 5 deg, Frontal is smaller)
    if prom < 0.01: prom = 0.01
    
    rough_peaks, _ = find_peaks(sig_smooth, distance=int(period*0.7), prominence=prom)
    
    if len(rough_peaks) < 3:
        return None, None
        
    candidates = []
    for p in rough_peaks:
        if p + period < len(signal):
            # Extract period length
            c = signal[p : p+period]
            candidates.append(resample(c, 101))
            
    if not candidates: return None, None
    
    # median template
    template = np.median(candidates, axis=0)
    return template, candidates

def load_data(sid):
    # Load MP
    sid_str = str(int(sid))
    search_pattern = f'{DATA_DIR}/{sid_str}/{sid_str}-1_front_pose_*.csv'
    files = glob.glob(search_pattern)
    if not files:
        search_pattern = f'{DATA_DIR}/{int(sid)}/{int(sid)}-1_front_pose_*.csv'
        files = glob.glob(search_pattern)
    if not files: return None, None, None
    
    df = pd.read_csv(files[0])
    hip = df[['RIGHT_HIP_x', 'RIGHT_HIP_y']].values
    knee = df[['RIGHT_KNEE_x', 'RIGHT_KNEE_y']].values
    ankle = df[['RIGHT_ANKLE_x', 'RIGHT_ANKLE_y']].values
    heel_y = df['RIGHT_HEEL_y'].values
    
    raw_angle = calculate_angle(hip, knee, ankle)
    valgus_proxy = 180 - raw_angle
    
    # Signal for Seg: -Heel Y (Peaks = Height Maxima = Swing? No, Heel Strike is usually lowest point visually, so Highest Y value. -HeelY means peak is Heel Strike?)
    # Let's try Ankle X (Lateral). It is very sinusoidal.
    # Let's stick to -HeelY as S3 worked.
    seg_signal = -heel_y
    
    # Load GT
    gt_sid = f"{int(sid):02d}"
    gt_path = f'{GT_DIR}/S1_{gt_sid}_gait_long.csv'
    if not os.path.exists(gt_path): return valgus_proxy, seg_signal, None
    
    gt_df = pd.read_csv(gt_path)
    gt_mean = gt_df[
        (gt_df['joint'] == 'r.kn.angle') & 
        (gt_df['plane'] == 'frontal')
    ].sort_values('gait_cycle')['condition1_avg'].values
    gt_mean = resample(gt_mean, 101)
    
    return valgus_proxy, seg_signal, gt_mean

def main():
    print("Starting Self-Driven Frontal Analysis (Like Sagittal)...")
    
    subject_dirs = sorted(glob.glob(f"{DATA_DIR}/[0-9]*"))
    subject_ids = [os.path.basename(d) for d in subject_dirs if os.path.basename(d).isdigit()]
    
    results = []
    
    for sid in subject_ids:
        mp_val, mp_seg, gt_curve = load_data(sid)
        if mp_val is None: continue
        
        # 1. Derive Self-Template
        template, candidates = derive_frontal_self_template(mp_seg)
        
        if template is None:
            print(f"S{sid}: Self-Seg Failed.")
            continue
            
        # 2. Segmentation
        starts = find_dtw_matches_euclidean(mp_seg, template)
        
        cycles = []
        for s in starts:
            w = mp_val[s : s+len(template)] # Warp window?
            # Actually find_dtw returns indices. We should take roughly period length.
            # Or use warping path.
            # Simplified: take 25 frames (approx template len)
            w = mp_val[s : s+25]
            if len(w) > 10:
                cycles.append(resample(w, 101))
                
        if not cycles: continue
        
        mp_mean = np.mean(cycles, axis=0)
        
        if gt_curve is not None:
            # Compare
            mp_c = mp_mean - np.mean(mp_mean)
            gt_c = gt_curve - np.mean(gt_curve)
            corr, _ = pearsonr(mp_c, gt_c)
            print(f"S{sid}: R={corr:.4f}")
            results.append({'Subject': sid, 'R': corr})
        else:
            print(f"S{sid}: Extracted (No GT)")
            
    if results:
        df = pd.DataFrame(results)
        print(f"Mean R (Self-Driven): {df['R'].mean():.4f}")
        df.to_csv('frontal_self_driven_results.csv', index=False)

if __name__ == "__main__":
    main()
