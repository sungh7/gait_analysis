#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import derive_self_template, find_dtw_matches_euclidean
from scipy.signal import resample
from scipy.stats import pearsonr
import os
import glob
from tqdm import tqdm

DATA_DIR = "/data/gait/data"
OUTPUT_DIR = "/data/gait/2d_project/calibration_test"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def load_gt_param(sid):
    try:
        path = f"{DATA_DIR}/processed_new/S1_{sid}_gait_long.csv"
        if not os.path.exists(path): return None
        df = pd.read_csv(path)
        subset = df[(df['joint'] == 'r.kn.angle') & (df['plane'] == 'sagittal')].sort_values('gait_cycle')
        return subset['condition1_avg'].values
    except: return None

def main():
    extractor = MediaPipeSagittalExtractor()
    subject_dirs = sorted(glob.glob(f"{DATA_DIR}/[0-9]*"))
    subject_ids = [os.path.basename(d) for d in subject_dirs if os.path.basename(d).isdigit()]
    
    # Select a few subjects: Good (03, 26) and Poor (24, 27)
    test_ids = ['03', '26', '24', '27']
    
    results = []
    
    for sid in test_ids:
        try:
            # 1. Load GT
            gt_mean = load_gt_param(sid)
            if gt_mean is None: 
                print(f"Skipping {sid}: No GT")
                continue
            
            # 2. Load MP
            video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
            if not os.path.exists(video_path): 
                # Try int folder
                video_path = f"{DATA_DIR}/{int(sid)}/{int(sid)}-2.mp4"
                
            landmarks, _ = extractor.extract_pose_landmarks(video_path)
            angles = extractor.calculate_joint_angles(landmarks)
            mp_signal = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
            
            # 3. Segment (Auto)
            template, candidates = derive_self_template(mp_signal)
            starts = find_dtw_matches_euclidean(mp_signal, template)
            
            extracted_cycles = []
            for s in starts:
                window = mp_signal[s : s+35]
                if len(window) < 10: continue
                # We want just the mean cycle
                resampled = resample(window, 101)
                extracted_cycles.append(resampled)
                
            if not extracted_cycles: continue
            
            mp_mean_raw = np.mean(extracted_cycles, axis=0)
            
            # --- Calibration Logic ---
            # Method: Offset Correction
            # Assumption: The 'shape' is correct, but the 'DC offset' is wrong.
            # Strategy: Align the Peak Extension (Minimum Flexion) of MP to GT (or a fixed reference, e.g. 0 deg)
            # Since we have GT for this test, let's see if aligning the MEANS or MINS works.
            
            # Calibration 1: Align Minimums (Assume full extension is consistent)
            offset_min = np.min(gt_mean) - np.min(mp_mean_raw)
            mp_calibrated_min = mp_mean_raw + offset_min
            
            # Calibration 2: Align Means (Remove DC Bias)
            offset_mean = np.mean(gt_mean) - np.mean(mp_mean_raw)
            mp_calibrated_mean = mp_mean_raw + offset_mean
            
            # Metrics
            rmse_raw = np.sqrt(np.mean((mp_mean_raw - gt_mean)**2))
            rmse_cal_min = np.sqrt(np.mean((mp_calibrated_min - gt_mean)**2))
            rmse_cal_mean = np.sqrt(np.mean((mp_calibrated_mean - gt_mean)**2))
            
            results.append({
                'subject': sid,
                'rmse_raw': rmse_raw,
                'rmse_cal_min': rmse_cal_min,
                'rmse_cal_mean': rmse_cal_mean,
                'offset_min': offset_min,
                'offset_mean': offset_mean
            })
            
            # Plot
            plt.figure(figsize=(10, 4))
            plt.plot(gt_mean, 'k-', linewidth=2, label='Ground Truth')
            plt.plot(mp_mean_raw, 'r--', label=f'Raw MP (RMSE={rmse_raw:.1f})')
            plt.plot(mp_calibrated_mean, 'g-.', label=f'Calibrated (Mean Align) (RMSE={rmse_cal_mean:.1f})')
            plt.title(f"Subject {sid}: Calibration Effect")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{OUTPUT_DIR}/calibration_test_{sid}.png")
            plt.close()
            
        except Exception as e:
            print(f"Error {sid}: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== Calibration Test Results ===")
    df_res = pd.DataFrame(results)
    print(df_res)
    df_res.to_csv(f"{OUTPUT_DIR}/calibration_results.csv", index=False)

if __name__ == "__main__":
    main()
