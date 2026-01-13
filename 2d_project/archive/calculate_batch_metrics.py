#!/usr/bin/env python3
import glob
import json
import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.stats import pearsonr
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import derive_self_template, find_dtw_matches_euclidean
from tqdm import tqdm
import os
import traceback

DATA_DIR = "/data/gait/data"
OUTPUT_DIR = "/data/gait/2d_project/research_metrics"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def load_gt_param(sid):
    # Load V3D Mean Cycle and Stride Count
    sid_str = f"{int(sid):02d}"
    gt_csv = f"/data/gait/data/processed_new/S1_{sid_str}_gait_long.csv"
    gt_json = f"/data/gait/data/processed_new/S1_{sid_str}_info.json"
    
    if not os.path.exists(gt_csv) or not os.path.exists(gt_json):
        return None, None
        
    # Read CSV for Mean Cycle
    try:
        df = pd.read_csv(gt_csv)
        # Filter for Right Knee Sagittal
        subset = df[
            (df['joint'] == 'r.kn.angle') & 
            (df['plane'] == 'sagittal')
        ].sort_values('gait_cycle')
        
        if subset.empty: return None, None
        
        gt_mean_cycle = subset['condition1_avg'].values
        # Ensure it is 101 points? 
        if len(gt_mean_cycle) != 101:
            gt_mean_cycle = resample(gt_mean_cycle, 101)
            
    except: return None, None

    # Read JSON for Stride Count
    try:
        with open(gt_json, 'r') as f:
            info = json.load(f)
            gt_count = info['demographics']['right_strides']
    except: gt_count = 0
    
    return gt_mean_cycle, gt_count

def calculate_rmse(x, y):
    return np.sqrt(np.mean((x - y)**2))

def normalize(v):
    return (v - np.mean(v)) / np.std(v)

def main():
    print("Starting Quantitative Analysis (N=26)...")
    extractor = MediaPipeSagittalExtractor()
    results = []
    
    subject_dirs = sorted(glob.glob(f"{DATA_DIR}/[0-9]*"))
    subject_ids = [os.path.basename(d) for d in subject_dirs if os.path.basename(d).isdigit()]
    
    for sid in tqdm(subject_ids):
        try:
            # 1. Load GT
            gt_mean, gt_count = load_gt_param(sid)
            if gt_mean is None: continue
            
            # 2. Load MP Data (Video)
            video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
            if not os.path.exists(video_path): continue
            
            # Extract
            landmarks, _ = extractor.extract_pose_landmarks(video_path) # Full video
            angles = extractor.calculate_joint_angles(landmarks)
            mp_signal = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
            
            # 3. Auto-Segmentation (Proposed Method)
            template, candidates = derive_self_template(mp_signal)
            if template is None: continue
            
            starts = find_dtw_matches_euclidean(mp_signal, template)
            
            # 4. Extract Cycles & Quality Control (QC)
            extracted_cycles = []
            for s in starts:
                window = mp_signal[s : s+35] # Assuming ~35 frames
                if len(window) < 10: continue
                resampled = resample(window, 101)
                extracted_cycles.append(resampled)
            
            if not extracted_cycles: continue
            
            # QC Filter: Remove Standing/Shuffling (Low ROM) and Bad Shapes (Low Corr)
            clean_cycles = []
            self_median = np.median(extracted_cycles, axis=0) # Robust reference
            
            for cyc in extracted_cycles:
                cyc_rom = np.max(cyc) - np.min(cyc)
                cyc_corr, _ = pearsonr(cyc, self_median)
                
                # Criteria: ROM > 30 (Walking) AND Corr > 0.8 (Consistent Shape)
                if cyc_rom > 30 and cyc_corr > 0.8:
                    clean_cycles.append(cyc)
            
            # Fallback if aggressive filtering removed everything (e.g. pathological/stiff gait)
            if len(clean_cycles) < 3:
                clean_cycles = extracted_cycles
            
            mp_mean = np.mean(clean_cycles, axis=0)
            mp_count = len(extracted_cycles)
            mp_count_clean = len(clean_cycles)
            
            # 5. Metrics
            # A. Pearson Correlation
            corr, _ = pearsonr(mp_mean, gt_mean)
            
            # B. RMSE (Absolute) - might be high due to offset?
            # V3D and MP often have offset differences.
            rmse_abs = calculate_rmse(mp_mean, gt_mean)
            
            # C. RMSE (Shape) - Z-score normalized
            rmse_shape = calculate_rmse(normalize(mp_mean), normalize(gt_mean))
            
            # D. Scalar Parameters for Bland-Altman
            gt_rom = np.max(gt_mean) - np.min(gt_mean)
            mp_rom = np.max(mp_mean) - np.min(mp_mean)
            
            res_dict = {
                'subject': sid,
                'mp_count': mp_count,
                'gt_count': gt_count,
                'correlation': corr,
                'rmse_abs': rmse_abs,
                'rmse_shape': rmse_shape,
                'gt_rom': gt_rom,
                'mp_rom': mp_rom
            }
            results.append(res_dict)
            
            # Incremental Save
            df_inc = pd.DataFrame([res_dict])
            header = not os.path.exists(f"{OUTPUT_DIR}/final_benchmarks.csv")
            df_inc.to_csv(f"{OUTPUT_DIR}/final_benchmarks.csv", mode='a', header=header, index=False)
            
            print(f"Subject {sid}: r={corr:.4f}, RMSE={rmse_shape:.4f}")
            
        except Exception as e:
            print(f"Error {sid}: {e}")
            traceback.print_exc()
            
    # Final Summary (Read from file to be safe)
    if os.path.exists(f"{OUTPUT_DIR}/final_benchmarks.csv"):
        df_res = pd.read_csv(f"{OUTPUT_DIR}/final_benchmarks.csv")
        print("\n=== Research Benchmark Results ===")
        print(f"Subjects Processed: {len(df_res)}")
        print(f"Avg Correlation: {df_res['correlation'].mean():.4f} ± {df_res['correlation'].std():.4f}")
        print(f"Avg RMSE (Shape): {df_res['rmse_shape'].mean():.4f}")

if __name__ == "__main__":
    main()
