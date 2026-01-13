
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import glob
from tqdm import tqdm
from scipy.signal import resample
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import derive_self_template, find_dtw_matches_euclidean

DATA_DIR = "/data/gait/data"
OUTPUT_DIR = "/data/gait/2d_project/research_metrics"

def load_gt_signal(sid):
    # Load standardized 101-point cycle
    gt_csv = f"/data/gait/data/processed_new/S1_{int(sid):02d}_gait_long.csv"
    if not os.path.exists(gt_csv): return None
    try:
        df = pd.read_csv(gt_csv)
        subset = df[(df['joint'] == 'r.kn.angle') & (df['plane'] == 'sagittal')]
        if subset.empty: return None
        return resample(subset['condition1_avg'].values, 101)
    except: return None

def calculate_icc(gt, mp):
    # Simplified ICC(2,1) proxy or just concordance
    # Here we stick to RMSE and Bias for clarity
    bias = np.mean(mp - gt)
    rmse = np.sqrt(np.mean((mp - gt)**2))
    return bias, rmse

def main():
    print("Simulating Individual Calibration Effect...")
    extractor = MediaPipeSagittalExtractor()
    results = []
    
    # Load previously validated subjects list from file or just scan
    subject_dirs = sorted(glob.glob(f"{DATA_DIR}/[0-9]*"))
    subject_ids = [os.path.basename(d) for d in subject_dirs if os.path.basename(d).isdigit()]
    
    for sid in tqdm(subject_ids):
        try:
            # GT
            gt_curve = load_gt_signal(sid)
            if gt_curve is None: continue
            
            # MP
            video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
            if not os.path.exists(video_path): continue
            
            landmarks, _ = extractor.extract_pose_landmarks(video_path)
            if not landmarks: continue
            angles = extractor.calculate_joint_angles(landmarks)
            mp_signal = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
            
            # Segment
            template, _ = derive_self_template(mp_signal)
            if template is None: continue
            starts = find_dtw_matches_euclidean(mp_signal, template)
            
            cycles = []
            for s in starts:
                c = mp_signal[s:s+35]
                if len(c) > 10: cycles.append(resample(c, 101))
            
            if len(cycles) < 3: continue
            mp_curve_raw = np.mean(cycles, axis=0) # Raw extracted
            
            # --- Calibration Simulation ---
            
            # 1. Raw Error
            bias_raw, rmse_raw = calculate_icc(gt_curve, mp_curve_raw)
            
            # 2. Offset Calibration (Static Simulation)
            # Shift mean of MP to match Mean of GT (Simulates perfect static calib)
            offset = np.mean(gt_curve) - np.mean(mp_curve_raw)
            mp_curve_offset = mp_curve_raw + offset
            _, rmse_offset = calculate_icc(gt_curve, mp_curve_offset)
            
            # 3. Scaling Calibration (Functional Simulation)
            # Match ROM (Simulates perfect squat/ROM check)
            rom_gt = np.max(gt_curve) - np.min(gt_curve)
            rom_mp = np.max(mp_curve_raw) - np.min(mp_curve_raw)
            scale_factor = rom_gt / rom_mp if rom_mp > 0 else 1.0
            
            # Apply Scale centered around mean
            mp_curve_scaled = (mp_curve_raw - np.mean(mp_curve_raw)) * scale_factor + np.mean(gt_curve)
            _, rmse_scaled = calculate_icc(gt_curve, mp_curve_scaled)
            
            results.append({
                'Subject': sid,
                'RMSE_Raw': rmse_raw,
                'RMSE_Offset': rmse_offset,
                'RMSE_FullyCalib': rmse_scaled,
                'Improvement_Pct': (rmse_raw - rmse_scaled) / rmse_raw * 100
            })
            
        except Exception as e:
            # print(e)
            pass

    if results:
        df = pd.DataFrame(results)
        print("\n--- Calibration Simulation Results ---")
        print(f"N = {len(df)}")
        print(f"Avg RMSE (Raw): {df['RMSE_Raw'].mean():.2f}")
        print(f"Avg RMSE (Offset Corrected): {df['RMSE_Offset'].mean():.2f}")
        print(f"Avg RMSE (Fully Calibrated): {df['RMSE_FullyCalib'].mean():.2f}")
        print(f"Avg Improvement: {df['Improvement_Pct'].mean():.1f}%")
        
        df.to_csv(f"{OUTPUT_DIR}/calibration_simulation.csv", index=False)
        print(f"Saved to {OUTPUT_DIR}/calibration_simulation.csv")

if __name__ == "__main__":
    main()
