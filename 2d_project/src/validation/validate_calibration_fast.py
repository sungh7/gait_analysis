
import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import pearsonr, ttest_rel
from scipy.signal import resample
import matplotlib.pyplot as plt
import sys
sys.path.append("/data/gait/2d_project")

# --- PATHS ---
DATA_DIR = "/data/gait/data"
PROCESSED_DIR = "/data/gait/data/processed_new"
OUTPUT_DIR = "/data/gait/2d_project/rebuttal_experiments"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import derive_self_template

def extract_and_match(max_subjects=30):
    results = []
    extractor = MediaPipeSagittalExtractor()
    print(f"Extracting and Matching Data (Max {max_subjects})...")
    
    subject_dirs = sorted(glob.glob(f"{DATA_DIR}/[0-9]*"), key=lambda x: int(os.path.basename(x)))
    # Process all available
    # subject_dirs = [d for d in subject_dirs if os.path.basename(d) == '1']
    
    count = 0
    for subj_dir in subject_dirs:
        if count >= max_subjects: break
        try:
            sid = os.path.basename(subj_dir)
            if not sid.isdigit(): continue
            sid_int = int(sid)
            
            # GT
            vicon_path = f"{PROCESSED_DIR}/S1_{sid_int:02d}_gait_long.csv"
            if not os.path.exists(vicon_path): continue
            
            df_gt = pd.read_csv(vicon_path)
            subset_gt = df_gt[(df_gt['joint'] == 'r.kn.angle') & (df_gt['plane'] == 'sagittal')]
            if subset_gt.empty: continue
            gt_curve = resample(subset_gt['condition1_avg'].values, 101)
            
            # MP
            vid_path = f"{subj_dir}/{sid}-2.mp4"
            if not os.path.exists(vid_path): continue
            
            # Extract
            lm, _ = extractor.extract_pose_landmarks(vid_path)
            ang = extractor.calculate_joint_angles(lm)
            sig = ang['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
            
            # Derive Template (Auto)
            mp_template, _ = derive_self_template(sig)
            if mp_template is None: continue
            
            results.append({
                'sid': sid_int,
                'gt': gt_curve,
                'mp': mp_template
            })
            print(f"Processed S{sid_int}")
            count += 1
            
        except Exception as e:
            print(f"Error S{sid}: {e}")
            
    return results

def run_calibration_experiment():
    data = extract_and_match(max_subjects=30)
    
    if not data:
        print("No paired data found.")
        return

    bias_pre, bias_post = [], []
    rmse_pre, rmse_post = [], []
    r_pre, r_post = [], []
    
    for item in data:
        gt = item['gt']
        mp = item['mp']
        
        # 1. Pre-Calibration (Raw)
        bias = np.mean(mp) - np.mean(gt)
        rmse = np.sqrt(np.mean((mp - gt)**2))
        r = pearsonr(gt, mp)[0]
        
        bias_pre.append(bias)
        rmse_pre.append(rmse)
        r_pre.append(r)
        
        # 2. Apply Calibration (Mean Offset)
        mp_cal = mp - np.mean(mp) + np.mean(gt)
        
        bias_c = np.mean(mp_cal) - np.mean(gt) 
        rmse_c = np.sqrt(np.mean((mp_cal - gt)**2))
        r_c = pearsonr(gt, mp_cal)[0] 
        
        bias_post.append(bias_c)
        rmse_post.append(rmse_c)
        r_post.append(r_c)
        
    df_res = pd.DataFrame({
        'Bias_Pre': bias_pre, 'Bias_Post': bias_post,
        'RMSE_Pre': rmse_pre, 'RMSE_Post': rmse_post,
        'Corr': r_pre
    })
    
    print("\n--- Calibration Results (N={}) ---".format(len(df_res)))
    print(df_res.describe())
    
    # Save
    df_res.to_csv(f"{OUTPUT_DIR}/calibration_fast_results.csv", index=False)

if __name__ == "__main__":
    run_calibration_experiment()
