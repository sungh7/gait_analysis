
import pandas as pd
import numpy as np
import os
import glob
from scipy.stats import pearsonr, ttest_rel
from scipy.signal import resample
import matplotlib.pyplot as plt

# --- PATHS ---
DATA_DIR = "/data/gait/data"
PROCESSED_DIR = "/data/gait/data/processed_new"
OUTPUT_DIR = "/data/gait/2d_project/rebuttal_experiments"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# --- UTILS ---

def calculate_icc(gt, pred):
    # simple ICC(2,1) approx or just correlation/agreement
    # For N paired samples:
    # We will use the Pingouin package if available, else simple implementation
    # Let's stick to simple statistics for now: Pearson r, Bias, RMSE
    return pearsonr(gt, pred)[0]

def load_data(n_subjects=30):
    paired_data = []
    
    for i in range(1, n_subjects+1):
        sid_str = f"{i}" # 1, 2, ...
        # Load Vicon (GT)
        # Note: filenames are inconsistent sometimes. S1_01 ...
        vicon_path = f"{PROCESSED_DIR}/S1_{i:02d}_gait_long.csv"
        
        # Load MediaPipe (Self-segmented or just raw cycles? We need matched cycles.)
        # Since we are validating *Kinematics*, we should use the "Average Cycle" 
        # extracted by the AT-DTW method or just the "Average Cycle" available in previous results?
        # A simpler way: Use the Vicon 'condition1_avg' and the MediaPipe 'Self-Template' (median).
        
        if not os.path.exists(vicon_path): continue
        
        try:
            # GT
            df_gt = pd.read_csv(vicon_path)
            subset_gt = df_gt[(df_gt['joint'] == 'r.kn.angle') & (df_gt['plane'] == 'sagittal')]
            if subset_gt.empty: continue
            gt_curve = resample(subset_gt['condition1_avg'].values, 101)
            
            # MP (Directly from video? or cache?)
            # To be robust, let's look for cached MP templates if they exist, or re-extract?
            # Re-extracting is slow.
            # Let's check if we have MP cycles saved.
            # actually, 'self_driven_segmentation.py' generates them but doesn't save to a nice DB.
            # Let's use the 'validation_results' from previous run if possible?
            # Or just Quick Process S1..S30.
            
            # FAST TRACK: Use pre-calculated cycles from GAVD validation script location? No that's GAVD.
            # We need standard healthy subjects (1-26).
            # Look at 'dtw_segment_results'.
            pass # We need to run extraction.
            
        except: continue
        
    return paired_data

# Since we need to extract MP data for N=21, let's import the extractor
import sys
sys.path.append("/data/gait/2d_project")
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import derive_self_template

def extract_and_match(max_subjects=30):
    results = []
    
    extractor = MediaPipeSagittalExtractor()
    
    print("Extracting and Matching Data...")
    
    subject_dirs = sorted(glob.glob(f"{DATA_DIR}/[0-9]*"))
    
    for subj_dir in subject_dirs:
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
            # To save time, check if we can reuse anything? No, just run it. 
            # It takes ~10s per video. 30 subjects = 5 mins. Acceptable.
            lm, _ = extractor.extract_pose_landmarks(vid_path)
            ang = extractor.calculate_joint_angles(lm)
            sig = ang['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
            
            # Derive Template (Auto)
            mp_template, _ = derive_self_template(sig)
            if mp_template is None: continue
            
            # We now have GT curve and MP curve (Self-Template)
            results.append({
                'sid': sid_int,
                'gt': gt_curve,
                'mp': mp_template
            })
            print(f"Processed S{sid_int}")
            
        except Exception as e:
            print(f"Error S{sid}: {e}")
            
    return results

def run_calibration_experiment():
    data = extract_and_match()
    
    if not data:
        print("No paired data found.")
        return

    # Metrics Lists
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
        # Calibrated MP = MP - (Mean_MP - Mean_GT_Reference)
        # Note: In real life we don't have GT. We align to "Neutral Standing" or "Population Normal".
        # Here we verify the "Feasibility" by aligning to GT mean (Best possible calibration).
        # Or align to a fixed offset?
        # The paper says "aligning subject's mean ROM to a reference".
        # Let's align MP mean to GT mean (perfect offset correction).
        mp_cal = mp - np.mean(mp) + np.mean(gt)
        
        bias_c = np.mean(mp_cal) - np.mean(gt) # Should be 0
        rmse_c = np.sqrt(np.mean((mp_cal - gt)**2))
        r_c = pearsonr(gt, mp_cal)[0] # Should be same as r_pre
        
        bias_post.append(bias_c)
        rmse_post.append(rmse_c)
        r_post.append(r_c)
        
    # Stats
    df_res = pd.DataFrame({
        'Bias_Pre': bias_pre, 'Bias_Post': bias_post,
        'RMSE_Pre': rmse_pre, 'RMSE_Post': rmse_post,
        'Corr_Pre': r_pre, 'Corr_Post': r_post
    })
    
    print("\n--- Calibration Results (N={}) ---".format(len(df_res)))
    print(df_res.describe())
    
    # Save
    df_res.to_csv(f"{OUTPUT_DIR}/calibration_experiment_results.csv", index=False)
    
    # Plot RMSE Comparison
    plt.figure(figsize=(6, 6))
    plt.boxplot([df_res['RMSE_Pre'], df_res['RMSE_Post']], labels=['Raw (Pre)', 'Calibrated (Post)'])
    plt.title("Impact of Offset Calibration on RMSE")
    plt.ylabel("RMSE (Degrees)")
    plt.savefig(f"{OUTPUT_DIR}/calibration_rmse_boxplot.png")
    print("Saved plots.")

import sys
if __name__ == "__main__":
    run_calibration_experiment()
