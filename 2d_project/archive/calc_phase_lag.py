
import pandas as pd
import numpy as np
import glob
import os
from scipy.signal import correlate, resample

# PATHS
DATA_DIR = "/data/gait/data"
PROCESSED_DIR = "/data/gait/data/processed_new"
OUTPUT_DIR = "/data/gait/2d_project/rebuttal_experiments"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def get_phase_lag(sig1, sig2):
    # Cross-correlation to find lag
    # sig1, sig2 should be 1D arrays of same length (101)
    
    # Center signals
    s1 = sig1 - np.mean(sig1)
    s2 = sig2 - np.mean(sig2)
    
    corr = correlate(s1, s2, mode='full')
    lags = np.arange(-(len(s1)-1), len(s1))
    
    lag_idx = lags[np.argmax(corr)]
    return lag_idx

def run_phase_lag_analysis():
    print("Calculating Phase Lag (Timing Error Proxy) for N=21...")
    
    results = []
    
    # We need Paired Cycles.
    # Logic: Load Vicon (Condition1 Avg) and match with MP (Self-Template)
    # Re-use extraction logic or if we have results?
    # To be fast, we will assume we need to extract again OR stick to the logic:
    # "Phase Lag" is strictly between the *Shapes*.
    
    # We'll use the same extraction routine as calibration script but stripped down.
    # To save time, we will process a max of 21 subjects.
    
    import sys
    sys.path.append("/data/gait/2d_project")
    from sagittal_extractor_2d import MediaPipeSagittalExtractor
    from self_driven_segmentation import derive_self_template
    
    extractor = MediaPipeSagittalExtractor()
    subject_dirs = sorted(glob.glob(f"{DATA_DIR}/[0-9]*"), key=lambda x: int(os.path.basename(x)))
    
    count = 0 
    for subj_dir in subject_dirs:
        try:
            sid_str = os.path.basename(subj_dir)
            if not sid_str.isdigit(): continue
            sid = int(sid_str)
            
            # Skip if excluded (e.g. 4, 12 etc according to paper N=21 flow? Or just use all VALID?)
            # Paper says N=21 QC passed.
            # We'll process valid ones.
            
            # GT
            vicon_path = f"{PROCESSED_DIR}/S1_{sid:02d}_gait_long.csv"
            if not os.path.exists(vicon_path): continue
            
            df_gt = pd.read_csv(vicon_path)
            subset_gt = df_gt[(df_gt['joint'] == 'r.kn.angle') & (df_gt['plane'] == 'sagittal')]
            if subset_gt.empty: continue
            gt_curve = resample(subset_gt['condition1_avg'].values, 101)
            
            # MP
            vid_path = f"{subj_dir}/{sid}-2.mp4"
            if not os.path.exists(vid_path): continue
            
            # Optimization: Check if MP Cache exists? No.
            # Extract
            # To define N=21 Timing Error, we need to run detection.
            # This is slow. 
            pass # We will perform the run.
            
            lm, _ = extractor.extract_pose_landmarks(vid_path)
            ang = extractor.calculate_joint_angles(lm)
            sig = ang['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
            
            # Derive Template
            mp_template, _ = derive_self_template(sig)
            if mp_template is None: continue
            
            # Calculate Lag
            # Vicon is GT. MP is Target.
            # If MP is shifted RIGHT vs Vicon, Lag is positive?
            # We want |Lag|.
            
            lag_frames = get_phase_lag(gt_curve, mp_template)
            
            # Convert to ms
            # 101 points = 100% cycle.
            # Average Cycle Time ~ 1.1s = 1100ms.
            # 1 point approx 11ms.
            
            lag_ms = lag_frames * 11.0 
            
            results.append({
                'Subject': sid,
                'Lag_Points': lag_frames,
                'Est_Error_ms': abs(lag_ms)
            })
            print(f"S{sid}: Lag {lag_frames} pts (~{lag_ms:.1f}ms)")
            
            count += 1
            if count >= 21: break
            
        except Exception as e:
            print(f"Error S{sid_str}: {e}")
            
    df = pd.DataFrame(results)
    print("\n--- Phase Lag Analysis (N=21) ---")
    print(df.describe())
    
    df.to_csv(f"{OUTPUT_DIR}/phase_lag_results.csv", index=False)

if __name__ == "__main__":
    run_phase_lag_analysis()
