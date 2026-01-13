#!/usr/bin/env python3
import numpy as np
import pandas as pd
import glob
import os
from scipy.signal import resample
from scipy.stats import pearsonr

from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import derive_self_template, find_dtw_matches_euclidean

DATA_DIR = "/data/gait/data"
GT_DIR = "/data/gait/data/processed_new"

def load_data_pair(sid):
    # Load GT
    try:
        gt_csv = glob.glob(f"{GT_DIR}/S1_{int(sid):02d}_gait_long.csv")[0]
        df = pd.read_csv(gt_csv)
        gt_mean = df[(df['joint'] == 'r.kn.angle') & (df['plane'] == 'sagittal')].sort_values('gait_cycle')['condition1_avg'].values
        gt_mean = resample(gt_mean, 101)
    except: return None, None
    
    # Load MP
    try:
        video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
        if not os.path.exists(video_path): return None, None
        extractor = MediaPipeSagittalExtractor()
        landmarks, _ = extractor.extract_pose_landmarks(video_path)
        angles = extractor.calculate_joint_angles(landmarks)
        mp_signal = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
        
        # Segment w/ Auto-Template + Phase Alignment + QC
        template, candidates = derive_self_template(mp_signal)
        if template is None: return None, None
        starts = find_dtw_matches_euclidean(mp_signal, template)
        
        cycles = []
        for s in starts:
            w = mp_signal[s:s+35]
            if len(w) > 10: cycles.append(resample(w, 101))
            
        if not cycles: return None, None
        
        # QC Filter
        self_median = np.median(cycles, axis=0)
        clean_cycles = [c for c in cycles if (np.max(c)-np.min(c))>30 and pearsonr(c, self_median)[0]>0.8]
        if len(clean_cycles) < 3: clean_cycles = cycles
        
        mp_mean = np.mean(clean_cycles, axis=0)
    except: return None, None
    
    return gt_mean, mp_mean

# Manual ICC(2,1) Calculation using Numpy
def calculate_icc(df):
    # df columns: Subject, Method, Score
    # Convert to matrix: Rows=Subjects, Cols=Methods
    matrix = df.pivot(index='Subject', columns='Method', values='Score').values
    
    # n = number of subjects
    # k = number of raters/methods (2)
    n, k = matrix.shape
    
    # Grand Mean
    grand_mean = np.mean(matrix)
    
    # Sum of Squares
    S_total = np.sum((matrix - grand_mean)**2)
    
    # Between-Subjects Sum of Squares (Rows)
    row_means = np.mean(matrix, axis=1)
    S_row = k * np.sum((row_means - grand_mean)**2)
    
    # Between-Methods Sum of Squares (Columns)
    col_means = np.mean(matrix, axis=0)
    S_col = n * np.sum((col_means - grand_mean)**2)
    
    # Residual Sum of Squares
    S_resid = S_total - S_row - S_col
    
    # Mean Squares
    MS_row = S_row / (n - 1)
    MS_col = S_col / (k - 1)
    MS_resid = S_resid / ((n - 1) * (k - 1))
    
    # ICC(2,1) Formula: Absolute Agreement
    # (MS_row - MS_resid) / (MS_row + (k-1)*MS_resid + (k/n)*(MS_col - MS_resid))
    
    numerator = MS_row - MS_resid
    denominator = MS_row + (k - 1) * MS_resid + (k / n) * (MS_col - MS_resid)
    
    icc = numerator / denominator
    return icc

def main():
    print("Calculating ICC on N=21 Subjects...")
    
    subjects = []
    methods = []
    scores = []
    
    subjects_calib = []
    methods_calib = []
    scores_calib = []
    
    subject_dirs = sorted(glob.glob(f"{DATA_DIR}/[0-9]*"))
    subject_ids = [os.path.basename(d) for d in subject_dirs if os.path.basename(d).isdigit()]
    
    # LIMIT FOR SPEED (User Check)
    subject_ids = subject_ids[:5]
    
    # Assuming QC passed subjects from paper (N=21)
    # We will just try running on all and see which work
    
    valid_count = 0
    for sid in subject_ids:
        gt, mp = load_data_pair(sid)
        if gt is None or mp is None: continue
        
        # 1. Raw Data (Method A)
        # Use Mean ROM as the scalar metric for ICC? Or Mean Value?
        # Typically ICC for waveform uses specific points (Peak Ext/Flex) or Mean.
        # Let's use Mean Max Flexion (Peak) as the key gait parameter.
        
        val_gt = np.max(gt)
        val_mp = np.max(mp)
        
        subjects.extend([sid, sid])
        methods.extend(['GT', 'MP'])
        scores.extend([val_gt, val_mp])
        
        # 2. Calibrated Data (Method B)
        # Offset Correction: MP_calib = MP - (Mean_MP - Mean_GT_GrandMean) ?? 
        # No, Calibration typically means aligning baseline. 
        # Ideally: MP_calib = MP - Min(MP) (Set extension to 0) + Min(GT) (Set GT extension to 0)
        # Assuming Min ~ 0.
        
        offset = np.min(mp)
        val_mp_calib = val_mp - offset # Zero-referenced Peak (i.e. ROM)
        val_gt_calib = val_gt - np.min(gt) # Zero-referenced Peak (i.e. ROM)
        
        subjects_calib.extend([sid, sid])
        methods_calib.extend(['GT', 'MP_Calib'])
        scores_calib.extend([val_gt_calib, val_mp_calib])
        
        print(f"S{sid}: Raw(GT={val_gt:.1f}, MP={val_mp:.1f}) | Calib(GT={val_gt_calib:.1f}, MP={val_mp_calib:.1f})")
        valid_count += 1
        
    print(f"\nComputing ICC on {valid_count} subjects...")
    
    # Method A: Raw Absolute Agreement
    df_raw = pd.DataFrame({'Subject': subjects, 'Method': methods, 'Score': scores})
    icc_raw = calculate_icc(df_raw)
    
    # Method B: Zero-Referenced Agreement (Calibrated)
    df_calib = pd.DataFrame({'Subject': subjects_calib, 'Method': methods_calib, 'Score': scores_calib})
    icc_calib = calculate_icc(df_calib)
    
    print(f"\n=== ICC Results (Single Measures, Absolute Agreement) ===")
    print(f"Raw ICC (Absolute Peak): {icc_raw:.4f}")
    print(f"Calibrated ICC (Zero-Referenced Peak / effective ROM): {icc_calib:.4f}")

if __name__ == "__main__":
    main()
