#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample, correlate
from pathlib import Path
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from tqdm import tqdm

# Configuration
DATA_DIR = "/data/gait/data"
OUTPUT_DIR = "/data/gait/2d_project/full_comparison_results"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def parse_gt_csv(csv_path):
    """
    Parse V3D exported CSV where each row corresponds to a specific parameter/frame.
    Structure seems to be:
    Col 0: Frame Index?
    Col 1: Parameter Name (e.g. 'r.kn.angle')
    Col 4, 5, 6: X, Y, Z values?
    """
    try:
        # Read raw, assuming no useful header or header at line 4 but we scan col 1
        df = pd.read_csv(csv_path, header=None, skiprows=4)
        
        # Filter for relevant angles
        # We need r.kn.angle (Right Knee), r.hi.angle (Right Hip), r.an.angle (Right Ankle)
        target_params = ['r.kn.angle', 'r.hi.angle', 'r.an.angle']
        
        # Check column 1 for parameter names
        if df.shape[1] < 5:
            print("CSV has too few columns")
            return None
            
        data = {}
        
        for param in target_params:
            # Filter rows
            rows = df[df[1] == param]
            if rows.empty:
                continue
            
            # Extract data. Assuming Col 4 is X (Flexion usually), Col 5 Y, Col 6 Z.
            # We need to sort by Frame (Col 0) just in case, though usually sorted.
            rows = rows.sort_values(by=0)
            
            # Values
            try:
                # Column 4 = X (Flexion/Extension usually)
                vals_x = pd.to_numeric(rows[4], errors='coerce').values
                vals_y = pd.to_numeric(rows[5], errors='coerce').values
                vals_z = pd.to_numeric(rows[6], errors='coerce').values
                
                # Store all 3 just in case
                data[param] = {'x': vals_x, 'y': vals_y, 'z': vals_z}
            except Exception as e:
                print(f"Error parsing values for {param}: {e}")
                
        if not data:
            print("No angle data found in CSV")
            return None
            
        return data

    except Exception as e:
        print(f"Error parsing GT {csv_path}: {e}")
        return None

def align_signals(ref_signal, target_signal):
    """
    Align target_signal to ref_signal using cross-correlation.
    Returns aligned_target (trimmed/padded) and lag.
    """
    # Normalize
    ref_norm = (ref_signal - np.mean(ref_signal)) / (np.std(ref_signal) + 1e-6)
    target_norm = (target_signal - np.mean(target_signal)) / (np.std(target_signal) + 1e-6)
    
    correlation = correlate(ref_norm, target_norm, mode='full')
    lags = np.arange(-len(target_norm) + 1, len(ref_norm))
    best_lag = lags[np.argmax(correlation)]
    
    # Shift target
    if best_lag > 0:
        # Target starts earlier? No, positive lag means ref is ahead?
        # shift target right by lag
        aligned_target = np.roll(target_signal, best_lag)
        # trim
        aligned_target[:best_lag] = np.nan
    else:
        aligned_target = np.roll(target_signal, best_lag)
        aligned_target[best_lag:] = np.nan
        
    return aligned_target, best_lag

def process_subject(subject_id, extractor, max_frames=None):
    subject_dir = Path(DATA_DIR) / str(subject_id)
    if not subject_dir.exists():
        return None
        
    # Find Side Video
    video_path = subject_dir / f"{subject_id}-2.mp4"
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return None
        
    # Find GT CSV
    # Pattern: excel/S{subject_id}_01_edited.csv or similar
    gt_pattern = subject_dir / "excel" / f"S{subject_id}_*_edited.csv"
    gt_files = list(glob.glob(str(gt_pattern)))
    if not gt_files:
        # Try generic search in excel
        gt_files = list(glob.glob(str(subject_dir / "excel" / "*_edited.csv")))
        
    if not gt_files:
        print(f"GT not found for subject {subject_id}")
        return None
    
    
    gt_path = gt_files[0]
    print(f"Processing Subject {subject_id} | Video: {video_path.name} | GT: {Path(gt_path).name}")
    
    # 1. Extract 2D Angles
    # Run full video (no max_frames)
    # Note: process_video was removed to assume direct extraction
    
    landmarks_df, vid_info = extractor.extract_pose_landmarks(str(video_path), max_frames=max_frames)
    angles_df = extractor.calculate_joint_angles(landmarks_df)
    
    # 2. Load GT
    gt_data = parse_gt_csv(gt_path)
    if gt_data is None:
        return None
        
    # 3. Compare Signals (Right Knee Flexion)
    # 2D: 'right_knee_angle'
    # GT: 'r.kn.angle' -> 'x' component (Flexion)
    
    # Extract 2D signal
    signal_2d = angles_df['right_knee_angle'].values
    # Clean NaNs
    signal_2d = pd.Series(signal_2d).interpolate(limit_direction='both').values
    
    # Extract GT signal
    if 'r.kn.angle' in gt_data:
        # Use X component for Sagittal Flexion
        signal_gt = gt_data['r.kn.angle']['x']
    else:
        print("GT r.kn.angle not found in parsed data")
        return None
             
    # Clean GT (strings to float)
    try:
        signal_gt = pd.to_numeric(signal_gt, errors='coerce')
        # Handle outliers/NaNs
        signal_gt = pd.Series(signal_gt).interpolate(limit_direction='both').values
    except:
        print("Failed to clean GT signal")
        return None

    # 4. Synchronization / Resampling
    # Downsample GT (100Hz?) to Video (30Hz)
    # Or just resample both to common length?
    # Better: Resample GT to match length of Video * (GT_Duration / Video_Duration)
    # If we assume they record same event, durations should match approx?
    # Often they verify length diff.
    
    # Simple approach: Resample GT to same number of frames as 2D
    # (Assuming recordings start/stop roughly same time or we align later)
    signal_gt_resampled = resample(signal_gt, len(signal_2d))
    
    # Align
    aligned_gt, lag = align_signals(signal_2d, signal_gt_resampled)
    
    # Trim NaNs
    valid_mask = ~np.isnan(aligned_gt) & ~np.isnan(signal_2d)
    s1 = signal_2d[valid_mask]
    s2 = aligned_gt[valid_mask]
    
    if len(s1) < 100:
        print("Not enough overlap")
        return None
        
    # Calc Metrics
    corr = np.corrcoef(s1, s2)[0, 1]
    rmse = np.sqrt(np.mean((s1 - s2)**2))
    
    # Save Plot
    plt.figure(figsize=(10, 6))
    plt.plot(s1, label='2D MediaPipe', alpha=0.8)
    plt.plot(s2, label='GT (Aligned)', alpha=0.8)
    plt.title(f"Subject {subject_id} Right Knee: r={corr:.3f}, RMSE={rmse:.2f}")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/subject_{subject_id}_comparison.png")
    plt.close()
    
    return {
        'subject': subject_id,
        'correlation': corr,
        'rmse': rmse,
        'frames': len(s1)
    }

import traceback

def main():
    print("Starting Full Scale 2D vs GT Comparison...")
    extractor = MediaPipeSagittalExtractor()
    
    results = []
    
    # Iterate all potential subject folders
    # Assuming folders 1, 2, ...
    subject_dirs = sorted(glob.glob(f"{DATA_DIR}/[0-9]*"))
    subject_ids = [Path(d).name for d in subject_dirs if Path(d).name.isdigit()]
    
    for sid in tqdm(subject_ids):
        # Skip if already done
        if (Path(OUTPUT_DIR) / f"subject_{sid}_comparison.png").exists():
            print(f"Skipping Subject {sid} (Already processed)")
            continue

        try:
            res = process_subject(sid, extractor)
            if res:
                results.append(res)
        except Exception as e:
            print(f"CRITICAL ERROR processing Subject {sid}: {e}")
            traceback.print_exc()
            
    if results:
        df_res = pd.DataFrame(results)
        out_csv = f"{OUTPUT_DIR}/final_correlation_results.csv"
        df_res.to_csv(out_csv, index=False)
        print(f"\nFinal Results saved to {out_csv}")
        print(df_res.describe())
    else:
        print("No valid comparisons found.")

if __name__ == "__main__":
    main()
