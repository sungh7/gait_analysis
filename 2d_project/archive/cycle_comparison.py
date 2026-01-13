#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from tqdm import tqdm
import traceback

# Configuration
DATA_DIR = "/data/gait/data"
OUTPUT_DIR = "/data/gait/2d_project/cycle_comparison_results"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def parse_vector_gt_csv(csv_path):
    """
    Parse V3D exported CSV to dictionary of waveforms.
    """
    try:
        # Read raw
        df = pd.read_csv(csv_path, header=None, skiprows=4)
        target_params = ['r.kn.angle', 'r.hi.angle', 'r.an.angle']
        
        if df.shape[1] < 5:
            return None
            
        data = {}
        for param in target_params:
            rows = df[df[1] == param]
            if rows.empty:
                continue
            rows = rows.sort_values(by=0)
            try:
                # Col 4=X, 5=Y, 6=Z
                vals_x = pd.to_numeric(rows[4], errors='coerce').values
                vals_y = pd.to_numeric(rows[5], errors='coerce').values
                vals_z = pd.to_numeric(rows[6], errors='coerce').values
                data[param] = {'x': vals_x, 'y': vals_y, 'z': vals_z}
            except:
                pass
                
        return data
    except Exception as e:
        print(f"Error parsing GT {csv_path}: {e}")
        return None

def segment_gt_cycles_by_knee_extension(signal, fps=100):
    """
    Segment GT signal into cycles based on Knee Extension (Valleys).
    Input: Knee Flexion Angle signal (High=Flexion, Low=Extension)
    """
    # Smooth slightly?
    # Usually GT is clean.
    
    # Find Valleys (Peaks of inverted signal)
    # Extension -> Minima
    inverted = -signal
    
    # Prominence: Need significant extension.
    # Min distance: 20 frames (0.2s if 100fps) to catch faster gait
    # Prominence: 1 degree (GT is clean)
    
    peaks, _ = find_peaks(inverted, distance=20, prominence=1) 
    
    if len(peaks) < 2:
        return []
        
    cycles = []
    for i in range(len(peaks)-1):
        start = peaks[i]
        end = peaks[i+1]
        segment = signal[start:end]
        
        # Resample to 100
        x_old = np.linspace(0, 100, len(segment))
        x_new = np.linspace(0, 100, 100)
        resampled = np.interp(x_new, x_old, segment)
        cycles.append(resampled)
        
    return cycles

def process_subject(subject_id, extractor, max_frames=None):
    subject_dir = Path(DATA_DIR) / str(subject_id)
    video_path = subject_dir / f"{subject_id}-2.mp4"
    
    gt_pattern = subject_dir / "excel" / f"S{subject_id}_*_edited.csv"
    gt_files = list(glob.glob(str(gt_pattern)))
    if not gt_files:
        gt_files = list(glob.glob(str(subject_dir / "excel" / "*_edited.csv")))
        
    if not video_path.exists() or not gt_files:
        return None
        
    gt_path = gt_files[0]
    print(f"Processing Subject {subject_id}...")
    
    # 1. MP Extraction (Average Cycle)
    # process_video returns averaged_cycle dataframe (Long Format: joint, angle_mean, etc.)
    try:
        avg_mp_df, cycles_mp, info = extractor.process_video(str(video_path), side='right', max_frames=max_frames)
        
        if avg_mp_df is None or avg_mp_df.empty:
             print("MP Extraction returned None or empty")
             return None
             
        # Extract Right Knee Angle
        # Format: joint='right.kn.angle', angle_mean
        target_joint = 'right.kn.angle'
        knee_data = avg_mp_df[avg_mp_df['joint'] == target_joint]
        
        if knee_data.empty:
            print(f"Joint {target_joint} not found in MP output")
            return None
            
        # Ensure sorted by gait cycle (0-100)
        knee_data = knee_data.sort_values('gait_cycle')
        mp_signal = knee_data['angle_mean'].values
        
    except Exception as e:
        print(f"MP Error: {e}")
        traceback.print_exc()
        return None

    # 2. GT Extraction (Average Cycle)
    gt_data = parse_vector_gt_csv(gt_path)
    if not gt_data or 'r.kn.angle' not in gt_data:
        print("GT Data missing")
        return None
        
    gt_raw = gt_data['r.kn.angle']['y'] # Flexion is Y axis
    # Interpolate NaNs
    gt_raw = pd.Series(gt_raw).interpolate().values
    
    # GT is already time-normalized (101 points)
    if len(gt_raw) == 101:
        # Assume it is already the Average Gait Cycle (0-100%)
        # And usually 0% is Heel Strike
        gt_avg = gt_raw
        gt_cycles = [gt_raw] # Treat as 1 cycle for plotting consistency
        print(f"Loaded Normalized GT (101 points)")
    else:
        # Fallback for raw data if any (legacy support)
        gt_cycles = segment_gt_cycles_by_knee_extension(gt_raw, fps=100)
        if not gt_cycles:
             print(f"No GT cycles found (len={len(gt_raw)})")
             return None
        gt_avg = np.mean(gt_cycles, axis=0)
    
    # 3. Compare
    # Correlate
    corr = np.corrcoef(mp_signal, gt_avg)[0, 1]
    rmse = np.sqrt(np.mean((mp_signal - gt_avg)**2))
    
    # Plot
    plt.figure(figsize=(10, 6))
    x = np.linspace(0, 100, 100)
    
    # Plot individual GT cycles light
    for c in gt_cycles:
        plt.plot(x, c, color='gray', alpha=0.1)
        
    plt.plot(x, gt_avg, 'k-', linewidth=2, label=f'GT Mean (N={len(gt_cycles)})')
    plt.plot(x, mp_signal, 'r--', linewidth=2, label=f'MP Mean (N={len(cycles_mp)})')
    
    plt.title(f"Subject {subject_id} Gait Cycle Comparison\nr={corr:.3f}, RMSE={rmse:.2f}")
    plt.xlabel("% Gait Cycle")
    plt.ylabel("Knee Flexion Angle (deg)")
    plt.legend()
    plt.savefig(f"{OUTPUT_DIR}/subject_{subject_id}_cycle.png")
    plt.close()
    
    return {
        'subject': subject_id,
        'correlation': corr,
        'rmse': rmse,
        'mp_cycles': len(cycles_mp),
        'gt_cycles': len(gt_cycles)
    }

def main():
    print("Starting Cycle-Based Comparison (All Subjects)...")
    extractor = MediaPipeSagittalExtractor()
    results = []
    
    subject_dirs = sorted(glob.glob(f"{DATA_DIR}/[0-9]*"))
    subject_ids = [Path(d).name for d in subject_dirs if Path(d).name.isdigit()]
    
    for sid in tqdm(subject_ids):
        # Skip if done
        if (Path(OUTPUT_DIR) / f"subject_{sid}_cycle.png").exists():
             # Check if we need to re-run (e.g. if we want to fix Subject 1)
             # Since Subject 1 failed previously (no file), it's fine.
             print(f"Skipping Subject {sid} (Already processed)")
             continue

        try:
            res = process_subject(sid, extractor)
            if res:
                results.append(res)
        except Exception as e:
            print(f"Error {sid}: {e}")
            traceback.print_exc()
            
    if results:
        df = pd.DataFrame(results)
        df.to_csv(f"{OUTPUT_DIR}/cycle_results_debug.csv", index=False)
        print(df.describe())

if __name__ == "__main__":
    main()
