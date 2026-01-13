#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, resample, coherence
from fastdtw import fastdtw
from pathlib import Path
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from tqdm import tqdm
import traceback

DATA_DIR = "/data/gait/data"
OUTPUT_DIR = "/data/gait/2d_project/full_cycle_results"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def load_gt_template(sid):
    # Map sid '1' -> '01'
    sid_str = f"{int(sid):02d}"
    csv_path = Path(f"/data/gait/data/processed_new/S1_{sid_str}_gait_long.csv")
    if not csv_path.exists(): return None
    try:
        df = pd.read_csv(csv_path)
        subset = df[(df['joint'] == 'r.kn.angle') & (df['plane'] == 'sagittal')].sort_values('gait_cycle')
        if subset.empty: return None
        return subset['condition1_avg'].values # 101 points
    except: return None

def segment_cycles(signal, distance=20, prominence=None):
    # Find peaks (Extension ~ Minima for Flexion signal?)
    # GT is Heel Strike to Heel Strike.
    # We verified V3D GT is Flexion Angle.
    # V3D Flexion usually positive. Extension near 0.
    # So Heel Strike (Max Extension) corresponds to MINIMA in Flexion graph.
    
    # Invert signal to find minima as peaks
    inverted = -signal
    if prominence is None: prominence = max(signal)*0.1 # Dynamic prominence?
    
    # Use loose parameters to catch everything
    peaks, _ = find_peaks(inverted, distance=distance, prominence=5) 
    
    cycles = []
    # Segments are from peak to peak
    for i in range(len(peaks)-1):
        start, end = peaks[i], peaks[i+1]
        length = end - start
        # Filter impossible lengths (too short/long)
        # 30fps ... cycle ~1s = 30-40 frames.
        if 20 <= length <= 60: 
            segment = signal[start:end]
            cycles.append(segment)
            
    return cycles

def main():
    print("Starting Full Cycle Comparison...")
    extractor = MediaPipeSagittalExtractor() # High quality
    results = []
    
    subject_dirs = sorted(glob.glob(f"{DATA_DIR}/[0-9]*"))
    subject_ids = [Path(d).name for d in subject_dirs if Path(d).name.isdigit()]
    
    for sid in tqdm(subject_ids):
        try:
            # 1. Load GT
            gt_template = load_gt_template(sid)
            if gt_template is None: continue
            
            # 2. Extract MP
            video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
            if not os.path.exists(video_path): continue
            
            landmarks, _ = extractor.extract_pose_landmarks(video_path)
            angles = extractor.calculate_joint_angles(landmarks)
            mp_signal = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
            
            # 3. Segment MP Cycles
            mp_cycles = segment_cycles(mp_signal)
            if not mp_cycles: 
                print(f"No cycles found for {sid}")
                continue
                
            # 4. Compare Each Cycle & Grid Plot
            num_cycles = len(mp_cycles)
            cols = 5
            rows = (num_cycles // cols) + (1 if num_cycles % cols > 0 else 0)
            
            fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
            axes = axes.flatten()
            
            cycle_details = []
            
            # Plot GT reference on all subplots? Or just 1? 
            # User wants "separated comparison", implying comparing EACH MP to GT.
            
            for i, cyc in enumerate(mp_cycles):
                ax = axes[i]
                
                # Resample MP to 100 points
                cyc_norm = resample(cyc, 101)
                
                # DTW
                dist, _ = fastdtw(cyc_norm, gt_template, radius=1, dist=lambda x,y: abs(x-y))
                
                # Plot
                ax.plot(np.linspace(0, 100, 101), gt_template, 'k-', linewidth=2, label='GT')
                ax.plot(np.linspace(0, 100, 101), cyc_norm, 'r-', linewidth=2, label='MP')
                ax.set_title(f"Cycle {i+1} (DTW: {dist:.1f})")
                
                if i == 0: ax.legend()
                
                cycle_details.append({
                    'subject': sid,
                    'cycle_idx': i+1,
                    'dtw_dist': dist
                })
            
            # Hide empty subplots
            for j in range(i+1, len(axes)):
                axes[j].axis('off')
                
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/subject_{sid}_grid_comparison.png")
            plt.close()
            
            # Save individual details immediately (append mode)
            pd.DataFrame(cycle_details).to_csv(f"{OUTPUT_DIR}/individual_cycles.csv", mode='a', header=not os.path.exists(f"{OUTPUT_DIR}/individual_cycles.csv"), index=False)
            
            results.append({
                'subject': sid,
                'num_cycles_found': num_cycles,
                'avg_dtw_dist': np.mean([d['dtw_dist'] for d in cycle_details]),
            })
            
        except Exception as e:
            print(f"Error {sid}: {e}")
            traceback.print_exc()

    pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/full_results.csv", index=False)

if __name__ == "__main__":
    main()
