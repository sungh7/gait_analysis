#!/usr/bin/env python3
print("Initializing...", flush=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

# Import core dependencies in safe order
from scipy.signal import resample
print("Importing Extractor...", flush=True)
from sagittal_extractor_2d import MediaPipeSagittalExtractor
print("Importing Segmentation...", flush=True)
from self_driven_segmentation import derive_self_template, find_dtw_matches_euclidean

EXCLUDED_IDS = ['04', '05', '06', '07', '12']
DATA_DIR = "/data/gait/data"
OUTPUT_DIR = "/data/gait/2d_project/excluded_subjects_test"
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def main():
    print("Instantiating Extractor...", flush=True)
    extractor = MediaPipeSagittalExtractor()
    print("Extractor Ready.", flush=True)
    
    for sid in EXCLUDED_IDS:
        print(f"\nProcessing Subject {sid}...", flush=True)
        folder_path = f"{DATA_DIR}/{int(sid)}"
        video_path = f"{folder_path}/{int(sid)}-2.mp4"
        
        if not os.path.exists(video_path):
            # Try padded folder name
            folder_path = f"{DATA_DIR}/{sid}"
            video_path = f"{folder_path}/{sid}-2.mp4"
            if not os.path.exists(video_path):
                print(f"Skipping {sid}: Video not found.", flush=True)
                continue
        
        try:
            # Extract
            print("  > Extracting landmarks...", flush=True)
            landmarks, _ = extractor.extract_pose_landmarks(video_path)
            
            print("  > Calculating angles...", flush=True)
            angles = extractor.calculate_joint_angles(landmarks)
            mp_signal = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
            
            # Auto-Template
            print("  > Deriving template...", flush=True)
            self_template, candidates = derive_self_template(mp_signal)
            
            if self_template is None:
                print("  > Failed to derive template.")
                continue
                
            # Segment
            starts = find_dtw_matches_euclidean(mp_signal, self_template)
            print(f"  > Found {len(starts)} cycles.", flush=True)
            
            # Grid Plot
            cycles = []
            for s in starts:
                seg = mp_signal[s:s+35]
                if len(seg) < 10: continue
                seg_res = resample(seg, 101)
                cycles.append(seg_res)
            
            if not cycles: continue
            
            N = len(cycles)
            cols = 5
            rows = (N // cols) + 1
            if rows > 10: rows = 10 # Limit plot size
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
            axes = axes.flatten()
            
            print("  > Plotting...", flush=True)
            for i in range(len(axes)):
                if i < N:
                    axes[i].plot(cycles[i], 'r-', alpha=0.8)
                    axes[i].plot(self_template, 'k--', linewidth=1, alpha=0.5)
                    axes[i].set_title(f"C{i+1}")
                    axes[i].axis('off')
                else:
                    axes[i].axis('off')
            
            plt.tight_layout()
            plt.suptitle(f"Subject {sid}: Auto-Template Check", fontsize=16)
            plt.subplots_adjust(top=0.95)
            
            save_path = f"{OUTPUT_DIR}/subject_{sid}_check.png"
            plt.savefig(save_path)
            plt.close()
            print(f"  > Saved {save_path}", flush=True)
            
        except Exception as e:
            print(f"Error on {sid}: {e}", flush=True)
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
