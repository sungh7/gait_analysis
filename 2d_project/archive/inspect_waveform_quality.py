
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.signal import resample
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import derive_self_template, find_dtw_matches_euclidean

DATA_DIR = "/data/gait/data"
OUTPUT_DIR = "/data/gait/2d_project/research_metrics"

def load_gt_signal(sid):
    gt_csv = f"/data/gait/data/processed_new/S1_{int(sid):02d}_gait_long.csv"
    if not os.path.exists(gt_csv): return None
    try:
        df = pd.read_csv(gt_csv)
        subset = df[(df['joint'] == 'r.kn.angle') & (df['plane'] == 'sagittal')]
        if subset.empty: return None
        return resample(subset['condition1_avg'].values, 101)
    except: return None

def extract_mp_signal(sid):
    video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
    if not os.path.exists(video_path): 
         # Try fallback paths
         video_path = f"{DATA_DIR}/{sid}/{sid}.mp4"
         if not os.path.exists(video_path): return None

    # We need the full MP signal to segment cycles
    extractor = MediaPipeSagittalExtractor()
    landmarks, _ = extractor.extract_pose_landmarks(video_path)
    if not landmarks: return None
    angles = extractor.calculate_joint_angles(landmarks)
    signal = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
    
    # Segment
    template, _ = derive_self_template(signal)
    if template is None: return None
    starts = find_dtw_matches_euclidean(signal, template)
    
    cycles = []
    for s in starts:
        c = signal[s:s+35] 
        if len(c) > 10: cycles.append(resample(c, 101))
    
    if not cycles: return None
    return np.mean(cycles, axis=0) # Mean MP cycle

def inspect_waveforms():
    subjects = [26, 10] # 26=Good Pattern(Scaled), 10=Bad Pattern
    
    plt.figure(figsize=(12, 5))
    
    for i, sid in enumerate(subjects):
        gt = load_gt_signal(sid)
        mp = extract_mp_signal(str(sid))
        
        if gt is None or mp is None:
            print(f"Skipping S{sid} (Data missing)")
            continue
            
        plt.subplot(1, 2, i+1)
        
        # Plot Scaled MP to overlap for shape comparison
        # Scale MP to match GT range for visual shape check
        mp_norm = (mp - np.min(mp)) / (np.max(mp) - np.min(mp))
        gt_norm = (gt - np.min(gt)) / (np.max(gt) - np.min(gt))
        
        # Also plot Raw
        plt.plot(gt, 'k-', linewidth=2, label='Vicon (GT)')
        plt.plot(mp, 'r--', linewidth=2, label=f'MediaPipe (Raw)')
        
        # Calculate Correlation
        corr = np.corrcoef(gt, mp)[0,1]
        
        plt.title(f"Subject {sid} (r={corr:.2f})\n{ 'Pattern OK' if corr>0.8 else 'Pattern Distorted' }")
        plt.xlabel("% Gait Cycle")
        plt.ylabel("Knee Angle (deg)")
        plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/waveform_quality_check.png")
    print(f"Saved to {OUTPUT_DIR}/waveform_quality_check.png")

if __name__ == "__main__":
    inspect_waveforms()
