import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Add current directory to path
sys.path.append("/data/gait")
from mediapipe_csv_processor import MediaPipeCSVProcessor

def analyze_subject(subject_id, mp_file, gt_file, output_dir):
    print(f"Analyzing {subject_id}...")
    
    # Load Data
    processor = MediaPipeCSVProcessor(
        conversion_params_path="/data/gait/angle_conversion_params.json"
    )
    # Ensure calibration is loaded (it loads from json by default if exists)
    
    try:
        mp_results = processor.process_csv_file(mp_file)
        gt_df = pd.read_csv(gt_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Prepare plots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"Gait Pattern Analysis: {subject_id}", fontsize=16)
    
    joints = ['Hip', 'Knee', 'Ankle']
    
    for i, joint in enumerate(joints):
        # Use Left side for visualization
        side = 'left'
        joint_code = f"l.{joint.lower()[:2]}.angle"
        mp_joint_name = f"left_{joint.lower()}_angle"
        
        # Get MP Data (Averaged Cycle)
        mp_cycle = mp_results[side]['averaged_cycle']
        if mp_cycle is None: continue
        mp_data = mp_cycle[mp_cycle['joint'] == joint_code]['angle_mean'].values
        
        # Get GT Data
        gt_data_rows = gt_df[(gt_df['joint'] == joint_code) & (gt_df['plane'] == 'y')].sort_values('gait_cycle')
        if len(gt_data_rows) != 101: continue
        gt_data = gt_data_rows['condition1_avg'].values
        
        if len(mp_data) != 101: continue
        
        # 1. DTW Calculation
        # Use simple absolute difference for 1D signals to avoid scipy dimension errors
        distance, path = fastdtw(mp_data, gt_data, dist=lambda x, y: abs(x - y))
        path = np.array(path)
        
        # Align MP to GT
        mp_aligned = np.zeros_like(gt_data)
        counts = np.zeros_like(gt_data)
        
        for mp_idx, gt_idx in path:
            mp_aligned[gt_idx] += mp_data[mp_idx]
            counts[gt_idx] += 1
            
        mp_aligned = mp_aligned / np.maximum(counts, 1)
        
        # Metrics
        norm_dist = distance / len(path)
        corr_orig = np.corrcoef(mp_data, gt_data)[0,1]
        corr_aligned = np.corrcoef(mp_aligned, gt_data)[0,1]
        
        # Plot 1: Original Signals
        ax1 = axes[i, 0]
        ax1.plot(gt_data, 'k-', label='Hospital GT', linewidth=2)
        ax1.plot(mp_data, 'r--', label='MediaPipe (Raw)', linewidth=2)
        ax1.set_title(f"{joint} - Original (Corr: {corr_orig:.2f})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Aligned Signals
        ax2 = axes[i, 1]
        ax2.plot(gt_data, 'k-', label='Hospital GT', linewidth=2)
        ax2.plot(mp_aligned, 'b--', label='MediaPipe (Aligned)', linewidth=2)
        ax2.set_title(f"{joint} - DTW Aligned (Corr: {corr_aligned:.2f})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Warping Path
        ax3 = axes[i, 2]
        ax3.plot(path[:, 1], path[:, 0], 'g-')
        ax3.plot([0, 100], [0, 100], 'k:', alpha=0.5) # Diagonal reference
        ax3.set_title(f"{joint} - Warping Path (Dist: {norm_dist:.2f})")
        ax3.set_xlabel("Hospital GT Index")
        ax3.set_ylabel("MediaPipe Index")
        ax3.grid(True, alpha=0.3)
        
        # Calculate Temporal Bias (Area between path and diagonal)
        # Above diagonal = MP is ahead (early)
        # Below diagonal = MP is behind (delayed)
        bias = np.mean(path[:, 0] - path[:, 1])
        ax3.text(5, 90, f"Lag Bias: {bias:.1f} frames", fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        print(f"  {joint}:")
        print(f"    Corr (Orig): {corr_orig:.3f}")
        print(f"    Corr (Aligned): {corr_aligned:.3f}")
        print(f"    DTW Dist (Norm): {norm_dist:.2f}")
        print(f"    Lag Bias: {bias:.2f} frames")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = Path(output_dir) / f"{subject_id}_dtw_analysis.png"
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

def main():
    data_dir = Path("/data/gait/data")
    processed_dir = Path("/data/gait/processed")
    output_dir = Path("/data/gait/analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Select a few representative subjects
    # Based on previous validation:
    # S1_18 (Good?), S1_23 (Average?), S1_25 (Bad?)
    subjects = ['S1_18', 'S1_23', 'S1_25'] 
    
    for subject_id in subjects:
        # Find files
        # Need to handle filename variations
        subject_num = int(subject_id.split('_')[1])
        
        mp_files = list(data_dir.glob(f"**/{subject_num}-*_side_pose*.csv"))
        if not mp_files:
             mp_files = list(data_dir.glob(f"**/{subject_num}_*_side_pose*.csv"))
             
        if not mp_files:
            print(f"No MP file for {subject_id}")
            continue
            
        mp_file = mp_files[0] # Take first one
        gt_file = processed_dir / f"{subject_id}_gait_long.csv"
        
        if not gt_file.exists():
            print(f"No GT file for {subject_id}")
            continue
            
        analyze_subject(subject_id, mp_file, gt_file, output_dir)

if __name__ == "__main__":
    main()
