import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("/data/gait")
from mediapipe_csv_processor import MediaPipeCSVProcessor
from dtw_alignment import DTWAligner

def analyze_s1_18_cycles():
    # Paths
    mp_path = "/data/gait/data/18/18-2_side_pose_fps30.csv"
    gt_path = "/data/gait/processed/S1_18_gait_long.csv"
    
    # Load GT
    gt = pd.read_csv(gt_path)
    # Get Left Knee GT (usually most sensitive to occlusion)
    gt_row = gt[(gt['joint'] == 'l.kn.angle') & (gt['plane'] == 'y')].sort_values('gait_cycle')
    gt_angles = gt_row['condition1_avg'].values
    
    # Process MP
    processor = MediaPipeCSVProcessor(conversion_params_path="/data/gait/angle_conversion_params.json")
    mp_results = processor.process_csv_file(mp_path)
    
    # Get all individual cycles for Left side
    # The processor stores them in 'gait_cycles' list in the results dict?
    # Let's check MediaPipeCSVProcessor.process_csv_file return structure.
    # It returns {'left': {'averaged_cycle': ..., 'gait_cycles': ...}, ...}
    
    left_cycles = mp_results['left']['raw_cycles']
    print(f"Found {len(left_cycles)} individual left gait cycles.")
    
    aligner = DTWAligner()
    correlations = []
    valid_cycles = []
    
    for i, cycle in enumerate(left_cycles):
        # Column name in raw_cycles is 'left_knee_angle'
        col_name = 'left_knee_angle'
        
        if col_name not in cycle.columns:
            print(f"Column {col_name} not found in cycle {i}")
            continue
            
        # Interpolate to 101 points
        cycle_interp = np.interp(
            np.linspace(0, 100, 101),
            cycle['gait_cycle'],
            cycle[col_name]
        )
        
        # Align and correlate
        aligned, _ = aligner.align_single_cycle(cycle_interp, gt_angles)
        metrics = aligner._calculate_metrics(aligned, gt_angles)
        corr = metrics['correlation']
        
        correlations.append(corr)
        valid_cycles.append(cycle_interp)
        
    # Plot Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(correlations, bins=20, kde=True)
    plt.title('Distribution of Left Knee Correlations (S1_18)', fontsize=16)
    plt.xlabel('Correlation with GT')
    plt.ylabel('Count')
    plt.axvline(0.7, color='red', linestyle='--', label='Threshold (0.7)')
    plt.legend()
    plt.savefig("/data/gait/s1_18_left_cycle_distribution.png")
    print("Saved distribution plot.")
    
    # Filter Good Cycles (> 0.7)
    good_cycles = [c for c, corr in zip(valid_cycles, correlations) if corr > 0.7]
    print(f"Found {len(good_cycles)} good cycles out of {len(left_cycles)}")
    
    if good_cycles:
        # Average Good Cycles
        good_avg = np.mean(good_cycles, axis=0)
        
        # Calculate metrics for cleaned average
        aligned_avg, _ = aligner.align_single_cycle(good_avg, gt_angles)
        final_metrics = aligner._calculate_metrics(aligned_avg, gt_angles)
        
        print(f"\nCleaned Average Results (Left Knee):")
        print(f"  Correlation: {final_metrics['correlation']:.3f}")
        print(f"  RMSE: {final_metrics['rmse']:.1f}")
        
        # Plot Comparison
        plt.figure(figsize=(10, 6))
        plt.plot(gt_angles, label='Ground Truth', color='black', linewidth=2)
        plt.plot(good_avg, label='MediaPipe (Cleaned Avg)', color='green', linewidth=2)
        
        # Plot bad cycles faintly
        bad_cycles = [c for c, corr in zip(valid_cycles, correlations) if corr <= 0.7]
        if bad_cycles:
            plt.plot(bad_cycles[0], color='red', alpha=0.1, label='Bad Cycles (Occluded)')
            for c in bad_cycles[1:]:
                plt.plot(c, color='red', alpha=0.1)
                
        plt.title(f"S1_18 Left Knee: Cleaned vs Raw (Corr: {final_metrics['correlation']:.2f})")
        plt.legend()
        plt.grid(True)
        plt.savefig("/data/gait/s1_18_left_cleaned_comparison.png")
        print("Saved cleaned comparison plot.")

if __name__ == "__main__":
    analyze_s1_18_cycles()
