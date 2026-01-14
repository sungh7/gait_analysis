import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

sys.path.append("/data/gait")
from mediapipe_csv_processor import MediaPipeCSVProcessor
from dtw_alignment import DTWAligner

def analyze_all_subjects():
    data_dir = Path("/data/gait/data")
    processed_dir = Path("/data/gait/processed")
    
    # Find all MediaPipe files
    mp_files = sorted([f for f in data_dir.glob("**/*side_pose*.csv") if f.is_file()])
    
    print(f"Found {len(mp_files)} MediaPipe files")
    
    processor = MediaPipeCSVProcessor(
        conversion_params_path="/data/gait/angle_conversion_params.json"
    )
    
    aligner = DTWAligner()
    
    results = []
    
    for mp_file in mp_files:
        # Extract Subject ID
        filename = mp_file.name
        parts = filename.split('-')
        if not (len(parts) >= 2 and parts[0].isdigit()):
            continue
        subject_id = f"S1_{int(parts[0]):02d}"
        
        gt_file = processed_dir / f"{subject_id}_gait_long.csv"
        if not gt_file.exists():
            print(f"Skipping {subject_id} (no GT file)")
            continue
            
        try:
            print(f"\nProcessing {subject_id}...")
            
            # Process MediaPipe
            mp_results = processor.process_csv_file(mp_file)
            
            # Load GT
            gt_df = pd.read_csv(gt_file)
            gt_df = gt_df[gt_df['plane'] == 'y']
            
            # Analyze each side
            for side in ['left', 'right']:
                mp_cycle = mp_results[side]['averaged_cycle']
                if mp_cycle is None:
                    continue
                    
                prefix = side[0]
                
                # Get Spatio-Temporal Parameters
                params = mp_results[side]['parameters']
                
                # Analyze each joint
                for joint in ['hip', 'knee', 'ankle']:
                    joint_code = f"{prefix}.{joint[:2]}.angle"
                    
                    # Get MP angles
                    mp_joint_rows = mp_cycle[mp_cycle['joint'] == joint_code]
                    if len(mp_joint_rows) != 101:
                        continue
                    mp_angles = mp_joint_rows['angle_mean'].values
                    
                    # Get GT angles
                    gt_joint_data = gt_df[gt_df['joint'] == joint_code].sort_values('gait_cycle')
                    if len(gt_joint_data) != 101:
                        continue
                    gt_angles = gt_joint_data['condition1_avg'].values
                    
                    # Calculate metrics BEFORE DTW
                    before_metrics = aligner._calculate_metrics(mp_angles, gt_angles)
                    
                    # Apply DTW
                    mp_aligned, dtw_dist = aligner.align_single_cycle(mp_angles, gt_angles)
                    
                    # Calculate metrics AFTER DTW
                    after_metrics = aligner._calculate_metrics(mp_aligned, gt_angles)
                    
                    # Calculate ROM
                    mp_rom = mp_angles.max() - mp_angles.min()
                    gt_rom = gt_angles.max() - gt_angles.min()
                    
                    results.append({
                        'subject_id': subject_id,
                        'side': side,
                        'joint': joint,
                        'num_cycles': mp_results[side]['num_cycles'],
                        # Before DTW
                        'rmse_before': before_metrics['rmse'],
                        'corr_before': before_metrics['correlation'],
                        'icc_before': before_metrics['icc'],
                        # After DTW
                        'rmse_after': after_metrics['rmse'],
                        'corr_after': after_metrics['correlation'],
                        'icc_after': after_metrics['icc'],
                        # ROM
                        'mp_rom': mp_rom,
                        'gt_rom': gt_rom,
                        'rom_ratio': mp_rom / gt_rom if gt_rom > 0 else np.nan,
                        # Spatio-Temporal (same for all joints of same side)
                        'cadence_spm': params.get('cadence_spm', np.nan),
                        'stride_time_s': params.get('stride_time_s', np.nan),
                        'velocity_mps': params.get('velocity_mps', np.nan),
                    })
                    
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
            continue
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Save to CSV
    output_file = "/data/gait/comprehensive_analysis_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Print Summary Statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS (After DTW)")
    print("="*60)
    
    for joint in ['hip', 'knee', 'ankle']:
        joint_data = df_results[df_results['joint'] == joint]
        print(f"\n{joint.upper()}:")
        print(f"  RMSE: {joint_data['rmse_after'].mean():.2f} ± {joint_data['rmse_after'].std():.2f}")
        print(f"  Correlation: {joint_data['corr_after'].mean():.3f} ± {joint_data['corr_after'].std():.3f}")
        print(f"  ICC: {joint_data['icc_after'].mean():.3f} ± {joint_data['icc_after'].std():.3f}")
        print(f"  ROM Ratio (MP/GT): {joint_data['rom_ratio'].mean():.3f} ± {joint_data['rom_ratio'].std():.3f}")
        
    # Spatio-Temporal Summary (unique by subject + side)
    st_data = df_results.drop_duplicates(subset=['subject_id', 'side'])
    print(f"\nSPATIO-TEMPORAL PARAMETERS:")
    print(f"  Cadence: {st_data['cadence_spm'].mean():.1f} ± {st_data['cadence_spm'].std():.1f} spm")
    print(f"  Stride Time: {st_data['stride_time_s'].mean():.2f} ± {st_data['stride_time_s'].std():.2f} s")
    print(f"  Velocity: {st_data['velocity_mps'].mean():.2f} ± {st_data['velocity_mps'].std():.2f} m/s")
    
    return df_results

if __name__ == "__main__":
    df = analyze_all_subjects()
