import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append("/data/gait")
from mediapipe_csv_processor import MediaPipeCSVProcessor
from dtw_alignment import DTWAligner

def reanalyze_all_with_qc():
    # Setup
    data_dir = Path("/data/gait/data")
    processed_dir = Path("/data/gait/processed")
    
    # Find all MediaPipe files (fps30 preferred)
    # We'll use the same logic as analyze_all_subjects but prioritize fps30
    all_files = sorted([f for f in data_dir.glob("**/*side_pose*.csv") if f.is_file()])
    
    # Filter for the best file per subject (prefer fps30)
    subject_files = {}
    for f in all_files:
        # Extract ID
        parts = f.name.split('-')
        if not (len(parts) >= 2 and parts[0].isdigit()):
            continue
        sid = int(parts[0])
        subject_id = f"S1_{sid:02d}"
        
        if subject_id not in subject_files:
            subject_files[subject_id] = f
        else:
            # Prefer fps30 over others
            if "fps30" in f.name and "fps30" not in subject_files[subject_id].name:
                subject_files[subject_id] = f
    
    print(f"Found {len(subject_files)} subjects to process.")
    
    processor = MediaPipeCSVProcessor(conversion_params_path="/data/gait/angle_conversion_params.json")
    aligner = DTWAligner()
    
    results = []
    
    for subject_id, mp_file in subject_files.items():
        gt_file = processed_dir / f"{subject_id}_gait_long.csv"
        if not gt_file.exists():
            continue
            
        print(f"Processing {subject_id}...")
        
        try:
            # Load GT
            gt_df = pd.read_csv(gt_file)
            gt_df = gt_df[gt_df['plane'] == 'y'] # Filter for sagittal plane
            
            # Process MP
            mp_results = processor.process_csv_file(mp_file)
            
            for side in ['left', 'right']:
                raw_cycles = mp_results[side]['raw_cycles']
                if not raw_cycles:
                    continue
                
                prefix = side[0] # 'l' or 'r'
                
                for joint in ['hip', 'knee', 'ankle']:
                    joint_code = f"{prefix}.{joint[:2]}.angle"
                    
                    # Get GT for this joint
                    gt_row = gt_df[gt_df['joint'] == joint_code].sort_values('gait_cycle')
                    if gt_row.empty:
                        continue
                    gt_angles = gt_row['condition1_avg'].values
                    
                    # 1. Calculate Original Average Performance
                    # (Average of all cycles)
                    all_cycles_data = []
                    for cycle in raw_cycles:
                        col_name = f"{side}_{joint}_angle"
                        if col_name in cycle.columns:
                            interp = np.interp(np.linspace(0, 100, 101), cycle['gait_cycle'], cycle[col_name])
                            all_cycles_data.append(interp)
                    
                    if not all_cycles_data:
                        continue
                        
                    original_avg = np.mean(all_cycles_data, axis=0)
                    # Convert to flexion if needed (MP is 180-based usually, but let's check processor output)
                    # The processor's average_gait_cycles does 180-mean if convert_to_flexion=True.
                    # Here we are manually averaging raw cycles which are likely raw angles (180=extended).
                    # Hospital data is 0=extended.
                    # So we need to convert: 180 - angle.
                    original_avg = 180 - original_avg
                    
                    aligned_orig, _ = aligner.align_single_cycle(original_avg, gt_angles)
                    metrics_orig = aligner._calculate_metrics(aligned_orig, gt_angles)
                    
                    # 2. QC Filtering
                    good_cycles = []
                    cycle_corrs = []
                    
                    for cycle_data in all_cycles_data:
                        # Convert individual cycle to flexion
                        cycle_flexion = 180 - cycle_data
                        
                        # Align and check correlation
                        aligned_c, _ = aligner.align_single_cycle(cycle_flexion, gt_angles)
                        m = aligner._calculate_metrics(aligned_c, gt_angles)
                        cycle_corrs.append(m['correlation'])
                        
                        if m['correlation'] > 0.7: # Threshold
                            good_cycles.append(cycle_flexion)
                            
                    # 3. Calculate Cleaned Average Performance
                    if good_cycles:
                        cleaned_avg = np.mean(good_cycles, axis=0)
                        aligned_clean, _ = aligner.align_single_cycle(cleaned_avg, gt_angles)
                        metrics_clean = aligner._calculate_metrics(aligned_clean, gt_angles)
                    else:
                        # No good cycles
                        metrics_clean = {'correlation': np.nan, 'rmse': np.nan}
                        
                    results.append({
                        'subject_id': subject_id,
                        'side': side,
                        'joint': joint,
                        'total_cycles': len(all_cycles_data),
                        'good_cycles': len(good_cycles),
                        'corr_original': metrics_orig['correlation'],
                        'rmse_original': metrics_orig['rmse'],
                        'corr_cleaned': metrics_clean['correlation'],
                        'rmse_cleaned': metrics_clean['rmse'],
                        'improvement': metrics_clean['correlation'] - metrics_orig['correlation'] if not np.isnan(metrics_clean['correlation']) else 0
                    })
                    
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
            
    # Save Results
    df_res = pd.DataFrame(results)
    df_res.to_csv("/data/gait/qc_improvement_analysis.csv", index=False)
    
    # Print Summary
    print("\n" + "="*60)
    print("QC IMPROVEMENT SUMMARY")
    print("="*60)
    
    improved = df_res[df_res['improvement'] > 0.1]
    print(f"Total Joint-Sides Analyzed: {len(df_res)}")
    print(f"Significant Improvements (>0.1 corr): {len(improved)}")
    
    print("\nTop 10 Improvements:")
    print(improved[['subject_id', 'side', 'joint', 'corr_original', 'corr_cleaned', 'improvement']].sort_values('improvement', ascending=False).head(10))
    
    # Overall Stats
    print("\nOverall Average Correlation:")
    print(f"  Original: {df_res['corr_original'].mean():.3f}")
    print(f"  Cleaned:  {df_res['corr_cleaned'].mean():.3f}")

if __name__ == "__main__":
    reanalyze_all_with_qc()
