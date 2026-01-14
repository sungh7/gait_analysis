"""
GAVD Feature Extraction - Fast Test Version
Processes first 50 files to quickly test the pipeline
"""

import sys
sys.path.append("/data/gait")

import pandas as pd
import numpy as np
from pathlib import Path
import json
from mediapipe_csv_processor import MediaPipeCSVProcessor

def load_clinical_annotations():
    """Load and combine all GAVD clinical annotation files"""
    gavd_data_dir = Path("/data/datasets/GAVD/data")
    
    dfs = []
    for i in range(1, 6):
        csv_path = gavd_data_dir / f"GAVD_Clinical_Annotations_{i}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path, low_memory=False)
            dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Get unique video sequences with their labels
    seq_labels = combined.groupby('seq').agg({
        'id': 'first',
        'dataset': 'first',
        'gait_pat': 'first',
        'cam_view': 'first'
    }).reset_index()
    
    return seq_labels

def process_gavd_files(max_files=50):
    """Process GAVD MediaPipe CSV files with our pipeline"""
    
    # Load clinical labels
    print("Loading clinical annotations...")
    seq_labels = load_clinical_annotations()
    print(f"Found {len(seq_labels)} unique sequences")
    
    # Find MediaPipe CSV files (only from right_side for consistency)
    mediapipe_dir = Path("/data/datasets/GAVD/mediapipe_pose")
    view_path = mediapipe_dir / "right_side"  # Use only right side view
    mp_files = list(view_path.glob("*.csv"))[:max_files]  # Limit to max_files
    
    print(f"Processing {len(mp_files)} MediaPipe CSV files (right_side view)")
    
    # Initialize processor
    processor = MediaPipeCSVProcessor(
        conversion_params_path="/data/gait/angle_conversion_params.json"
    )
    
    # Process each file
    results = []
    for i, mp_file in enumerate(mp_files):
        print(f"Processing {i+1}/{len(mp_files)}: {mp_file.name}...")
        
        try:
            # Extract seq ID from filename
            filename = mp_file.stem
            seq_id = filename.split('_')[0]
            
            # Find corresponding label
            label_row = seq_labels[seq_labels['seq'] == seq_id]
            if label_row.empty:
                print(f"  → No label found, skipping")
                continue
                
            label_row = label_row.iloc[0]
            
            # Process with our pipeline
            mp_results = processor.process_csv_file(mp_file)
            
            # Extract features for left and right sides
            for side in ['left', 'right']:
                if side not in mp_results:
                    continue
                    
                side_data = mp_results[side]
                
                if side_data['averaged_cycle'] is None:
                    continue
                
                # Extract parameters
                params = side_data.get('parameters', {})
                
                # Calculate ROM and other features from averaged cycle
                avg_cycle = side_data['averaged_cycle']
                
                features = {
                    'seq': seq_id,
                    'video_id': label_row['id'],
                    'side': side,
                    'dataset': label_row['dataset'],
                    'gait_pattern': label_row['gait_pat'],
                    'cam_view': label_row['cam_view'],
                    'num_cycles': side_data['num_cycles'],
                    
                    # Spatio-temporal parameters
                    'cadence_spm': params.get('cadence_spm', np.nan),
                    'stride_time_s': params.get('stride_time_s', np.nan),
                    'step_time_s': params.get('step_time_s', np.nan),
                    'step_length_m': params.get('step_length_m', np.nan),
                    'stride_length_m': params.get('stride_length_m', np.nan),
                    'velocity_mps': params.get('velocity_mps', np.nan),
                }
                
                # Extract joint-specific features (ROM, mean, std)
                for joint in ['Hip', 'Knee', 'Ankle']:
                    joint_code = f"{side[0]}.{joint.lower()[:2]}.angle"
                    joint_data = avg_cycle[avg_cycle['joint'] == joint_code]
                    
                    if not joint_data.empty:
                        angles = joint_data['angle_mean'].values
                        features[f'{joint.lower()}_rom'] = angles.max() - angles.min()
                        features[f'{joint.lower()}_mean'] = angles.mean()
                        features[f'{joint.lower()}_std'] = joint_data['angle_std'].mean()
                
                results.append(features)
                
        except Exception as e:
            print(f"  → Error: {e}")
            continue
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    return df_results

def main():
    print("="*60)
    print("GAVD Feature Extraction - Fast Test Version (50 files)")
    print("="*60)
    
    # Process files
    df_features = process_gavd_files(max_files=50)
    
    # Save results
    output_file = "/data/gait/gavd_extracted_features_test.csv"
    df_features.to_csv(output_file, index=False)
    
    print(f"\n✓ Extracted features from {len(df_features)} gait sequences")
    print(f"✓ Saved to {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nTotal samples: {len(df_features)}")
    print(f"\nDataset distribution:")
    print(df_features['dataset'].value_counts())
    print(f"\nGait pattern distribution:")
    print(df_features['gait_pattern'].value_counts())
    
    # Feature statistics
    print(f"\n\nSpatio-Temporal Parameters (mean ± std):")
    for col in ['cadence_spm', 'stride_time_s', 'velocity_mps']:
        if col in df_features.columns:
            mean_val = df_features[col].mean()
            std_val = df_features[col].std()
            print(f"  {col}: {mean_val:.2f} ± {std_val:.2f}")
    
    print(f"\n\nJoint ROM (mean ± std):")
    for joint in ['hip', 'knee', 'ankle']:
        col = f'{joint}_rom'
        if col in df_features.columns:
            mean_val = df_features[col].mean()
            std_val = df_features[col].std()
            print(f"  {joint.capitalize()}: {mean_val:.2f} ± {std_val:.2f}")

if __name__ == "__main__":
    main()
