import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

sys.path.append("/data/gait")
from mediapipe_csv_processor import MediaPipeCSVProcessor

def extract_paired_data():
    data_dir = Path("/data/gait/data")
    processed_dir = Path("/data/gait/processed")
    
    # Find all MediaPipe files
    mp_files = sorted([f for f in data_dir.glob("**/*side_pose*.csv") if f.is_file()])
    
    print(f"Found {len(mp_files)} MediaPipe files")
    
    processor = MediaPipeCSVProcessor(
        conversion_params_path="/data/gait/angle_conversion_params.json"
    )
    
    dataset = []
    
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
            print(f"Processing {subject_id}...")
            
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
                
                # Analyze each joint
                for joint in ['hip', 'knee', 'ankle']:
                    joint_code = f"{prefix}.{joint[:2]}.angle"
                    
                    # Get MP angles
                    mp_joint_rows = mp_cycle[mp_cycle['joint'] == joint_code]
                    
                    # Handle duplicates if present
                    if len(mp_joint_rows) > 101:
                        mp_joint_rows = mp_joint_rows.drop_duplicates(subset=['gait_cycle'])
                        
                    if len(mp_joint_rows) != 101:
                        print(f"Skipping {subject_id} {side} {joint}: MP rows {len(mp_joint_rows)} != 101. Joint code: {joint_code}")
                        # Print unique joints in mp_cycle to debug
                        if len(mp_joint_rows) == 0:
                             print(f"Available MP joints: {mp_cycle['joint'].unique()}")
                        continue
                    mp_angles = mp_joint_rows['angle_mean'].values.tolist()
                    
                    # Get GT angles
                    gt_joint_data = gt_df[gt_df['joint'] == joint_code].sort_values('gait_cycle')
                    if len(gt_joint_data) != 101:
                        print(f"Skipping {subject_id} {side} {joint}: GT rows {len(gt_joint_data)} != 101")
                        continue
                    gt_angles = gt_joint_data['condition1_avg'].values.tolist()
                    
                    dataset.append({
                        'subject_id': subject_id,
                        'side': side,
                        'joint': joint,
                        'mp_cycle': mp_angles,
                        'gt_cycle': gt_angles
                    })
                    print(f"Added {subject_id} {side} {joint}")
                    
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
            continue
    
    # Save to JSON
    output_file = "/data/gait/paired_waveforms.json"
    with open(output_file, 'w') as f:
        json.dump(dataset, f)
    
    print(f"\nExtracted {len(dataset)} paired cycles.")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    extract_paired_data()
