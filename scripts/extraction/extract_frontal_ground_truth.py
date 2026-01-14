import pandas as pd
import json
import numpy as np
from pathlib import Path
import glob

def extract_vicon_angles(data_dir):
    results = {}
    
    # Find all edited.csv files
    csv_files = glob.glob(f"{data_dir}/*/excel/*_edited.csv")
    print(f"Found {len(csv_files)} CSV files.")
    
    for csv_path in csv_files:
        try:
            subject_id = Path(csv_path).stem.split('_')[0] + "_" + Path(csv_path).stem.split('_')[1] # S1_01
            print(f"Processing {subject_id}...")
            
            # Read CSV. The header is on line 0 (1-based line 1).
            # But the data structure is complex.
            # We rely on "VarName" column.
            
            # Read all lines to find the start of data blocks?
            # Actually pandas can read it, but the header is weird.
            # Let's read with header=0.
            df = pd.read_csv(csv_path, header=0)
            
            # Filter for relevant rows
            # The 'VarName' column contains 'r.hi.angle', 'r.kn.angle'
            
            joints = ['r.hi.angle', 'r.kn.angle', 'l.hi.angle', 'l.kn.angle']
            
            subject_data = {}
            
            for joint in joints:
                # Filter rows where VarName == joint
                # Note: The CSV structure shown in `head` implies multiple rows per joint?
                # "102,r.kn.angle,0,0,..."
                # "103,r.kn.angle,1,0,..."
                # So "Value1" seems to be Frame Number?
                # Let's check the head output again.
                # "1,r.an.angle,0,0,..." -> Value1 is 0.
                # "2,r.an.angle,1,0,..." -> Value1 is 1.
                # So Value1 is Frame Index.
                
                joint_df = df[df['VarName'] == joint].sort_values('Value1')
                
                if joint_df.empty:
                    continue
                
                # Extract X, Y, Z columns
                # "Condition 1: X", "Condition 1: Y", "Condition 1: Z"
                
                x_vals = joint_df['Condition 1: X'].values.tolist()
                y_vals = joint_df['Condition 1: Y'].values.tolist()
                z_vals = joint_df['Condition 1: Z'].values.tolist()
                
                subject_data[joint] = {
                    'x': x_vals,
                    'y': y_vals,
                    'z': z_vals
                }
            
            if subject_data:
                results[subject_id] = subject_data
                
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            
    return results

if __name__ == "__main__":
    data_dir = "/data/gait/data"
    output_file = "/data/gait/frontal_ground_truth.json"
    
    data = extract_vicon_angles(data_dir)
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved ground truth to {output_file}")
