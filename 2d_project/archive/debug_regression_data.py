
import pandas as pd
import numpy as np

CSV_PATH = "/data/gait/2d_project/research_metrics/final_benchmarks.csv"

def debug_data():
    df = pd.read_csv(CSV_PATH)
    print(f"Total rows: {len(df)}")
    
    # Check correlation
    valid_corr = df[df['correlation'] > 0.5]
    print(f"Rows with r > 0.5: {len(valid_corr)}")
    print("Valid Correlations:", valid_corr['correlation'].tolist())
    
    # Check Metadata simulation (mimicking the failure script logic)
    print("\n--- Metadata Checks ---")
    # I suspect Age or Cadence is None/NaN
    # Let's verify 'mp_cadence' computation from script logic? 
    # Can't run script parts easily, so I'll just check if subjects in valid_corr have info.json
    
    import os
    import json
    DATA_DIR = "/data/gait/data"
    
    for idx, row in valid_corr.iterrows():
        sid = str(row['subject'])
        sid_str = f"{int(float(sid)):02d}"
        
        json_path = f"{DATA_DIR}/processed_new/S1_{sid_str}_info.json"
        video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
        
        print(f"Subject {sid}:")
        print(f"  JSON Exists: {os.path.exists(json_path)}")
        print(f"  Video Exists: {os.path.exists(video_path)}")
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                info = json.load(f)
                demo = info.get('demographics', {})
                print(f"  Age: {demo.get('age')}")
                print(f"  Height: {demo.get('height_cm')}")
                print(f"  Weight: {demo.get('weight_kg', demo.get('weight_cm'))}")

if __name__ == "__main__":
    debug_data()
