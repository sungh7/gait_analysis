import glob
from pathlib import Path
import pandas as pd
import os

DATA_DIR = "/data/gait/data"
subject_dirs = sorted(glob.glob(f"{DATA_DIR}/[0-9]*"))
print(f"Found {len(subject_dirs)} dirs")

for d in subject_dirs:
    sid = Path(d).name
    if not sid.isdigit(): continue
    print(f"Checking Subject {sid}")
    
    # Check Video
    video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
    print(f"  Video: {video_path} -> {os.path.exists(video_path)}")
    
    # Check GT
    sid_str = f"{int(sid):02d}"
    csv_path = Path(f"/data/gait/data/processed_new/S1_{sid_str}_gait_long.csv")
    print(f"  GT: {csv_path} -> {csv_path.exists()}")
    
    if os.path.exists(video_path) and csv_path.exists():
        print("  MATCH: Ready to process.")
        break # Just check the first one
