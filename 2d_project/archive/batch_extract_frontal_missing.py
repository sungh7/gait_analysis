import os
import glob
import cv2
import pandas as pd
from sagittal_extractor_2d import MediaPipeSagittalExtractor

DATA_DIR = "/data/gait/data"

def main():
    print("Starting Batch Frontal Extraction...")
    extractor = MediaPipeSagittalExtractor()
    
    # 1. Scan for Frontal Videos
    # Pattern: /data/gait/data/{sid}/{sid}-1.mp4
    # We will iterate 1..30 to be safe
    
    count = 0
    for i in range(1, 35):
        sid = str(i)
        
        # Check potential paths (some folders might be padded? No, previous checks showed '1', '2' etc without padding)
        # But 'processed_new' uses padded 'S1_01'. Raw data uses '1'.
        
        dir_path = f"{DATA_DIR}/{sid}"
        if not os.path.exists(dir_path):
            continue
            
        video_path = f"{dir_path}/{sid}-1.mp4"
        if not os.path.exists(video_path):
            # print(f"Skipping S{sid}: No frontal video found ({video_path})")
            continue
            
        # Check if CSV already exists
        # We need to determine FPS for the filename convention
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        output_csv = f"{dir_path}/{sid}-1_front_pose_fps{fps}.csv"
        
        if os.path.exists(output_csv):
            print(f"Skipping S{sid}: CSV already exists ({output_csv})")
            continue
            
        print(f"Extracting S{sid} (FPS {fps})...")
        try:
            landmarks_df = extractor.extract_pose_landmarks(video_path)
            landmarks_df.to_csv(output_csv, index=False)
            print(f"Saved {output_csv}")
            count += 1
        except Exception as e:
            print(f"Failed S{sid}: {e}")
            
    print(f"\nBatch Extraction Complete. Processed {count} videos.")

if __name__ == "__main__":
    main()
