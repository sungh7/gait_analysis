import sys
sys.path.append("/data/gait")
from mediapipe_sagittal_extractor import MediaPipeSagittalExtractor
import pandas as pd
import os

def reextract_s1_25():
    video_path = "/data/gait/data/25/25-2.mp4"
    output_path = "/data/gait/data/25/25-2_side_pose_fps30_new.csv"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    print(f"Re-extracting pose for S1_25 from {video_path}...")
    
    extractor = MediaPipeSagittalExtractor(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    # Process video
    averaged_cycle, gait_cycles, video_info = extractor.process_video(video_path, side='right')
    
    if averaged_cycle is not None:
        averaged_cycle.to_csv(output_path, index=False)
        print(f"Saved new extraction to {output_path}")
        
        # Also save the raw landmarks to check for jitter
        # We need to access the internal landmarks_df from the process_video call if we want raw data
        # But process_video returns averaged_cycle.
        # Let's just trust the averaged cycle for now.
        
        print("Extraction successful.")
    else:
        print("Extraction failed: No gait cycles detected.")

if __name__ == "__main__":
    reextract_s1_25()
