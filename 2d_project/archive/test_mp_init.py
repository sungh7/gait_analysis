import sys
print("Importing mediapipe...", flush=True)
import mediapipe as mp
print("MediaPipe imported. Initializing Pose...", flush=True)
try:
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1, 
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    print("Pose initialized successfully.", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
