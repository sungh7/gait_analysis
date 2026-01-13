import cv2
import numpy as np

videos = [
    "/data/gait/data/1/1-1.mp4",
    "/data/gait/data/1/1-2.mp4"
]

for path in videos:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Failed to open {path}")
        continue
        
    timestamps = []
    # Read first 300 frames
    for _ in range(300):
        ret, frame = cap.read()
        if not ret: break
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        
    cap.release()
    
    if len(timestamps) < 2:
        print(f"{path}: Not enough frames.")
        continue
        
    intervals = np.diff(timestamps)
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)
    
    effective_fps = 1000.0 / mean_interval if mean_interval > 0 else 0
    header_fps = cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS)
    
    print(f"--- {path} ---")
    print(f"Header FPS: {header_fps:.2f}")
    print(f"Effective FPS: {effective_fps:.4f}")
    print(f"Interval Mean: {mean_interval:.2f} ms")
    print(f"Interval Std: {std_interval:.2f} ms")
    if std_interval > 1.0:
        print("WARNING: Variable Frame Rate (VFR) detected!")
    else:
        print("Constant Frame Rate (CFR) confirmed.")
    print("")
