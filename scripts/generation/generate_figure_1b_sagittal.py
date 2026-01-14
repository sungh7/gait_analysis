import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def draw_skeleton(img, row, w, h):
    # MediaPipe connections for pose
    CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
        (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
        (29, 31), (30, 32), (27, 31), (28, 32)
    ]
    
    # Map landmark names to indices
    # 11: left_shoulder, 12: right_shoulder
    # 23: left_hip, 24: right_hip
    # 25: left_knee, 26: right_knee
    # 27: left_ankle, 28: right_ankle
    # 29: left_heel, 30: right_heel
    # 31: left_foot_index, 32: right_foot_index
    
    # Extract coordinates
    # CSV cols: LEFT_SHOULDER_x, etc.
    # We need to map index to column name
    
    idx_to_name = {
        11: 'LEFT_SHOULDER', 12: 'RIGHT_SHOULDER',
        13: 'LEFT_ELBOW', 14: 'RIGHT_ELBOW',
        15: 'LEFT_WRIST', 16: 'RIGHT_WRIST',
        23: 'LEFT_HIP', 24: 'RIGHT_HIP',
        25: 'LEFT_KNEE', 26: 'RIGHT_KNEE',
        27: 'LEFT_ANKLE', 28: 'RIGHT_ANKLE',
        29: 'LEFT_HEEL', 30: 'RIGHT_HEEL',
        31: 'LEFT_FOOT_INDEX', 32: 'RIGHT_FOOT_INDEX'
    }
    
    points = {}
    for idx, name in idx_to_name.items():
        if f'{name}_x' in row:
            x = int(row[f'{name}_x'] * w)
            y = int(row[f'{name}_y'] * h)
            points[idx] = (x, y)
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1) # Red dots
            
    for start, end in CONNECTIONS:
        if start in points and end in points:
            cv2.line(img, points[start], points[end], (255, 255, 255), 2) # White lines
            
    return img

def main():
    # Paths
    video_path = "/data/gait/data/1/1-2.mp4"
    csv_path = "/data/gait/data/1/1-2_side_pose_fps30.csv"
    output_path = "/data/gait/figures/Figure1B_Segmentation_Sagittal.png"
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Check format (Wide vs Long)
    if 'position' in df.columns:
         # Pivot to wide format
         df = df.pivot(index='frame', columns='position', values=['x', 'y', 'z', 'visibility'])
         # Flatten columns: (x, nose) -> nose_x
         df.columns = [f'{pos.upper()}_{col}' for col, pos in df.columns]
         df = df.reset_index()
    
    # Ensure columns are uppercase for consistency with idx_to_name
    # My pivot logic above made them UPPERCASE_x (e.g. NOSE_x)
    # idx_to_name uses LEFT_SHOULDER
    # So LEFT_SHOULDER_x should match.
    
    # Select Frame
    target_frame = 160 # Arbitrary mid-walk frame
    
    # Get Frame Data
    if 'frame' not in df.columns:
        # Should be there after reset_index
        print("Error: 'frame' column missing after pivot")
        return

    row = df[df['frame'] == target_frame].iloc[0]
    
    # Read Video Frame
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read video frame")
        return
        
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    
    # Draw Skeleton on Frame
    frame_overlay = frame.copy()
    frame_overlay = draw_skeleton(frame_overlay, row, w, h)
    
    # Setup Figure
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.5])
    
    # Panel 1: Video Overlay
    ax1 = plt.subplot(gs[0])
    ax1.imshow(frame_overlay)
    ax1.set_title("Sagittal View (Raw)", fontsize=12)
    ax1.axis('off')
    
    # Panel 2: Stick Figure
    ax2 = plt.subplot(gs[1])
    # Plot connections
    CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
        (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
        (29, 31), (30, 32), (27, 31), (28, 32)
    ]
    idx_to_name = {
        11: 'LEFT_SHOULDER', 12: 'RIGHT_SHOULDER',
        23: 'LEFT_HIP', 24: 'RIGHT_HIP',
        25: 'LEFT_KNEE', 26: 'RIGHT_KNEE',
        27: 'LEFT_ANKLE', 28: 'RIGHT_ANKLE',
        29: 'LEFT_HEEL', 30: 'RIGHT_HEEL',
        31: 'LEFT_FOOT_INDEX', 32: 'RIGHT_FOOT_INDEX'
    }
    
    for start, end in CONNECTIONS:
        name_s = idx_to_name.get(start)
        name_e = idx_to_name.get(end)
        if name_s and name_e:
            xs = [row[f'{name_s}_x'], row[f'{name_e}_x']]
            ys = [row[f'{name_s}_y'], row[f'{name_e}_y']]
            ax2.plot(xs, ys, 'k-', linewidth=2)
            
    # Plot joints
    for idx, name in idx_to_name.items():
        ax2.plot(row[f'{name}_x'], row[f'{name}_y'], 'ro', markersize=5)
        
    ax2.set_xlim(0, 1)
    ax2.set_ylim(1, 0) # Invert Y to match image
    ax2.set_aspect('equal')
    ax2.set_title("2D Skeleton Reconstruction", fontsize=12)
    ax2.set_xlabel("X (Normalized)")
    ax2.set_ylabel("Y (Normalized)")
    
    # Panel 3: Heel Trajectory
    ax3 = plt.subplot(gs[2])
    
    # Extract trajectory
    frames = df['frame']
    heel_y_r = df['RIGHT_HEEL_y']
    heel_y_l = df['LEFT_HEEL_y']
    
    # Plot window around target frame
    window = 50
    start_f = max(0, target_frame - window)
    end_f = min(len(df), target_frame + window)
    
    ax3.plot(frames[start_f:end_f], heel_y_r[start_f:end_f], 'orange', label='Right Heel Y')
    ax3.plot(frames[start_f:end_f], heel_y_l[start_f:end_f], 'blue', label='Left Heel Y')
    
    # Mark current frame
    ax3.axvline(x=target_frame, color='k', linestyle='--', label='Current Frame')
    
    # Mark peaks (Heel Strikes approx)
    # Simple peak detection for visualization
    from scipy.signal import find_peaks
    peaks_r, _ = find_peaks(heel_y_r[start_f:end_f], distance=10)
    ax3.plot(frames.iloc[start_f:end_f].iloc[peaks_r], heel_y_r.iloc[start_f:end_f].iloc[peaks_r], 'rx')
    
    ax3.set_title("Gait Segmentation (Heel Y-Trajectory)", fontsize=12)
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Vertical Position (Y)")
    ax3.legend()
    ax3.invert_yaxis() # Y increases downwards in MP
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")

if __name__ == "__main__":
    main()
