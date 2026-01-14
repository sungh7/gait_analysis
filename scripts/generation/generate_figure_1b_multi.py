import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mediapipe as mp
import glob
from pathlib import Path

def get_pose_landmarks(image):
    """Run MediaPipe Pose on a single image to get screen landmarks."""
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return results.pose_landmarks

def draw_overlay_colored(image, landmarks):
    """Draw landmarks with Left=Blue, Right=Orange."""
    if not landmarks: return image.copy()
    
    annotated_image = image.copy()
    h, w, c = image.shape
    
    # Define colors (BGR for OpenCV)
    # Blue: #1f77b4 -> (180, 119, 31)
    # Orange: #ff7f0e -> (14, 127, 255)
    color_left = (180, 119, 31) 
    color_right = (14, 127, 255)
    color_mid = (200, 200, 200) # Gray for midline
    
    # Map landmarks to pixels
    points = {}
    for i, lm in enumerate(landmarks.landmark):
        points[i] = (int(lm.x * w), int(lm.y * h))
        
    mp_pose = mp.solutions.pose
    for start, end in mp_pose.POSE_CONNECTIONS:
        if start not in points or end not in points: continue
        
        is_left = (start % 2 != 0) and (end % 2 != 0)
        is_right = (start % 2 == 0) and (end % 2 == 0)
        
        if is_left: color = color_left
        elif is_right: color = color_right
        else: color = color_mid
        
        cv2.line(annotated_image, points[start], points[end], color, 3)
        
    # Draw joints
    for i, pt in points.items():
        if i > 32: continue 
        if i % 2 != 0: color = color_left
        else: color = color_right
        if i == 0: color = color_mid
        
        cv2.circle(annotated_image, pt, 4, color, -1)
        
    return annotated_image

def scan_video_for_passes(video_path):
    """Scan video to extract Hip X trajectory and detect passes."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Subsample for speed
    step = 5 
    frames = []
    hip_xs = []
    
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=0) as pose: # Use complexity 0 for speed
        for f in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if not ret: break
            
            # Process
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                avg_x = (left_hip.x + right_hip.x) / 2
                hip_xs.append(avg_x)
                frames.append(f)
            else:
                # If missed, append previous or None
                if hip_xs: hip_xs.append(hip_xs[-1])
                else: hip_xs.append(0.5)
                frames.append(f)
                
    cap.release()
    
    # Convert to Series for smoothing
    hip_x_series = pd.Series(hip_xs, index=frames)
    
    # Smooth
    # Window of ~1.5 seconds. FPS*1.5 / Step
    win = int((fps * 1.5) / step)
    if win < 3: win = 3
    hip_x_smooth = hip_x_series.rolling(window=win, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    
    # Find Peaks/Valleys
    from scipy.signal import find_peaks
    # Distance ~ 3 seconds
    dist = int((fps * 3) / step)
    peaks, _ = find_peaks(hip_x_smooth, distance=dist, prominence=0.05)
    valleys, _ = find_peaks(-hip_x_smooth, distance=dist, prominence=0.05)
    
    turn_indices = sorted(list(peaks) + list(valleys))
    turn_frames = [frames[i] for i in turn_indices]
    
    # Construct Passes
    boundaries = [0] + turn_frames + [total_frames]
    boundaries = sorted(list(set(boundaries)))
    
    passes = []
    min_len_frames = fps * 2 # 2 seconds
    min_disp = 0.2 # Screen width fraction
    
    for i in range(len(boundaries)-1):
        start = boundaries[i]
        end = boundaries[i+1]
        duration = end - start
        
        if duration < min_len_frames: continue
        
        # Check displacement
        # Find indices in our sampled data corresponding to start/end
        # Approximate
        try:
            s_idx = frames.index(start) if start in frames else min(range(len(frames)), key=lambda x: abs(frames[x]-start))
            e_idx = frames.index(end) if end in frames else min(range(len(frames)), key=lambda x: abs(frames[x]-end))
            
            disp = hip_x_smooth.iloc[e_idx] - hip_x_smooth.iloc[s_idx]
        except:
            disp = 0
            
        if abs(disp) < min_disp: continue
        
        direction = 'right' if disp > 0 else 'left'
        
        # Trim 10%
        margin = int(duration * 0.1)
        safe_start = start + margin
        safe_end = end - margin
        
        passes.append({
            'start': safe_start,
            'end': safe_end,
            'direction': direction,
            'center': (safe_start + safe_end) // 2
        })
        
    return passes

def process_subject(subj_id):
    print(f"Processing Subject {subj_id}...")
    output_path = f"/data/gait/figures/Figure1B_Subject_{subj_id}.png"
    
    video_files = glob.glob(f"/data/gait/data/{subj_id}/*-{subj_id}.mp4")
    csv_files = glob.glob(f"/data/gait/data/{subj_id}/*_side_pose_fps*.csv")
    
    if not video_files or not csv_files:
        print(f"  Missing files for Subject {subj_id}")
        return
        
    video_path = video_files[0]
    csv_path = csv_files[0]
    
    df = pd.read_csv(csv_path)
    if 'position' in df.columns:
         df = df.pivot(index='frame', columns='position', values=['x', 'y', 'z', 'visibility'])
         df.columns = [f'{pos.upper()}_{col}' for col, pos in df.columns]
         df = df.reset_index()
         
    print(f"  Scanning video: {video_path}")
    all_passes = scan_video_for_passes(video_path)
    print(f"  Found {len(all_passes)} passes.")
    
    # Select passes
    # Try to get 3 distinct passes if possible
    if len(all_passes) >= 4:
        passes = all_passes[1:4] # Skip 1, take 3
    elif len(all_passes) >= 3:
        passes = all_passes[:3] # Take 3
    elif len(all_passes) >= 1:
        passes = all_passes # Take whatever
    else:
        print(f"  No passes found for Subject {subj_id}")
        return

    fig = plt.figure(figsize=(16, 10))
    # Increase wspace to prevent overlap (User permitted wider spacing)
    gs = gridspec.GridSpec(3, 3, width_ratios=[1.6, 0.8, 1.4], wspace=0.4, hspace=0.3)
    
    for i, pass_info in enumerate(passes):
        if i >= 3: break
        
        target_frame = pass_info['center']
        pass_dir = pass_info.get('direction', 'right')
        
        # Read Video Frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()
        
        if not ret: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Landmarks
        landmarks = get_pose_landmarks(frame)
        
        # 1. Overlay (Panel A) - Colored
        # Convert RGB to BGR for OpenCV drawing, then back
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        overlay_img_bgr = draw_overlay_colored(frame_bgr, landmarks)
        overlay_img = cv2.cvtColor(overlay_img_bgr, cv2.COLOR_BGR2RGB)
        
        ax1 = plt.subplot(gs[i, 0])
        ax1.imshow(overlay_img)
        if i == 0: ax1.set_title(f"(A) Sagittal View (Subj {subj_id})", fontsize=14, fontweight='bold')
        ax1.set_ylabel(f"Pass {i+1}\n({pass_dir.title()})", fontsize=12, rotation=90, labelpad=10)
        ax1.axis('off')
        
        # 2. Stick Figure (Panel B) - Colored
        ax2 = plt.subplot(gs[i, 1])
        if landmarks:
            points = []
            for lm in landmarks.landmark:
                points.append([lm.x, lm.y])
            points = np.array(points)
            points[:, 1] = -points[:, 1]
            left_hip = points[23]; right_hip = points[24]
            center = (left_hip + right_hip) / 2
            points -= center
            min_y = np.min(points[:, 1]); max_y = np.max(points[:, 1])
            # Scale factor: Use 1.8 (90%) to leave margin for full body
            scale_factor = 1.8 / (max_y - min_y) if (max_y - min_y) > 0 else 1
            points *= scale_factor
            
            CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
            for start, end in CONNECTIONS:
                p_start = points[start]; p_end = points[end]
                
                is_left = (start % 2 != 0) and (end % 2 != 0)
                is_right = (start % 2 == 0) and (end % 2 == 0)
                
                if is_left: color = '#1f77b4' # Blue
                elif is_right: color = '#ff7f0e' # Orange
                else: color = 'gray'
                
                ax2.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color=color, linewidth=3)
                
            # Joints
            for idx, pt in enumerate(points):
                if idx > 32: continue
                if idx % 2 != 0: color = '#1f77b4'
                else: color = '#ff7f0e'
                if idx == 0: color = 'gray'
                ax2.plot(pt[0], pt[1], 'o', color=color, markersize=4)
            
        ax2.axhline(0, color='k', linewidth=0.5, linestyle='--')
        ax2.axvline(0, color='k', linewidth=0.5, linestyle='--')
        ax2.set_xlim(-1, 1); ax2.set_ylim(-1, 1); ax2.set_aspect('equal')
        ax2.grid(True, linestyle=':', alpha=0.6)
        if i == 0: ax2.set_title("(B) 2D Reconstruction", fontsize=14, fontweight='bold')
        if i == 2:
            ax2.set_xlabel("Horizontal Pos. (norm)", fontsize=10)
            ax2.set_ylabel("Vertical Pos. (norm)", fontsize=10)
        else:
            ax2.set_xticklabels([]); ax2.set_yticklabels([])
            
        # 3. Trajectory (Panel C)
        # Increase hspace slightly to avoid label overlap within the panel
        gs_inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i, 2], hspace=0.25)
        
        frames = df['frame']
        l_heel_y = df['LEFT_HEEL_y']
        r_heel_y = df['RIGHT_HEEL_y']
        start_f = pass_info['start']; end_f = pass_info['end']
        
        # Top: Left
        ax3_top = plt.subplot(gs_inner[0])
        ax3_top.plot(frames[start_f:end_f], l_heel_y[start_f:end_f], color='#1f77b4', linewidth=1.5, label='Left Heel')
        from scipy.signal import find_peaks
        peaks_l, _ = find_peaks(l_heel_y[start_f:end_f], distance=15, prominence=0.05)
        lhs_frames = frames.iloc[start_f:end_f].iloc[peaks_l].values
        for p in lhs_frames: ax3_top.axvline(x=p, color='k', linestyle='-', linewidth=1, alpha=0.7)
        ax3_top.plot([], [], 'k|', label='LHS')
        
        if i == 2:
            ax3_top.set_ylabel("L_Heel_Y", fontsize=10)
        else:
            ax3_top.set_yticklabels([])
            
        ax3_top.set_xticklabels([])
        # ax3_top.invert_yaxis() # Removed as requested
        ax3_top.legend(loc='upper right', fontsize=8, framealpha=0.8)
        if i == 0: ax3_top.set_title("(C) Gait Segmentation", fontsize=14, fontweight='bold')
        
        # Bottom: Right
        ax3_bot = plt.subplot(gs_inner[1])
        ax3_bot.plot(frames[start_f:end_f], r_heel_y[start_f:end_f], color='#ff7f0e', linewidth=1.5, label='Right Heel')
        peaks_r, _ = find_peaks(r_heel_y[start_f:end_f], distance=15, prominence=0.05)
        rhs_frames = frames.iloc[start_f:end_f].iloc[peaks_r].values
        for p in rhs_frames: ax3_bot.axvline(x=p, color='k', linestyle='-', linewidth=1, alpha=0.7)
        ax3_bot.plot([], [], 'k|', label='RHS')
        
        # Always show X label/ticks for bottom plot
        ax3_bot.set_xlabel("Frame", fontsize=10)
        
        if i == 2:
            ax3_bot.set_ylabel("R_Heel_Y", fontsize=10)
        else:
            ax3_bot.set_yticklabels([])
            
        # ax3_bot.invert_yaxis() # Removed as requested
        ax3_bot.legend(loc='upper right', fontsize=8, framealpha=0.8)
        
        # Current Frame Marker (Thicker)
        ax3_top.axvline(x=target_frame, color='g', linestyle='--', linewidth=2.5, alpha=0.8)
        ax3_bot.axvline(x=target_frame, color='g', linestyle='--', linewidth=2.5, alpha=0.8)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")

def main():
    # Find all subject directories
    data_dir = "/data/gait/data"
    subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()])
    
    print(f"Found {len(subdirs)} subjects: {subdirs}")
    
    for subj_id in subdirs:
        try:
            process_subject(subj_id)
        except Exception as e:
            print(f"Error processing subject {subj_id}: {e}")

if __name__ == "__main__":
    import os
    main()
