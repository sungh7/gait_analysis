import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mediapipe_csv_processor import MediaPipeCSVProcessor
import json
import os

def generate_figure():
    # Paths
    mp_path = "/data/gait/data/1/1-2_side_pose_fps30.csv"
    gt_path = "/data/gait/processed/S1_01_gait_long.csv"
    output_path = "/data/gait/waveform_comparison.png"
    
    print(f"Loading MediaPipe data from {mp_path}...")
    processor = MediaPipeCSVProcessor()
    
    # Load calibration
    if os.path.exists("calibration_parameters.json"):
        with open("calibration_parameters.json", "r") as f:
            processor.calibration = json.load(f)
            print("Loaded calibration parameters.")
    
    # Process MP data
    df_mp_raw = processor.load_csv(mp_path)
    res = processor.preprocessor.process(df_mp_raw, fps=30)
    df_mp = processor.calculate_joint_angles(res.dataframe)
    
    # Segment into gait cycles (Left side)
    print("Segmenting gait cycles...")
    gait_cycles = processor.segment_gait_cycles(df_mp, side='left', fps=30)
    
    if not gait_cycles:
        print("Error: No gait cycles detected.")
        return

    # Average MP cycles
    # Interpolate each cycle to 101 points
    mp_cycles_norm = {'hip': [], 'knee': [], 'ankle': []}
    x_norm = np.linspace(0, 100, 101)
    
    for cycle in gait_cycles:
        # Hip
        if 'left_hip_angle' in cycle.columns:
            y = cycle['left_hip_angle'].values
            x = np.linspace(0, 100, len(y))
            mp_cycles_norm['hip'].append(np.interp(x_norm, x, y))
            
        # Knee
        if 'left_knee_angle' in cycle.columns:
            y = cycle['left_knee_angle'].values
            x = np.linspace(0, 100, len(y))
            mp_cycles_norm['knee'].append(np.interp(x_norm, x, y))
            
        # Ankle
        if 'left_ankle_angle' in cycle.columns:
            y = cycle['left_ankle_angle'].values
            x = np.linspace(0, 100, len(y))
            mp_cycles_norm['ankle'].append(np.interp(x_norm, x, y))
            
    # Calculate Mean and SD for MP
    mp_stats = {}
    for joint in ['hip', 'knee', 'ankle']:
        data = np.array(mp_cycles_norm[joint])
        mp_stats[joint] = {
            'mean': np.nanmean(data, axis=0),
            'std': np.nanstd(data, axis=0)
        }
        
    print(f"Loading Ground Truth data from {gt_path}...")
    df_gt = pd.read_csv(gt_path)
    
    # Filter GT data (S1_01, Left side)
    # Based on previous inspection:
    # Hip: plane='y' (Flexion)
    # Knee: plane='y' (Flexion)
    # Ankle: plane='z' (Dorsiflexion)
    
    gt_config = {
        'Hip': {'joint': 'l.hi.angle', 'plane': 'y'},
        'Knee': {'joint': 'l.kn.angle', 'plane': 'y'},
        'Ankle': {'joint': 'l.an.angle', 'plane': 'z'}
    }
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    joints_map = {'Hip': 'hip', 'Knee': 'knee', 'Ankle': 'ankle'}
    
    for i, (joint_name, config) in enumerate(gt_config.items()):
        ax = axes[i]
        mp_key = joints_map[joint_name]
        
        # Get GT data
        gt_subset = df_gt[
            (df_gt['joint'] == config['joint']) & 
            (df_gt['plane'] == config['plane'])
        ].sort_values('gait_cycle')
        
        gt_mean = gt_subset['condition1_avg'].values
        # Some GT files might have SD, but let's just plot Mean for clarity or use normal_avg for reference?
        # Let's plot Subject GT Mean vs MP Mean
        
        # Get MP data
        mp_mean = mp_stats[mp_key]['mean']
        mp_std = mp_stats[mp_key]['std']
        
        # Plot MP (Red)
        ax.plot(x_norm, mp_mean, 'r-', linewidth=2, label='MediaPipe')
        ax.fill_between(x_norm, mp_mean - mp_std, mp_mean + mp_std, color='r', alpha=0.2)
        
        # Plot GT (Black)
        if len(gt_mean) == 101:
            ax.plot(x_norm, gt_mean, 'k-', linewidth=2, label='Vicon')
        else:
            print(f"Warning: GT data for {joint_name} has {len(gt_mean)} points, expected 101.")
            
        ax.set_title(f"{joint_name} Flexion/Extension", fontsize=14, fontweight='bold')
        ax.set_xlabel("Gait Cycle (%)")
        if i == 0:
            ax.set_ylabel("Angle (degrees)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Calculate Correlation
        if len(gt_mean) == 101:
            corr = np.corrcoef(mp_mean, gt_mean)[0, 1]
            rmse = np.sqrt(np.mean((mp_mean - gt_mean)**2))
            ax.text(0.05, 0.9, f'r = {corr:.2f}\nRMSE = {rmse:.1f}°', transform=ax.transAxes,
                   bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Figure saved to {output_path}")

if __name__ == "__main__":
    generate_figure()
