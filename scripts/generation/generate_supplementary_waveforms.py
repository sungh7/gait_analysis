import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mediapipe_csv_processor import MediaPipeCSVProcessor
import json
import os
from pathlib import Path

def generate_supplementary_waveforms():
    print("Generating Supplementary Waveforms PDF...")
    
    # Setup
    data_dir = Path("/data/gait/data")
    processed_dir = Path("/data/gait/processed")
    output_pdf = "/data/gait/supplementary_waveforms.pdf"
    
    # Load calibration
    processor = MediaPipeCSVProcessor()
    if os.path.exists("calibration_parameters.json"):
        with open("calibration_parameters.json", "r") as f:
            processor.calibration = json.load(f)
            
    # Find subjects with both MP and GT data
    mp_files = sorted([f for f in data_dir.glob("**/*side_pose*.csv") if f.is_file()])
    
    with PdfPages(output_pdf) as pdf:
        # Title Page
        plt.figure(figsize=(11, 8.5))
        plt.text(0.5, 0.5, "Supplementary Material: Individual Waveform Comparisons", 
                 ha='center', va='center', fontsize=24)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        for mp_file in mp_files:
            # Extract ID
            filename = mp_file.name
            parts = filename.split('-')
            if not (len(parts) >= 2 and parts[0].isdigit()):
                continue
            subject_id = f"S1_{int(parts[0]):02d}"
            
            gt_file = processed_dir / f"{subject_id}_gait_long.csv"
            if not gt_file.exists():
                continue
                
            print(f"Processing {subject_id}...")
            
            try:
                # Process MP
                df_mp_raw = processor.load_csv(mp_file)
                res = processor.preprocessor.process(df_mp_raw, fps=30)
                df_mp = processor.calculate_joint_angles(res.dataframe)
                
                # Segment cycles
                gait_cycles = processor.segment_gait_cycles(df_mp, side='left', fps=30)
                if not gait_cycles:
                    continue
                    
                # Normalize MP
                mp_cycles_norm = {'hip': [], 'knee': [], 'ankle': []}
                x_norm = np.linspace(0, 100, 101)
                
                for cycle in gait_cycles:
                    for joint, col in [('hip', 'left_hip_angle'), ('knee', 'left_knee_angle'), ('ankle', 'left_ankle_angle')]:
                        if col in cycle.columns:
                            y = cycle[col].values
                            x = np.linspace(0, 100, len(y))
                            mp_cycles_norm[joint].append(np.interp(x_norm, x, y))
                            
                mp_stats = {}
                for joint in ['hip', 'knee', 'ankle']:
                    data = np.array(mp_cycles_norm[joint])
                    if len(data) > 0:
                        mp_stats[joint] = {
                            'mean': np.nanmean(data, axis=0),
                            'std': np.nanstd(data, axis=0)
                        }
                    else:
                        mp_stats[joint] = None

                # Load GT
                df_gt = pd.read_csv(gt_file)
                
                # Plot
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.suptitle(f"Subject {subject_id} - Left Side Kinematics", fontsize=16)
                
                gt_config = {
                    'Hip': {'joint': 'l.hi.angle', 'plane': 'y', 'mp': 'hip'},
                    'Knee': {'joint': 'l.kn.angle', 'plane': 'y', 'mp': 'knee'},
                    'Ankle': {'joint': 'l.an.angle', 'plane': 'z', 'mp': 'ankle'}
                }
                
                for i, (joint_name, config) in enumerate(gt_config.items()):
                    ax = axes[i]
                    
                    # GT
                    gt_subset = df_gt[
                        (df_gt['joint'] == config['joint']) & 
                        (df_gt['plane'] == config['plane'])
                    ].sort_values('gait_cycle')
                    
                    if not gt_subset.empty:
                        gt_mean = gt_subset['condition1_avg'].values
                        if len(gt_mean) == 101:
                            ax.plot(x_norm, gt_mean, 'k-', linewidth=2, label='Vicon')
                    
                    # MP
                    mp_key = config['mp']
                    if mp_stats[mp_key]:
                        mp_mean = mp_stats[mp_key]['mean']
                        mp_std = mp_stats[mp_key]['std']
                        ax.plot(x_norm, mp_mean, 'r-', linewidth=2, label='MediaPipe')
                        ax.fill_between(x_norm, mp_mean - mp_std, mp_mean + mp_std, color='r', alpha=0.2)
                        
                        # Correlation
                        if not gt_subset.empty and len(gt_mean) == 101:
                            corr = np.corrcoef(mp_mean, gt_mean)[0, 1]
                            rmse = np.sqrt(np.mean((mp_mean - gt_mean)**2))
                            ax.text(0.05, 0.9, f'r = {corr:.2f}\nRMSE = {rmse:.1f}', transform=ax.transAxes,
                                   bbox=dict(facecolor='white', alpha=0.8))
                    
                    ax.set_title(joint_name)
                    ax.set_xlabel("Gait Cycle (%)")
                    if i == 0:
                        ax.set_ylabel("Angle (deg)")
                    ax.grid(True, alpha=0.3)
                    if i == 2:
                        ax.legend()
                        
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                
            except Exception as e:
                print(f"Error processing {subject_id}: {e}")
                
    print(f"Saved supplementary waveforms to {output_pdf}")

if __name__ == "__main__":
    generate_supplementary_waveforms()
