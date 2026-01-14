import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import seaborn as sns

def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def resample_cycle(cycle, target_len=100):
    x_old = np.linspace(0, 100, len(cycle))
    x_new = np.linspace(0, 100, target_len)
    f = interp1d(x_old, cycle, kind='linear')
    return f(x_new)

def plot_ensemble(data, output_path):
    joints = ['hip', 'ankle']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i, joint in enumerate(joints):
        mp_cycles = []
        gt_cycles = []
        
        for subj_id, subj_data in data.items():
            if joint in subj_data:
                # Use calibrated MP waveform
                mp = subj_data[joint]['mp_waveform']
                gt = subj_data[joint]['gt_waveform']
                
                mp_cycles.append(resample_cycle(mp))
                gt_cycles.append(resample_cycle(gt))
        
        mp_cycles = np.array(mp_cycles)
        gt_cycles = np.array(gt_cycles)
        
        # Calculate Mean and SD
        mp_mean = np.mean(mp_cycles, axis=0)
        mp_sd = np.std(mp_cycles, axis=0)
        gt_mean = np.mean(gt_cycles, axis=0)
        gt_sd = np.std(gt_cycles, axis=0)
        
        x = np.linspace(0, 100, 100)
        ax = axes[i]
        
        # Plot GT (Black)
        ax.plot(x, gt_mean, 'k-', linewidth=2, label='Ground Truth (Vicon)')
        ax.fill_between(x, gt_mean - gt_sd, gt_mean + gt_sd, color='gray', alpha=0.2)
        
        # Plot MP (Blue)
        ax.plot(x, mp_mean, 'b--', linewidth=2, label='MediaPipe (Calibrated)')
        ax.fill_between(x, mp_mean - mp_sd, mp_mean + mp_sd, color='blue', alpha=0.1)
        
        ax.set_title(f"{joint.capitalize()} Joint Ensemble (N={len(mp_cycles)})")
        ax.set_xlabel("Gait Cycle (%)")
        ax.set_ylabel("Angle (deg)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right')
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Ensemble plot to {output_path}")

def plot_dtw_example(data, output_path):
    # Find a good example (e.g., Ankle of a subject with high improvement)
    # Subject 13 or 18 usually good. Let's try to find one with 'ankle' data.
    subj_id = None
    for sid in ['13', '18', '01', '02']:
        if sid in data and 'ankle' in data[sid]:
            subj_id = sid
            break
            
    if not subj_id:
        print("No suitable subject found for DTW plot.")
        return

    # Get Raw MP and GT
    raw_mp = np.array(data[subj_id]['ankle']['raw_mp_waveform'])
    gt = np.array(data[subj_id]['ankle']['gt_waveform'])
    
    # Normalize for DTW calculation (z-score)
    raw_mp_norm = (raw_mp - np.mean(raw_mp)) / np.std(raw_mp)
    gt_norm = (gt - np.mean(gt)) / np.std(gt)
    
    # Compute DTW
    distance, path = fastdtw(raw_mp_norm.reshape(-1, 1), gt_norm.reshape(-1, 1), dist=euclidean)
    path = np.array(path)
    
    # Plot
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 3], height_ratios=[3, 1])
    
    # Main: Warping Path
    ax_main = fig.add_subplot(gs[0, 1])
    ax_main.plot(path[:, 0], path[:, 1], 'r-', linewidth=2)
    ax_main.set_title(f"DTW Warping Path (Subject {subj_id} Ankle)")
    ax_main.set_xlim(0, len(raw_mp_norm))
    ax_main.set_ylim(0, len(gt_norm))
    ax_main.grid(True)
    ax_main.set_ylabel("Ground Truth Index")
    
    # Top: Raw MP Waveform (x-axis)
    ax_top = fig.add_subplot(gs[1, 1], sharex=ax_main)
    ax_top.plot(raw_mp_norm, 'b-', label='Raw MediaPipe')
    ax_top.set_xlabel("MediaPipe Frame Index")
    ax_top.set_yticks([])
    ax_top.legend(loc='upper right', fontsize='small')
    
    # Left: GT Waveform (y-axis) - needs to be plotted vertically
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_left.plot(gt_norm, np.arange(len(gt_norm)), 'k-', label='Ground Truth')
    ax_left.set_ylabel("Ground Truth Frame Index")
    ax_left.set_xticks([])
    ax_left.invert_xaxis() # Mirror to match y-axis of main
    ax_left.legend(loc='upper left', fontsize='small')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved DTW plot to {output_path}")

def main():
    json_path = "/data/gait/final_validation_results.json"
    output_dir = "/data/gait/figures"
    
    data = load_data(json_path)
    
    plot_ensemble(data, f"{output_dir}/Figure8_Ensemble.png")
    plot_dtw_example(data, f"{output_dir}/Figure9_DTW_Path.png")

if __name__ == "__main__":
    main()
