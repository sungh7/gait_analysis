#!/usr/bin/env python3
"""
GT-Free Gait Scalar Parameter Extraction
Extracts temporal and kinematic parameters using only video data (no force plate)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from scipy.signal import resample

sys.path.append("/data/gait/2d_project")
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import derive_self_template, find_dtw_matches_euclidean

DATA_DIR = "/data/gait/data"
OUTPUT_DIR = "/data/gait/2d_project"


def extract_scalar_parameters(signal, starts, fps=30, side='right'):
    """
    Extract scalar gait parameters from detected cycles.
    
    Args:
        signal: knee angle signal
        starts: detected cycle start frames
        fps: video frame rate
        side: 'left' or 'right'
    
    Returns:
        dict of scalar parameters
    """
    if len(starts) < 2:
        return None
    
    cycles = []
    cycle_durations = []
    
    for i in range(len(starts) - 1):
        start, end = starts[i], starts[i + 1]
        cycle_signal = signal[start:end]
        
        if len(cycle_signal) > 10:
            cycles.append(cycle_signal)
            cycle_durations.append(end - start)
    
    if not cycles:
        return None
    
    # === TEMPORAL PARAMETERS ===
    mean_cycle_frames = np.mean(cycle_durations)
    std_cycle_frames = np.std(cycle_durations)
    
    stride_time = mean_cycle_frames / fps  # seconds
    stride_time_variability = std_cycle_frames / fps
    cadence = 60 / stride_time  # steps per minute (actually strides/min)
    
    # === KINEMATIC PARAMETERS ===
    roms = []
    peak_flexions = []
    peak_extensions = []
    hs_angles = []  # angle at heel strike (start of cycle)
    
    for cycle in cycles:
        roms.append(np.max(cycle) - np.min(cycle))
        peak_flexions.append(np.max(cycle))
        peak_extensions.append(np.min(cycle))
        hs_angles.append(cycle[0])  # First point = heel strike
    
    # Normalize cycles and compute average waveform
    normalized_cycles = [resample(c, 101) for c in cycles]
    avg_waveform = np.mean(normalized_cycles, axis=0)
    
    return {
        'side': side,
        'n_cycles': len(cycles),
        
        # Temporal
        'stride_time_sec': stride_time,
        'stride_time_std': stride_time_variability,
        'cadence_per_min': cadence,
        'cycle_variability_cv': std_cycle_frames / mean_cycle_frames,  # Coefficient of variation
        
        # Kinematic
        'rom_mean': np.mean(roms),
        'rom_std': np.std(roms),
        'peak_flexion_mean': np.mean(peak_flexions),
        'peak_extension_mean': np.mean(peak_extensions),
        'hs_angle_mean': np.mean(hs_angles),
        
        # Raw data for visualization
        '_cycles': cycles,
        '_avg_waveform': avg_waveform
    }


def compute_symmetry(left_params, right_params):
    """Compute left-right symmetry indices"""
    if left_params is None or right_params is None:
        return None
    
    # Symmetry Index = (2 * |L - R|) / (L + R) * 100%
    # Perfect symmetry = 0%
    
    def sym_idx(l, r):
        if l + r == 0:
            return 0
        return 2 * abs(l - r) / (l + r) * 100
    
    return {
        'rom_symmetry_idx': sym_idx(left_params['rom_mean'], right_params['rom_mean']),
        'stride_time_symmetry_idx': sym_idx(left_params['stride_time_sec'], right_params['stride_time_sec']),
        'cadence_symmetry_idx': sym_idx(left_params['cadence_per_min'], right_params['cadence_per_min']),
        'rom_ratio_LR': left_params['rom_mean'] / right_params['rom_mean'] if right_params['rom_mean'] > 0 else np.nan,
    }


def visualize_results(left_params, right_params, symmetry, sid, output_path):
    """Create visualization of extracted parameters"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Layout: 2x3 grid
    # Row 1: Left waveform, Right waveform, Overlay
    # Row 2: Temporal bar chart, Kinematic bar chart, Symmetry
    
    # === Row 1: Waveforms ===
    ax1 = fig.add_subplot(2, 3, 1)
    if left_params and '_avg_waveform' in left_params:
        ax1.plot(np.linspace(0, 100, 101), left_params['_avg_waveform'], 'b-', lw=2)
        ax1.fill_between(np.linspace(0, 100, 101), left_params['_avg_waveform'], alpha=0.3)
    ax1.set_title(f'Left Knee (N={left_params["n_cycles"] if left_params else 0} cycles)')
    ax1.set_xlabel('Gait Cycle (%)')
    ax1.set_ylabel('Knee Flexion (°)')
    ax1.grid(alpha=0.3)
    
    ax2 = fig.add_subplot(2, 3, 2)
    if right_params and '_avg_waveform' in right_params:
        ax2.plot(np.linspace(0, 100, 101), right_params['_avg_waveform'], 'r-', lw=2)
        ax2.fill_between(np.linspace(0, 100, 101), right_params['_avg_waveform'], alpha=0.3, color='red')
    ax2.set_title(f'Right Knee (N={right_params["n_cycles"] if right_params else 0} cycles)')
    ax2.set_xlabel('Gait Cycle (%)')
    ax2.set_ylabel('Knee Flexion (°)')
    ax2.grid(alpha=0.3)
    
    ax3 = fig.add_subplot(2, 3, 3)
    if left_params and '_avg_waveform' in left_params:
        ax3.plot(np.linspace(0, 100, 101), left_params['_avg_waveform'], 'b-', lw=2, label='Left')
    if right_params and '_avg_waveform' in right_params:
        ax3.plot(np.linspace(0, 100, 101), right_params['_avg_waveform'], 'r-', lw=2, label='Right')
    ax3.set_title('L/R Overlay')
    ax3.set_xlabel('Gait Cycle (%)')
    ax3.set_ylabel('Knee Flexion (°)')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # === Row 2: Bar charts ===
    
    # Temporal parameters
    ax4 = fig.add_subplot(2, 3, 4)
    x = np.arange(3)
    width = 0.35
    
    left_vals = [left_params['stride_time_sec'], left_params['cadence_per_min']/60, 
                 left_params['cycle_variability_cv']] if left_params else [0, 0, 0]
    right_vals = [right_params['stride_time_sec'], right_params['cadence_per_min']/60,
                  right_params['cycle_variability_cv']] if right_params else [0, 0, 0]
    
    ax4.bar(x - width/2, left_vals, width, label='Left', color='blue', alpha=0.7)
    ax4.bar(x + width/2, right_vals, width, label='Right', color='red', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Stride Time\n(sec)', 'Cadence\n(steps/sec)', 'Variability\n(CV)'])
    ax4.set_title('Temporal Parameters')
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')
    
    # Kinematic parameters
    ax5 = fig.add_subplot(2, 3, 5)
    x = np.arange(4)
    
    left_vals = [left_params['rom_mean'], left_params['peak_flexion_mean'],
                 left_params['peak_extension_mean'], left_params['hs_angle_mean']] if left_params else [0]*4
    right_vals = [right_params['rom_mean'], right_params['peak_flexion_mean'],
                  right_params['peak_extension_mean'], right_params['hs_angle_mean']] if right_params else [0]*4
    
    ax5.bar(x - width/2, left_vals, width, label='Left', color='blue', alpha=0.7)
    ax5.bar(x + width/2, right_vals, width, label='Right', color='red', alpha=0.7)
    ax5.set_xticks(x)
    ax5.set_xticklabels(['ROM\n(°)', 'Peak Flex\n(°)', 'Peak Ext\n(°)', 'HS Angle\n(°)'])
    ax5.set_title('Kinematic Parameters')
    ax5.legend()
    ax5.grid(alpha=0.3, axis='y')
    
    # Symmetry indices
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    if symmetry:
        text = f"""
╔══════════════════════════════════════════╗
║       SYMMETRY INDICES (L vs R)          ║
╠══════════════════════════════════════════╣
║  ROM Symmetry Index:     {symmetry['rom_symmetry_idx']:5.1f}%          ║
║  Stride Time Sym. Index: {symmetry['stride_time_symmetry_idx']:5.1f}%          ║
║  ROM Ratio (L/R):        {symmetry['rom_ratio_LR']:5.2f}           ║
╠══════════════════════════════════════════╣
║  Interpretation:                         ║
║  • Sym. Index < 10% = Normal             ║
║  • Sym. Index > 20% = Asymmetric         ║
╚══════════════════════════════════════════╝
"""
    else:
        text = "Symmetry calculation failed"
    
    ax6.text(0.5, 0.5, text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.suptitle(f'Subject {sid}: GT-Free Gait Scalar Parameters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("GT-FREE SCALAR GAIT PARAMETER EXTRACTION")
    print("(No Force Plate Required)")
    print("=" * 60)
    
    extractor = MediaPipeSagittalExtractor()
    
    # Test subject
    sid = 1
    video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
    
    print(f"\nProcessing Subject {sid}...")
    print("  Extracting poses (limited to 500 frames)...", end=" ", flush=True)
    
    try:
        landmarks, _ = extractor.extract_pose_landmarks(video_path, max_frames=500)
        angles = extractor.calculate_joint_angles(landmarks)
        print("Done")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    fps = 30
    results = {}
    
    for side in ['right', 'left']:
        col_name = f'{side}_knee_angle'
        signal = angles[col_name].fillna(method='bfill').fillna(method='ffill').values
        
        print(f"  Processing {side} knee...", end=" ", flush=True)
        
        template_result = derive_self_template(signal)
        if template_result is None:
            print("Template failed")
            results[side] = None
            continue
        
        template, _ = template_result
        starts = find_dtw_matches_euclidean(signal, template)
        
        params = extract_scalar_parameters(signal, starts, fps, side)
        results[side] = params
        
        if params:
            print(f"Done ({params['n_cycles']} cycles)")
        else:
            print("Failed")
    
    # Compute symmetry
    symmetry = compute_symmetry(results.get('left'), results.get('right'))
    
    # === PRINT RESULTS ===
    print("\n" + "=" * 60)
    print("EXTRACTED SCALAR PARAMETERS (GT-FREE)")
    print("=" * 60)
    
    for side in ['right', 'left']:
        p = results.get(side)
        if p:
            print(f"\n{side.upper()} KNEE (N={p['n_cycles']} cycles):")
            print(f"  Temporal:")
            print(f"    Stride Time:     {p['stride_time_sec']:.3f} ± {p['stride_time_std']:.3f} sec")
            print(f"    Cadence:         {p['cadence_per_min']:.1f} steps/min")
            print(f"    Variability (CV):{p['cycle_variability_cv']:.3f}")
            print(f"  Kinematic:")
            print(f"    ROM:             {p['rom_mean']:.1f} ± {p['rom_std']:.1f}°")
            print(f"    Peak Flexion:    {p['peak_flexion_mean']:.1f}°")
            print(f"    Peak Extension:  {p['peak_extension_mean']:.1f}°")
            print(f"    HS Angle:        {p['hs_angle_mean']:.1f}°")
    
    if symmetry:
        print(f"\nSYMMETRY:")
        print(f"  ROM Sym. Index:    {symmetry['rom_symmetry_idx']:.1f}%")
        print(f"  ROM Ratio (L/R):   {symmetry['rom_ratio_LR']:.2f}")
    
    # Visualization
    output_path = f"{OUTPUT_DIR}/gt_free_scalars_S{sid}.png"
    visualize_results(results.get('left'), results.get('right'), symmetry, sid, output_path)
    
    # Save to CSV
    csv_data = []
    for side in ['right', 'left']:
        p = results.get(side)
        if p:
            row = {k: v for k, v in p.items() if not k.startswith('_')}
            csv_data.append(row)
    
    if csv_data:
        df = pd.DataFrame(csv_data)
        csv_path = f"{OUTPUT_DIR}/gt_free_scalars_S{sid}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")
    
    print("\n✅ GT-Free scalar extraction complete!")


if __name__ == "__main__":
    main()
