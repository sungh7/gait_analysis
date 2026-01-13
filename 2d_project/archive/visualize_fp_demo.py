#!/usr/bin/env python3
"""
Simplified Visualization: Force Plate GT vs AT-DTW Detection
Uses synthetic demo data to show concept without video processing.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "/data/gait/2d_project"


def generate_demo_gait_signal(n_cycles=12, fps=30, cycle_duration=1.1):
    """Generate a synthetic gait knee angle signal"""
    frames_per_cycle = int(fps * cycle_duration)
    total_frames = n_cycles * frames_per_cycle
    
    t = np.linspace(0, total_frames, total_frames)
    
    # Typical knee flexion pattern in gait cycle
    # Two peaks: stance (0-60%) and swing (60-100%)
    signal = np.zeros(total_frames)
    
    for i in range(n_cycles):
        start = i * frames_per_cycle
        end = (i + 1) * frames_per_cycle
        cycle_t = np.linspace(0, 2*np.pi, frames_per_cycle)
        
        # Simplified knee angle: extension at HS, flexion in swing
        # Add some variance
        amplitude = 50 + np.random.uniform(-5, 5)
        offset = 15 + np.random.uniform(-3, 3)
        cycle_signal = -amplitude * np.cos(cycle_t) + amplitude + offset
        
        # Add swing phase bump
        swing_bump = 15 * np.exp(-0.5 * ((cycle_t - 4.2) / 0.5)**2)
        cycle_signal += swing_bump
        
        signal[start:end] = cycle_signal
    
    # Add noise
    signal += np.random.normal(0, 2, total_frames)
    
    return signal, frames_per_cycle


def detect_hs_peaks(signal, min_dist=25):
    """Simple HS detection: find local minima (max extension)"""
    from scipy.signal import find_peaks
    
    # HS = minimum (max extension)
    inverted = -signal
    peaks, _ = find_peaks(inverted, distance=min_dist, prominence=10)
    return peaks


def main():
    np.random.seed(42)
    
    # Parameters
    n_gt_cycles = 5  # Force plate captures 5 cycles
    total_cycles = 12  # Total cycles in video
    fp_range = (0.25, 0.65)  # Force plate covers this range
    
    # Generate signal
    signal, period = generate_demo_gait_signal(n_cycles=total_cycles)
    total_frames = len(signal)
    
    # Detect all HS events (like AT-DTW would)
    at_starts = detect_hs_peaks(signal, min_dist=int(period * 0.7))
    
    # Force plate region
    fp_start = int(total_frames * fp_range[0])
    fp_end = int(total_frames * fp_range[1])
    
    # GT HS positions (only within force plate)
    gt_positions = np.linspace(fp_start, fp_end, n_gt_cycles + 1)[:-1].astype(int)
    
    # Count matching
    tolerance = 5
    matched_at = []
    for gt in gt_positions:
        for at in at_starts:
            if abs(gt - at) <= tolerance:
                matched_at.append(at)
                break
    
    # Detections in FP region
    at_in_fp = [s for s in at_starts if fp_start <= s < fp_end]
    at_outside_fp = [s for s in at_starts if s < fp_start or s >= fp_end]
    
    # === VISUALIZATION ===
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    # Panel 1: Full signal with all events
    ax1 = axes[0]
    frames = np.arange(len(signal))
    ax1.plot(frames, signal, 'gray', linewidth=1.2, alpha=0.8, label='Knee Angle (MediaPipe)')
    
    # Force plate region shading
    ax1.axvspan(fp_start, fp_end, alpha=0.15, color='blue', label='Force Plate Region')
    
    # AT-DTW detections (red for outside FP, green for inside FP)
    for i, s in enumerate(at_outside_fp):
        if i == 0:
            ax1.axvline(s, color='orange', linewidth=1.5, alpha=0.6, 
                       label=f'AT-DTW Outside FP ({len(at_outside_fp)} cycles)')
        else:
            ax1.axvline(s, color='orange', linewidth=1.5, alpha=0.6)
    
    for i, s in enumerate(at_in_fp):
        if i == 0:
            ax1.axvline(s, color='red', linewidth=2, alpha=0.8, 
                       label=f'AT-DTW Inside FP ({len(at_in_fp)} cycles)')
        else:
            ax1.axvline(s, color='red', linewidth=2, alpha=0.8)
    
    # GT positions (green dashed)
    for i, g in enumerate(gt_positions):
        if i == 0:
            ax1.axvline(g, color='green', linestyle='--', linewidth=3, 
                       label=f'Force Plate GT ({len(gt_positions)} cycles)')
        else:
            ax1.axvline(g, color='green', linestyle='--', linewidth=3)
    
    ax1.set_xlabel('Frame (30 fps)', fontsize=12)
    ax1.set_ylabel('Knee Flexion (°)', fontsize=12)
    ax1.set_title('Force Plate GT vs AT-DTW Detection: Full Trial View', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Zoomed force plate region
    ax2 = axes[1]
    zoom_signal = signal[fp_start:fp_end]
    zoom_frames = np.arange(fp_start, fp_end)
    ax2.plot(zoom_frames, zoom_signal, 'navy', linewidth=1.5)
    
    # GT and matched AT
    for g in gt_positions:
        ax2.axvline(g, color='green', linestyle='--', linewidth=3, alpha=0.9)
    
    for s in at_in_fp:
        ax2.axvline(s, color='red', linewidth=2.5, alpha=0.8)
    
    # Highlight matches
    for m in matched_at:
        ax2.axvline(m, color='lime', linewidth=4, alpha=0.3)
    
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Knee Flexion (°)', fontsize=12)
    ax2.set_title(f'Force Plate Region: {len(matched_at)}/{len(gt_positions)} GT Matched (±{tolerance} frames)', 
                  fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Statistics summary
    ax3 = axes[2]
    ax3.axis('off')
    
    # Calculate metrics
    tp = len(matched_at)
    fn = len(gt_positions) - tp
    fp_outside = len(at_outside_fp)  # These are valid cycles, just no GT
    fp_overdet = len(at_in_fp) - tp  # True over-segmentation
    
    recall = tp / len(gt_positions) * 100 if gt_positions.size > 0 else 0
    precision_raw = tp / len(at_starts) * 100 if len(at_starts) > 0 else 0
    precision_onplate = tp / (tp + fp_overdet) * 100 if (tp + fp_overdet) > 0 else 0
    
    text = f"""
╔══════════════════════════════════════════════════════════════╗
║                    MATCHING STATISTICS                       ║
╠══════════════════════════════════════════════════════════════╣
║  Ground Truth (Force Plate):       {len(gt_positions):3d} cycles                 ║
║  AT-DTW Total Detections:          {len(at_starts):3d} cycles                 ║
║    └─ Inside Force Plate:          {len(at_in_fp):3d} cycles                 ║
║    └─ Outside Force Plate:         {len(at_outside_fp):3d} cycles (no GT, not errors!) ║
╠══════════════════════════════════════════════════════════════╣
║  ✅ Matched (TP):                   {tp:3d} cycles                 ║
║  ❌ Missed (FN):                    {fn:3d} cycles                 ║
║  ⚠️  Over-segmentation (in FP):     {fp_overdet:3d} cycles                 ║
╠══════════════════════════════════════════════════════════════╣
║  Recall (TP / GT):                {recall:5.1f}%                      ║
║  Raw Precision (TP / All AT):     {precision_raw:5.1f}%   ← Misleading!        ║
║  On-Plate Precision:              {precision_onplate:5.1f}%   ← True metric        ║
╚══════════════════════════════════════════════════════════════╝

📌 Key Insight:
   "FP-A" (Orange lines outside blue region) = Valid gait cycles WITHOUT force plate GT
   These are NOT algorithm errors - they simply weren't measured by force plates!
"""
    ax3.text(0.5, 0.5, text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = f"{OUTPUT_DIR}/fp_vs_atdtw_demo.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved: {output_path}")
    print(f"\nRecall: {recall:.1f}%")
    print(f"On-Plate Precision: {precision_onplate:.1f}%")


if __name__ == "__main__":
    main()
