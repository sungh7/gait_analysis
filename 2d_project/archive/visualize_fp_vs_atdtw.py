#!/usr/bin/env python3
"""
Visualization: Force Plate GT Events vs AT-DTW Detected Heel Strikes
Shows how well the automatic detection aligns with ground truth events.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import json
from scipy.signal import resample

sys.path.append("/data/gait/2d_project")
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from self_driven_segmentation import derive_self_template, find_dtw_matches_euclidean

DATA_DIR = "/data/gait/data"
PROCESSED_DIR = "/data/gait/data/processed_new"
OUTPUT_DIR = "/data/gait/2d_project"


def load_gt_info(sid):
    """Load GT stride count from processed info.json"""
    info_path = f"{PROCESSED_DIR}/S1_{sid:02d}_info.json"
    if not os.path.exists(info_path):
        return None
    try:
        with open(info_path, 'r') as f:
            info = json.load(f)
        return info.get('demographics', {}).get('right_strides', 0)
    except:
        return None


def run_at_dtw(video_path):
    """Run AT-DTW detection and return signal + detected starts"""
    extractor = MediaPipeSagittalExtractor()
    lm, _ = extractor.extract_pose_landmarks(video_path)
    angles = extractor.calculate_joint_angles(lm)
    sig = angles['right_knee_angle'].fillna(method='bfill').fillna(method='ffill').values
    
    template, _ = derive_self_template(sig)
    if template is None:
        return sig, [], template
    
    starts = find_dtw_matches_euclidean(sig, template)
    return sig, starts, template


def estimate_gt_positions(signal_length, gt_count, fp_range=(0.2, 0.8)):
    """
    Estimate GT heel strike positions.
    Assumes force plate captures the middle portion of the walkway.
    """
    if gt_count == 0:
        return []
    
    fp_start = int(signal_length * fp_range[0])
    fp_end = int(signal_length * fp_range[1])
    
    # Evenly distribute GT events within force plate region
    gt_positions = np.linspace(fp_start, fp_end, gt_count + 1)[:-1]
    return gt_positions.astype(int)


def visualize_alignment(sid, signal, at_starts, gt_positions, template, output_path):
    """Create multi-panel visualization"""
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    # Panel 1: Full signal with GT and AT events
    ax1 = axes[0]
    frames = np.arange(len(signal))
    ax1.plot(frames, signal, 'gray', linewidth=1, alpha=0.8, label='Knee Angle (MediaPipe)')
    
    # Mark AT-DTW detections (red)
    for i, start in enumerate(at_starts):
        if i == 0:
            ax1.axvline(start, color='red', linestyle='-', linewidth=2, alpha=0.7, label='AT-DTW Detection')
        else:
            ax1.axvline(start, color='red', linestyle='-', linewidth=2, alpha=0.7)
    
    # Mark estimated GT positions (green)
    for i, gt in enumerate(gt_positions):
        if i == 0:
            ax1.axvline(gt, color='green', linestyle='--', linewidth=3, alpha=0.9, label='Force Plate GT (estimated)')
        else:
            ax1.axvline(gt, color='green', linestyle='--', linewidth=3, alpha=0.9)
    
    # Shade force plate region
    fp_start = int(len(signal) * 0.2)
    fp_end = int(len(signal) * 0.8)
    ax1.axvspan(fp_start, fp_end, alpha=0.1, color='blue', label='Force Plate Region')
    
    ax1.set_xlabel('Frame', fontsize=12)
    ax1.set_ylabel('Knee Flexion (°)', fontsize=12)
    ax1.set_title(f'Subject {sid}: Force Plate GT vs AT-DTW Detection', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Zoomed view of force plate region
    ax2 = axes[1]
    zoom_signal = signal[fp_start:fp_end]
    zoom_frames = np.arange(fp_start, fp_end)
    ax2.plot(zoom_frames, zoom_signal, 'navy', linewidth=1.5)
    
    # AT-DTW in zoom region
    for start in at_starts:
        if fp_start <= start < fp_end:
            ax2.axvline(start, color='red', linestyle='-', linewidth=2.5, alpha=0.8)
    
    # GT in zoom region (all should be here)
    for gt in gt_positions:
        ax2.axvline(gt, color='green', linestyle='--', linewidth=3, alpha=0.9)
    
    ax2.set_xlabel('Frame', fontsize=12)
    ax2.set_ylabel('Knee Flexion (°)', fontsize=12)
    ax2.set_title('Zoomed: Force Plate Region Only', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Self-derived template
    ax3 = axes[2]
    if template is not None:
        ax3.plot(np.linspace(0, 100, len(template)), template, 'red', linewidth=2)
        ax3.set_xlabel('Gait Cycle (%)', fontsize=12)
        ax3.set_ylabel('Knee Flexion (°)', fontsize=12)
        ax3.set_title('Auto-Generated Template (Self-Derived)', fontsize=12)
        ax3.axhline(np.min(template), color='green', linestyle='--', alpha=0.5, label='Heel Strike (Min = 0%)')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Template Generation Failed', ha='center', va='center', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_figure(all_results, output_path):
    """Create summary grid showing all subjects"""
    n = len(all_results)
    if n == 0:
        return
    
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    
    for idx, (sid, data) in enumerate(all_results.items()):
        ax = axes[idx]
        signal = data['signal']
        at_starts = data['at_starts']
        gt_positions = data['gt_positions']
        
        # Plot signal
        ax.plot(signal, 'gray', linewidth=0.8, alpha=0.6)
        
        # AT-DTW (red)
        for s in at_starts:
            ax.axvline(s, color='red', linewidth=1.5, alpha=0.6)
        
        # GT (green)
        for g in gt_positions:
            ax.axvline(g, color='green', linestyle='--', linewidth=2, alpha=0.8)
        
        # Force plate region shading
        fp_start = int(len(signal) * 0.2)
        fp_end = int(len(signal) * 0.8)
        ax.axvspan(fp_start, fp_end, alpha=0.1, color='blue')
        
        # Calculate matching stats for title
        tolerance = 5
        matched = 0
        for g in gt_positions:
            for s in at_starts:
                if abs(g - s) <= tolerance:
                    matched += 1
                    break
        
        recall = matched / len(gt_positions) * 100 if gt_positions.size > 0 else 0
        
        ax.set_title(f'S{sid}: GT={len(gt_positions)}, AT={len(at_starts)}, Match={matched} ({recall:.0f}%)', 
                     fontsize=10)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Knee (°)')
    
    # Hide empty subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='AT-DTW Detection'),
        Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Force Plate GT'),
        plt.Rectangle((0,0), 1, 1, fc='blue', alpha=0.1, label='Force Plate Region')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=11)
    
    plt.suptitle('Force Plate GT vs AT-DTW Detection: All Subjects', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("FORCE PLATE vs AT-DTW VISUALIZATION")
    print("=" * 60)
    
    # Test subjects (those with GT data)
    test_sids = [1]  # Single subject for quick visualization
    
    all_results = {}
    
    for sid in test_sids:
        print(f"\nProcessing Subject {sid}...")
        
        # Check video exists
        video_path = f"{DATA_DIR}/{sid}/{sid}-2.mp4"
        if not os.path.exists(video_path):
            print(f"  Video not found: {video_path}")
            continue
        
        # Load GT count
        gt_count = load_gt_info(sid)
        if gt_count is None or gt_count == 0:
            print(f"  No GT data for S{sid}")
            continue
        
        print(f"  GT Strides: {gt_count}")
        
        # Run AT-DTW
        signal, at_starts, template = run_at_dtw(video_path)
        print(f"  AT-DTW Detected: {len(at_starts)} cycles")
        
        # Estimate GT positions
        gt_positions = estimate_gt_positions(len(signal), gt_count)
        
        # Store for summary
        all_results[sid] = {
            'signal': signal,
            'at_starts': at_starts,
            'gt_positions': gt_positions,
            'template': template
        }
        
        # Individual visualization
        output_path = f"{OUTPUT_DIR}/fp_vs_atdtw_S{sid}.png"
        visualize_alignment(sid, signal, at_starts, gt_positions, template, output_path)
    
    # Summary grid
    if all_results:
        summary_path = f"{OUTPUT_DIR}/fp_vs_atdtw_summary.png"
        create_summary_figure(all_results, summary_path)
    
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
