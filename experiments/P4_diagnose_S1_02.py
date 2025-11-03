"""
Phase 4: Diagnose S1_02 Catastrophic Cadence Failure

S1_02 shows 60 steps/min error:
- GT cadence: 113.4 steps/min
- Predicted cadence: 136.5 steps/min (RANSAC)
- Right leg: 175.6 steps/min (massive overestimation)
- Left leg: 97.3 steps/min (underestimation)

This script investigates:
1. Video quality and turn detection
2. Template extraction quality
3. Heel strike timing distribution
4. RANSAC inlier/outlier patterns
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from mediapipe_csv_processor import MediaPipeCSVProcessor

def load_ground_truth():
    """Load ground truth data"""
    with open('ground_truth_gait_metrics.json', 'r') as f:
        return json.load(f)


def calculate_leg_angle(hip, knee, ankle, heel):
    """Calculate composite leg angle from hip-knee-ankle-heel."""
    v1 = knee - hip
    v2 = ankle - knee
    v3 = heel - ankle

    # Hip-knee angle
    cos_angle1 = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    angle1 = np.degrees(np.arccos(np.clip(cos_angle1, -1.0, 1.0)))

    # Knee-ankle angle
    cos_angle2 = np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3) + 1e-6)
    angle2 = np.degrees(np.arccos(np.clip(cos_angle2, -1.0, 1.0)))

    # Composite angle
    return (angle1 + angle2) / 2


def detect_strikes_template(signal, template, threshold=0.7, side='left'):
    """Template-based strike detection using FastDTW"""
    strikes = []
    window_size = len(template)

    for i in range(len(signal) - window_size + 1):
        window = signal[i:i+window_size]
        distance, _ = fastdtw(window, template, dist=euclidean)

        # Normalize distance by template length
        normalized_dist = distance / len(template)

        # Similarity score (1 - normalized distance)
        similarity = 1.0 / (1.0 + normalized_dist)

        if similarity >= threshold:
            # Check if this is a new peak (not too close to previous)
            if not strikes or (i - strikes[-1]) > window_size // 2:
                strikes.append(i)

    return strikes


def ransac_period_consensus(stride_times, tolerance=0.3, iterations=100):
    """RANSAC to find consensus stride period"""
    if len(stride_times) < 2:
        return np.mean(stride_times) if len(stride_times) > 0 else 0, np.ones(len(stride_times), dtype=bool)

    best_inliers = []
    best_period = 0

    for _ in range(iterations):
        # Sample random period
        idx = np.random.randint(0, len(stride_times))
        candidate_period = stride_times[idx]

        # Find inliers
        inliers = np.abs(stride_times - candidate_period) <= tolerance

        if np.sum(inliers) > len(best_inliers):
            best_inliers = inliers
            best_period = np.mean(stride_times[inliers])

    return best_period, best_inliers


def analyze_heel_strikes(subject_id='S1_02'):
    """Analyze heel strike detection for S1_02"""
    print(f"="*80)
    print(f"S1_02 Diagnostic Report")
    print(f"="*80)
    print()

    # Load pose data
    pose_file = Path('data/2/2-2_side_pose_fps30.csv')
    processor = MediaPipeCSVProcessor()
    df_wide = processor.load_csv(str(pose_file))
    print(f"✓ Loaded {len(df_wide)} frames from {pose_file}")

    # Calculate angles using the processor
    df_angles = processor.calculate_joint_angles(df_wide)
    angles_left = df_angles['leg_angle_left'].values
    angles_right = df_angles['leg_angle_right'].values

    print(f"✓ Calculated angles for {len(angles_left)} frames")
    print()

    # Load ground truth
    gt = load_ground_truth()
    gt_entry = gt['S1_02']
    print(f"Ground Truth:")
    print(f"  Left:  {gt_entry['step_left']} steps, {gt_entry['cadence_left']:.1f} steps/min")
    print(f"  Right: {gt_entry['step_right']} steps, {gt_entry['cadence_right']:.1f} steps/min")
    print(f"  Avg:   {gt_entry['cadence_avg']:.1f} steps/min")
    print()

    # Extract template (first 3 gait cycles)
    print("="*80)
    print("Template Extraction")
    print("="*80)

    # Detect peaks in angles to find template cycles
    peaks_l, _ = find_peaks(angles_left, prominence=0.3, distance=15)
    peaks_r, _ = find_peaks(angles_right, prominence=0.3, distance=15)

    print(f"Initial peaks (prominence=0.3, distance=15):")
    print(f"  Left:  {len(peaks_l)} peaks")
    print(f"  Right: {len(peaks_r)} peaks")
    print()

    if len(peaks_l) < 3 or len(peaks_r) < 3:
        print("⚠ WARNING: Not enough peaks for template extraction!")
        return

    # Extract templates (first 3 cycles)
    template_l = angles_left[peaks_l[0]:peaks_l[2]+1]
    template_r = angles_right[peaks_r[0]:peaks_r[2]+1]

    print(f"Template quality:")
    print(f"  Left:  {len(template_l)} frames, span: {peaks_l[0]}-{peaks_l[2]}")
    print(f"  Right: {len(template_r)} frames, span: {peaks_r[0]}-{peaks_r[2]}")
    print()

    # Template-based detection with threshold=0.7
    print("="*80)
    print("Template-Based Detection (threshold=0.7)")
    print("="*80)

    strikes_l = detect_strikes_template(
        angles_left, template_l, threshold=0.7, side='left'
    )
    strikes_r = detect_strikes_template(
        angles_right, template_r, threshold=0.7, side='right'
    )

    print(f"Detected strikes:")
    print(f"  Left:  {len(strikes_l)} strikes (ratio: {len(strikes_l)/gt_entry['step_left']:.2f}×)")
    print(f"  Right: {len(strikes_r)} strikes (ratio: {len(strikes_r)/gt_entry['step_right']:.2f}×)")
    print()

    if len(strikes_l) > 0:
        print(f"Left strike frames: {strikes_l[:10]}{'...' if len(strikes_l) > 10 else ''}")
    if len(strikes_r) > 0:
        print(f"Right strike frames: {strikes_r[:10]}{'...' if len(strikes_r) > 10 else ''}")
    print()

    # Analyze strike timing
    print("="*80)
    print("Strike Timing Analysis")
    print("="*80)

    if len(strikes_l) > 1:
        intervals_l = np.diff(strikes_l)
        print(f"Left intervals (frames between strikes):")
        print(f"  Mean: {np.mean(intervals_l):.1f}, Std: {np.std(intervals_l):.1f}")
        print(f"  Min:  {np.min(intervals_l)}, Max: {np.max(intervals_l)}")
        print(f"  Median: {np.median(intervals_l):.1f}")

    if len(strikes_r) > 1:
        intervals_r = np.diff(strikes_r)
        print(f"Right intervals (frames between strikes):")
        print(f"  Mean: {np.mean(intervals_r):.1f}, Std: {np.std(intervals_r):.1f}")
        print(f"  Min:  {np.min(intervals_r)}, Max: {np.max(intervals_r)}")
        print(f"  Median: {np.median(intervals_r):.1f}")
    print()

    # RANSAC analysis
    print("="*80)
    print("RANSAC Cadence Estimation")
    print("="*80)

    fps = 30
    video_duration = len(df) / fps

    if len(strikes_l) > 1:
        intervals_l = np.diff(strikes_l)
        stride_times_l = intervals_l / fps

        # RANSAC
        best_period_l, inliers_l = ransac_period_consensus(
            stride_times_l, tolerance=0.3, iterations=100
        )
        cadence_l = 60 / best_period_l if best_period_l > 0 else 0

        print(f"Left leg RANSAC:")
        print(f"  Best period: {best_period_l:.3f}s")
        print(f"  Cadence: {cadence_l:.1f} steps/min (GT: {gt_entry['cadence_left']:.1f})")
        print(f"  Error: {cadence_l - gt_entry['cadence_left']:.1f} steps/min")
        print(f"  Inliers: {np.sum(inliers_l)}/{len(inliers_l)} ({100*np.mean(inliers_l):.1f}%)")

        if len(stride_times_l) <= 20:
            print(f"  All intervals (s): {stride_times_l.round(3)}")
            print(f"  Inlier mask: {inliers_l}")
        else:
            print(f"  Sample intervals (s): {stride_times_l[:10].round(3)}")
            print(f"  Sample inliers: {inliers_l[:10]}")
        print()

    if len(strikes_r) > 1:
        intervals_r = np.diff(strikes_r)
        stride_times_r = intervals_r / fps

        # RANSAC
        best_period_r, inliers_r = ransac_period_consensus(
            stride_times_r, tolerance=0.3, iterations=100
        )
        cadence_r = 60 / best_period_r if best_period_r > 0 else 0

        print(f"Right leg RANSAC:")
        print(f"  Best period: {best_period_r:.3f}s")
        print(f"  Cadence: {cadence_r:.1f} steps/min (GT: {gt_entry['cadence_right']:.1f})")
        print(f"  Error: {cadence_r - gt_entry['cadence_right']:.1f} steps/min")
        print(f"  Inliers: {np.sum(inliers_r)}/{len(inliers_r)} ({100*np.mean(inliers_r):.1f}%)")

        if len(stride_times_r) <= 20:
            print(f"  All intervals (s): {stride_times_r.round(3)}")
            print(f"  Inlier mask: {inliers_r}")
        else:
            print(f"  Sample intervals (s): {stride_times_r[:10].round(3)}")
            print(f"  Sample inliers: {inliers_r[:10]}")
        print()

    # Turn detection analysis
    print("="*80)
    print("Turn Detection Analysis")
    print("="*80)

    # Simple turn detection based on X-coordinate direction changes
    hip_x = df_wide['LEFT_HIP_x'].values
    direction = np.diff(hip_x)
    turns = []

    # Find significant direction changes
    for i in range(1, len(direction) - 1):
        if direction[i-1] * direction[i+1] < 0:  # Sign change
            if abs(direction[i-1]) > 0.01 and abs(direction[i+1]) > 0.01:
                turns.append(i)

    print(f"Detected {len(turns)} turn points: {turns[:10]}")

    if len(turns) > 0:
        print(f"Turn frame examples:")
        for i, turn in enumerate(turns[:5]):
            print(f"  Turn {i+1}: frame {turn}")
    else:
        print("No turns detected - walking straight")
    print()

    # Visualization
    print("="*80)
    print("Generating Diagnostic Plots")
    print("="*80)

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # Plot 1: Angle signals with template regions
    ax = axes[0]
    frames = np.arange(len(angles_left))
    ax.plot(frames, angles_left, 'b-', alpha=0.7, label='Left leg angle')
    ax.plot(frames, angles_right, 'r-', alpha=0.7, label='Right leg angle')

    # Mark template regions
    ax.axvspan(peaks_l[0], peaks_l[2], alpha=0.2, color='blue', label='Left template')
    ax.axvspan(peaks_r[0], peaks_r[2], alpha=0.2, color='red', label='Right template')

    # Mark detected strikes
    if len(strikes_l) > 0:
        ax.scatter(strikes_l, [angles_left[s] for s in strikes_l],
                  color='blue', s=100, marker='v', zorder=5, label=f'Left strikes (n={len(strikes_l)})')
    if len(strikes_r) > 0:
        ax.scatter(strikes_r, [angles_right[s] for s in strikes_r],
                  color='red', s=100, marker='^', zorder=5, label=f'Right strikes (n={len(strikes_r)})')

    # Mark turns
    for turn in turns:
        ax.axvline(turn, color='orange', linestyle='--', alpha=0.7)

    ax.set_xlabel('Frame')
    ax.set_ylabel('Leg Angle (degrees)')
    ax.set_title(f'S1_02: Heel Strike Detection (GT: L={gt_entry["step_left"]}, R={gt_entry["step_right"]})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Stride intervals histogram
    ax = axes[1]
    if len(strikes_l) > 1 and len(strikes_r) > 1:
        stride_times_l = np.diff(strikes_l) / fps
        stride_times_r = np.diff(strikes_r) / fps

        ax.hist(stride_times_l, bins=20, alpha=0.5, color='blue', label=f'Left (n={len(stride_times_l)})')
        ax.hist(stride_times_r, bins=20, alpha=0.5, color='red', label=f'Right (n={len(stride_times_r)})')

        # Mark expected stride time
        expected_period_l = 60 / gt_entry['cadence_left']
        expected_period_r = 60 / gt_entry['cadence_right']
        ax.axvline(expected_period_l, color='blue', linestyle='--', linewidth=2, label=f'GT left: {expected_period_l:.2f}s')
        ax.axvline(expected_period_r, color='red', linestyle='--', linewidth=2, label=f'GT right: {expected_period_r:.2f}s')

        ax.set_xlabel('Stride Time (seconds)')
        ax.set_ylabel('Count')
        ax.set_title('S1_02: Stride Time Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 3: RANSAC inlier visualization
    ax = axes[2]
    if len(strikes_l) > 1:
        stride_times_l = np.diff(strikes_l) / fps
        best_period_l, inliers_l = ransac_period_consensus(stride_times_l, tolerance=0.3)

        colors_l = ['green' if inlier else 'lightgray' for inlier in inliers_l]
        ax.scatter(range(len(stride_times_l)), stride_times_l, c=colors_l, s=100, alpha=0.7, label='Left')
        ax.axhline(best_period_l, color='blue', linestyle='--', linewidth=2, label=f'RANSAC left: {best_period_l:.2f}s')

    if len(strikes_r) > 1:
        stride_times_r = np.diff(strikes_r) / fps
        best_period_r, inliers_r = ransac_period_consensus(stride_times_r, tolerance=0.3)

        colors_r = ['darkred' if inlier else 'lightcoral' for inlier in inliers_r]
        offset = len(stride_times_l) if len(strikes_l) > 1 else 0
        ax.scatter(range(offset, offset + len(stride_times_r)), stride_times_r,
                  c=colors_r, s=100, alpha=0.7, marker='s', label='Right')
        ax.axhline(best_period_r, color='red', linestyle='--', linewidth=2, label=f'RANSAC right: {best_period_r:.2f}s')

    ax.set_xlabel('Strike Interval Index')
    ax.set_ylabel('Stride Time (seconds)')
    ax.set_title('S1_02: RANSAC Inlier/Outlier Classification')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('P4_S1_02_diagnostic_plot.png', dpi=150, bbox_inches='tight')
    print("✓ Saved diagnostic plot: P4_S1_02_diagnostic_plot.png")
    print()

    # Summary
    print("="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)
    print()
    print("Likely causes of catastrophic error:")

    if len(strikes_r) / gt_entry['step_right'] > 1.5:
        print("✗ RIGHT LEG OVERDETECTION: Detecting false positive heel strikes")
        print("  → Template may be too permissive (threshold=0.7)")
        print("  → Consider inspecting video for unusual gait pattern")

    if len(strikes_l) / gt_entry['step_left'] < 0.8:
        print("✗ LEFT LEG UNDERDETECTION: Missing real heel strikes")
        print("  → Template may be too strict (threshold=0.7)")

    if len(turns) > 2:
        print(f"⚠ MULTIPLE TURNS: {len(turns)} turns detected")
        print("  → Turn filtering may affect stride selection")

    if len(strikes_r) > 1:
        stride_times_r = np.diff(strikes_r) / fps
        if np.std(stride_times_r) > 0.3:
            print(f"⚠ HIGH STRIDE VARIABILITY (Right): std={np.std(stride_times_r):.3f}s")
            print("  → Inconsistent stride timing may confuse RANSAC")

    print()


if __name__ == '__main__':
    analyze_heel_strikes('S1_02')
