#!/usr/bin/env python3
"""
Test ankle parameter fix on single subject.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from improved_signal_processing import process_angles_dataframe
from kinematic_constraints import KinematicConstraintEnforcer

print("="*80)
print(" TESTING ANKLE PARAMETER FIX")
print("="*80)

# Test on S1_12 (previously had some ankle ROM preserved: 178.9° → 20.2°)
# Should now preserve even more
VIDEO_PATH = "/data/gait/data/12/12-2.mp4"

print(f"\nProcessing: {VIDEO_PATH}")
print("Expected: Ankle ROM should be preserved (not → 0)")

# Extract baseline
print("\n[1/3] Extracting baseline angles...")
extractor = MediaPipeSagittalExtractor()
landmarks, fps = extractor.extract_pose_landmarks(VIDEO_PATH)
angles_baseline = extractor.calculate_joint_angles(landmarks)

# Fill NaNs
for col in angles_baseline.columns:
    angles_baseline[col] = angles_baseline[col].ffill().bfill()

print(f"  ✓ Extracted {len(angles_baseline)} frames")

# Apply improved processing
print("\n[2/3] Applying improved processing with FIXED ankle parameters...")
angles_improved, quality = process_angles_dataframe(angles_baseline, fps=30)

# Apply constraints
print("\n[3/3] Applying kinematic constraints...")
enforcer = KinematicConstraintEnforcer(fps=30, subject_height=1.70)
angles_final = enforcer.enforce_joint_angle_constraints(angles_improved)

# Check ankle ROM
print("\n" + "="*80)
print(" ANKLE ROM CHECK")
print("="*80)

ankle_baseline_rom = np.ptp(angles_baseline['right_ankle_angle'].values)
ankle_final_rom = np.ptp(angles_final['right_ankle_angle'].values)

print(f"\nBaseline ankle ROM: {ankle_baseline_rom:.1f}°")
print(f"Improved ankle ROM: {ankle_final_rom:.1f}°")
print(f"Preservation: {ankle_final_rom/ankle_baseline_rom*100:.1f}%")

if ankle_final_rom > 10:
    print("\n✅ SUCCESS: Ankle ROM preserved! (>10°)")
    ankle_preserved = True
else:
    print("\n⚠️  WARNING: Ankle ROM still too small (<10°)")
    ankle_preserved = False

# Check all joints
print("\n" + "="*80)
print(" ALL JOINTS SUMMARY")
print("="*80)

joints = ['right_knee_angle', 'right_hip_angle', 'right_ankle_angle']
joint_labels = ['Knee', 'Hip', 'Ankle']

for joint, label in zip(joints, joint_labels):
    baseline_signal = angles_baseline[joint].values
    final_signal = angles_final[joint].values

    baseline_jerk = np.std(np.gradient(np.gradient(baseline_signal)))
    final_jerk = np.std(np.gradient(np.gradient(final_signal)))
    jerk_reduction = (baseline_jerk - final_jerk) / baseline_jerk * 100

    baseline_rom = np.ptp(baseline_signal)
    final_rom = np.ptp(final_signal)

    q = quality[joint]

    print(f"\n{label}:")
    print(f"  Jerk reduction: {jerk_reduction:.1f}%")
    print(f"  ROM: {baseline_rom:.1f}° → {final_rom:.1f}°")
    print(f"  Quality: {q['overall_quality']:.2f}/1.0")
    print(f"  SNR: {q['snr']:.1f} dB")

# Visualize
print("\n" + "="*80)
print(" GENERATING COMPARISON PLOT")
print("="*80)

fig, axes = plt.subplots(3, 1, figsize=(14, 9))

for ax, joint, label in zip(axes, joints, joint_labels):
    ax.plot(angles_baseline[joint], 'lightgray', alpha=0.5, linewidth=1, label='Baseline')
    ax.plot(angles_final[joint], 'black', linewidth=2, label='Improved (Fixed Parameters)')

    # Show ROM
    baseline_rom = np.ptp(angles_baseline[joint].values)
    final_rom = np.ptp(angles_final[joint].values)

    ax.set_title(f"{label} - ROM: {baseline_rom:.1f}° → {final_rom:.1f}°",
                fontsize=12, fontweight='bold')
    ax.set_ylabel('Angle (deg)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Highlight ankle preservation
    if label == 'Ankle':
        if ankle_preserved:
            ax.text(0.5, 0.95, '✅ ROM PRESERVED!',
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=14, fontweight='bold', color='green',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            ax.text(0.5, 0.95, '⚠️ STILL TOO FLAT',
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=14, fontweight='bold', color='red',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

axes[-1].set_xlabel('Frame')
plt.tight_layout()
plt.savefig('/data/gait/2d_project/ankle_fix_test_S1_12.png', dpi=150)
print("  ✓ Saved: ankle_fix_test_S1_12.png")

print("\n" + "="*80)
print(" TEST COMPLETE")
print("="*80)

if ankle_preserved:
    print("\n✅ ✅ ✅  ANKLE FIX SUCCESSFUL!  ✅ ✅ ✅")
    print("\nRecommendation: Re-run batch processing on all subjects")
    print("Command: python batch_process_all_subjects.py")
else:
    print("\n⚠️  Ankle fix needs further adjustment")
    print("Try increasing sigma further: fps/10 → fps/8 (4Hz cutoff)")

print("="*80)
