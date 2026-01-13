#!/usr/bin/env python3
"""
Debug ankle ROM at each stage of the pipeline.
"""

import numpy as np
import pandas as pd
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from improved_signal_processing import process_angles_dataframe
from kinematic_constraints import KinematicConstraintEnforcer

VIDEO_PATH = "/data/gait/data/12/12-2.mp4"

print("="*80)
print(" DEBUGGING ANKLE ROM AT EACH PIPELINE STAGE")
print("="*80)

# Stage 0: Extract baseline
print("\n[Stage 0] Extracting raw MediaPipe angles...")
extractor = MediaPipeSagittalExtractor()
landmarks, fps = extractor.extract_pose_landmarks(VIDEO_PATH)
angles_raw = extractor.calculate_joint_angles(landmarks)

# Fill NaNs
for col in angles_raw.columns:
    angles_raw[col] = angles_raw[col].ffill().bfill()

ankle_raw = angles_raw['right_ankle_angle'].values
print(f"  Raw ankle ROM: {np.ptp(ankle_raw):.1f}° (range: {ankle_raw.min():.1f}° to {ankle_raw.max():.1f}°)")

# Stage 1: Apply improved signal processing
print("\n[Stage 1] After improved signal processing...")
angles_processed, quality = process_angles_dataframe(angles_raw, fps=30)

ankle_processed = angles_processed['right_ankle_angle'].values
print(f"  Processed ankle ROM: {np.ptp(ankle_processed):.1f}° (range: {ankle_processed.min():.1f}° to {ankle_processed.max():.1f}°)")
print(f"  ROM preservation: {np.ptp(ankle_processed)/np.ptp(ankle_raw)*100:.1f}%")

# Stage 2: Apply kinematic constraints
print("\n[Stage 2] After kinematic constraints (fixed version)...")
enforcer = KinematicConstraintEnforcer(fps=30, subject_height=1.70)
angles_constrained = enforcer.enforce_joint_angle_constraints(angles_processed)

ankle_constrained = angles_constrained['right_ankle_angle'].values
print(f"  Constrained ankle ROM: {np.ptp(ankle_constrained):.1f}° (range: {ankle_constrained.min():.1f}° to {ankle_constrained.max():.1f}°)")
print(f"  ROM preservation: {np.ptp(ankle_constrained)/np.ptp(ankle_raw)*100:.1f}%")

# Identify where the loss occurs
print("\n" + "="*80)
print(" ROM LOSS ANALYSIS")
print("="*80)

loss_stage1 = np.ptp(ankle_raw) - np.ptp(ankle_processed)
loss_stage2 = np.ptp(ankle_processed) - np.ptp(ankle_constrained)

print(f"\nROM loss in Stage 1 (signal processing): {loss_stage1:.1f}° ({loss_stage1/np.ptp(ankle_raw)*100:.1f}%)")
print(f"ROM loss in Stage 2 (kinematic constraints): {loss_stage2:.1f}° ({loss_stage2/np.ptp(ankle_raw)*100:.1f}%)")

if loss_stage1 > loss_stage2:
    print("\n⚠️  PRIMARY ISSUE: Signal processing stage is over-smoothing")
    print("    Action: Further reduce filtering aggressiveness in improved_signal_processing.py")
else:
    print("\n⚠️  PRIMARY ISSUE: Kinematic constraints stage is over-constraining")
    print("    Action: Further relax constraints in kinematic_constraints.py")

# Show actual values
print("\n" + "="*80)
print(" SAMPLE VALUES (first 10 frames)")
print("="*80)
print("Frame | Raw     | Processed | Constrained")
print("------+---------+-----------+------------")
for i in range(min(10, len(ankle_raw))):
    print(f"{i:5d} | {ankle_raw[i]:7.2f} | {ankle_processed[i]:9.2f} | {ankle_constrained[i]:11.2f}")

print("\n" + "="*80)
