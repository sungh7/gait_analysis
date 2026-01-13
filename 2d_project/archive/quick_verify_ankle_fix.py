#!/usr/bin/env python3
"""
Quick verification of ankle fix on 3 subjects.
"""

import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

from sagittal_extractor_2d import MediaPipeSagittalExtractor
from improved_signal_processing import process_angles_dataframe
from kinematic_constraints import KinematicConstraintEnforcer

print("="*80)
print(" QUICK ANKLE FIX VERIFICATION (3 Subjects)")
print("="*80)

# Test on 3 diverse subjects
test_subjects = [
    ("S1_01", "/data/gait/data/1/1-2.mp4"),
    ("S1_12", "/data/gait/data/12/12-2.mp4"),
    ("S1_20", "/data/gait/data/20/20-2.mp4")
]

extractor = MediaPipeSagittalExtractor()
enforcer = KinematicConstraintEnforcer(fps=30, subject_height=1.70)

results_summary = []

for subject, video_path in test_subjects:
    print(f"\n{'='*80}")
    print(f"Processing {subject}")
    print('='*80)

    try:
        # Extract and process
        print(f"  [1/3] Extracting landmarks...")
        landmarks, fps = extractor.extract_pose_landmarks(video_path)
        angles_raw = extractor.calculate_joint_angles(landmarks)

        # Fill NaNs
        for col in angles_raw.columns:
            angles_raw[col] = angles_raw[col].ffill().bfill()

        print(f"  [2/3] Applying improved processing...")
        angles_processed, quality = process_angles_dataframe(angles_raw, fps=30)

        print(f"  [3/3] Applying kinematic constraints...")
        angles_final = enforcer.enforce_joint_angle_constraints(angles_processed)

        # Check ankle ROM
        ankle_raw_rom = np.ptp(angles_raw['right_ankle_angle'].values)
        ankle_final_rom = np.ptp(angles_final['right_ankle_angle'].values)
        ankle_preservation = ankle_final_rom / ankle_raw_rom * 100 if ankle_raw_rom > 0 else 0

        # Check knee ROM (should still be good)
        knee_raw_rom = np.ptp(angles_raw['right_knee_angle'].values)
        knee_final_rom = np.ptp(angles_final['right_knee_angle'].values)

        # Jerk reduction
        ankle_jerk_raw = np.std(np.gradient(np.gradient(angles_raw['right_ankle_angle'].values)))
        ankle_jerk_final = np.std(np.gradient(np.gradient(angles_final['right_ankle_angle'].values)))
        ankle_jerk_reduction = (ankle_jerk_raw - ankle_jerk_final) / ankle_jerk_raw * 100

        print(f"\n  Results:")
        print(f"    Ankle ROM: {ankle_raw_rom:.1f}° → {ankle_final_rom:.1f}° ({ankle_preservation:.1f}% preserved)")
        print(f"    Ankle jerk reduction: {ankle_jerk_reduction:.1f}%")
        print(f"    Knee ROM: {knee_raw_rom:.1f}° → {knee_final_rom:.1f}°")

        success = ankle_preservation > 50  # At least 50% ROM preserved

        if success:
            print(f"    ✅ SUCCESS")
        else:
            print(f"    ⚠️  WARNING: Low ankle ROM preservation")

        results_summary.append({
            'subject': subject,
            'ankle_preservation': ankle_preservation,
            'ankle_final_rom': ankle_final_rom,
            'success': success
        })

    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        results_summary.append({
            'subject': subject,
            'ankle_preservation': 0,
            'ankle_final_rom': 0,
            'success': False
        })

# Summary
print(f"\n{'='*80}")
print(" SUMMARY")
print('='*80)

successes = sum([r['success'] for r in results_summary])
avg_preservation = np.mean([r['ankle_preservation'] for r in results_summary])

print(f"\nSuccessful subjects: {successes}/{len(test_subjects)}")
print(f"Average ankle ROM preservation: {avg_preservation:.1f}%")

if successes == len(test_subjects) and avg_preservation > 60:
    print("\n✅ ✅ ✅  ANKLE FIX VERIFIED!  ✅ ✅ ✅")
    print("\nThe fix works across multiple subjects.")
    print("Ready for full batch processing.")
else:
    print("\n⚠️  Some issues detected. Review individual results above.")

print('='*80)
