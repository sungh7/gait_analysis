#!/usr/bin/env python3
"""
Batch Processing: All 26 Subjects with Improvements

Processes all subjects through improved pipeline and compares with baseline.
Generates comprehensive reports for research paper update.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from improved_signal_processing import process_angles_dataframe
from kinematic_constraints import KinematicConstraintEnforcer
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" BATCH PROCESSING: ALL 26 SUBJECTS WITH IMPROVEMENTS")
print("="*80)

# Configuration
SUBJECTS = [f"S1_{i:02d}" for i in range(1, 27)]
DATA_DIR = Path("/data/gait/data")
VICON_DIR = Path("/data/gait/data/processed_new")
OUTPUT_DIR = Path("/data/gait/2d_project/batch_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize components
extractor = MediaPipeSagittalExtractor()
enforcer = KinematicConstraintEnforcer(fps=30, subject_height=1.70)

# Results storage
results = []
failed = []

print(f"\nProcessing {len(SUBJECTS)} subjects...")
print(f"Output directory: {OUTPUT_DIR}")

for subject_idx, subject in enumerate(SUBJECTS, 1):
    print(f"\n{'='*80}")
    print(f"[{subject_idx}/{len(SUBJECTS)}] Processing {subject}")
    print('='*80)

    # Find video file
    subject_num = int(subject.split('_')[1])
    video_path = DATA_DIR / str(subject_num) / f"{subject_num}-2.mp4"

    if not video_path.exists():
        print(f"⚠️  Video not found: {video_path}")
        failed.append({'subject': subject, 'reason': 'Video not found'})
        continue

    try:
        # =================================================================
        # STAGE 1: BASELINE PROCESSING
        # =================================================================
        print("\n[1/4] Baseline processing...")
        landmarks, fps_detected = extractor.extract_pose_landmarks(str(video_path))
        angles_baseline = extractor.calculate_joint_angles(landmarks)

        # Simple forward-fill (old method)
        for col in angles_baseline.columns:
            angles_baseline[col] = angles_baseline[col].ffill().bfill()

        print(f"  ✓ Extracted {len(angles_baseline)} frames")

        # =================================================================
        # STAGE 2: IMPROVED PROCESSING
        # =================================================================
        print("\n[2/4] Improved signal processing...")
        angles_improved, quality_report = process_angles_dataframe(angles_baseline, fps=30)

        avg_snr = np.mean([q['snr'] for q in quality_report.values()])
        avg_smoothness = np.mean([q['smoothness'] for q in quality_report.values()])
        avg_quality = np.mean([q['overall_quality'] for q in quality_report.values()])

        print(f"  ✓ Average SNR: {avg_snr:.1f} dB")
        print(f"  ✓ Average quality: {avg_quality:.2f}/1.0")

        # =================================================================
        # STAGE 3: KINEMATIC CONSTRAINTS
        # =================================================================
        print("\n[3/4] Applying kinematic constraints...")
        violations_before = enforcer.validate_angles(angles_improved)
        angles_final = enforcer.enforce_joint_angle_constraints(angles_improved)
        violations_after = enforcer.validate_angles(angles_final)

        total_violations_fixed = sum([
            (v['angle_violations'] + v['velocity_violations'] + v['acceleration_violations'])
            for v in violations_before.values()
        ]) - sum([
            (v['angle_violations'] + v['velocity_violations'] + v['acceleration_violations'])
            for v in violations_after.values()
        ])

        print(f"  ✓ Fixed violations: {total_violations_fixed}")

        # =================================================================
        # STAGE 4: QUALITY METRICS
        # =================================================================
        print("\n[4/4] Computing quality metrics...")

        joints = ['right_knee_angle', 'right_hip_angle', 'right_ankle_angle']
        joint_metrics = {}

        for joint in joints:
            if joint not in angles_baseline.columns or joint not in angles_final.columns:
                continue

            # Compute jerk (smoothness metric)
            baseline_signal = angles_baseline[joint].values
            final_signal = angles_final[joint].values

            baseline_jerk = np.std(np.gradient(np.gradient(baseline_signal)))
            final_jerk = np.std(np.gradient(np.gradient(final_signal)))
            jerk_reduction = (baseline_jerk - final_jerk) / baseline_jerk * 100

            # Range of motion
            baseline_rom = np.ptp(baseline_signal)
            final_rom = np.ptp(final_signal)

            joint_metrics[joint] = {
                'baseline_jerk': baseline_jerk,
                'final_jerk': final_jerk,
                'jerk_reduction_pct': jerk_reduction,
                'baseline_rom': baseline_rom,
                'final_rom': final_rom,
                'snr': quality_report[joint]['snr'],
                'quality_score': quality_report[joint]['overall_quality']
            }

            print(f"  {joint}: {jerk_reduction:.1f}% jerk reduction")

        # =================================================================
        # ATTEMPT VICON CORRELATION (if available)
        # =================================================================
        vicon_correlations = {}
        vicon_file = VICON_DIR / f"{subject}_gait_long.csv"

        if vicon_file.exists():
            print(f"\n[Bonus] Computing Vicon correlations...")
            try:
                vicon_df = pd.read_csv(vicon_file)

                # Try to extract Vicon angles for comparison
                for joint in joints:
                    joint_name = joint.split('_')[1]  # knee, hip, ankle

                    # Find matching Vicon column (naming convention varies)
                    vicon_cols = [col for col in vicon_df.columns if joint_name in col.lower()]

                    if vicon_cols:
                        vicon_signal = vicon_df[vicon_cols[0]].dropna().values

                        # Resample to match lengths (simple approach)
                        min_len = min(len(baseline_signal), len(final_signal), len(vicon_signal))

                        if min_len > 50:  # Need reasonable amount of data
                            baseline_corr, _ = pearsonr(baseline_signal[:min_len], vicon_signal[:min_len])
                            final_corr, _ = pearsonr(final_signal[:min_len], vicon_signal[:min_len])

                            vicon_correlations[joint] = {
                                'baseline_r': baseline_corr,
                                'improved_r': final_corr,
                                'improvement': final_corr - baseline_corr
                            }

                            print(f"  {joint}: r = {baseline_corr:.3f} → {final_corr:.3f} (+{final_corr - baseline_corr:+.3f})")

            except Exception as e:
                print(f"  ⚠️  Could not process Vicon data: {e}")

        # =================================================================
        # SAVE RESULTS
        # =================================================================
        subject_result = {
            'subject': subject,
            'frames': len(angles_baseline),
            'fps': fps_detected or 30,
            'avg_snr': avg_snr,
            'avg_smoothness': avg_smoothness,
            'avg_quality': avg_quality,
            'violations_fixed': total_violations_fixed,
            'joint_metrics': joint_metrics,
            'vicon_correlations': vicon_correlations,
            'status': 'SUCCESS'
        }

        results.append(subject_result)

        # Save individual plots
        fig, axes = plt.subplots(3, 1, figsize=(14, 9))

        for ax, joint, label in zip(axes, joints, ['Knee', 'Hip', 'Ankle']):
            if joint in angles_baseline.columns and joint in angles_final.columns:
                ax.plot(angles_baseline[joint], 'lightgray', alpha=0.5, linewidth=1, label='Baseline')
                ax.plot(angles_final[joint], 'black', linewidth=2, label='Improved')
                ax.set_title(f"{subject} - {label}", fontweight='bold')
                ax.set_ylabel('Angle (deg)')
                ax.legend()
                ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Frame')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{subject}_comparison.png", dpi=100)
        plt.close()

        # Save CSV
        angles_final.to_csv(OUTPUT_DIR / f"{subject}_improved_angles.csv", index=False)

        print(f"\n✓ {subject} complete!")

    except Exception as e:
        print(f"\n✗ {subject} FAILED: {e}")
        failed.append({'subject': subject, 'reason': str(e)})
        import traceback
        traceback.print_exc()

# =================================================================
# GENERATE SUMMARY REPORT
# =================================================================
print("\n" + "="*80)
print(" GENERATING SUMMARY REPORT")
print("="*80)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / 'batch_results_summary.csv', index=False)

print(f"\n✓ Processed: {len(results)}/{len(SUBJECTS)} subjects")
print(f"✗ Failed: {len(failed)}/{len(SUBJECTS)} subjects")

if len(results) > 0:
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Average SNR: {results_df['avg_snr'].mean():.1f} dB")
    print(f"   Average Quality: {results_df['avg_quality'].mean():.2f}/1.0")
    print(f"   Total Violations Fixed: {results_df['violations_fixed'].sum()}")

    # Extract joint-specific improvements
    all_jerk_reductions = []
    for result in results:
        for joint, metrics in result['joint_metrics'].items():
            all_jerk_reductions.append(metrics['jerk_reduction_pct'])

    print(f"   Average Jerk Reduction: {np.mean(all_jerk_reductions):.1f}%")

    # Vicon correlation improvements
    vicon_improvements = []
    for result in results:
        for joint, corr_data in result.get('vicon_correlations', {}).items():
            vicon_improvements.append(corr_data['improvement'])

    if vicon_improvements:
        print(f"\n📈 VICON CORRELATION IMPROVEMENTS:")
        print(f"   Average improvement: {np.mean(vicon_improvements):+.3f}")
        print(f"   Subjects improved: {sum(1 for x in vicon_improvements if x > 0)}/{len(vicon_improvements)}")

# Save detailed JSON report
with open(OUTPUT_DIR / 'detailed_results.json', 'w') as f:
    json.dump({
        'results': results,
        'failed': failed,
        'summary': {
            'processed': len(results),
            'failed': len(failed),
            'total': len(SUBJECTS)
        }
    }, f, indent=2)

print(f"\n📁 Results saved to: {OUTPUT_DIR}/")
print(f"   - batch_results_summary.csv")
print(f"   - detailed_results.json")
print(f"   - Individual plots: S1_XX_comparison.png")
print(f"   - Improved angles: S1_XX_improved_angles.csv")

print("\n" + "="*80)
print("✓ BATCH PROCESSING COMPLETE")
print("="*80)
