#!/usr/bin/env python3
"""
Integrated Improved Gait Analysis Pipeline.

Combines all accuracy improvements:
1. Enhanced signal processing
2. Landmark quality filtering
3. Kinematic constraints
4. Comprehensive quality metrics

Expected improvement: r=0.50 → r=0.80+ for problematic subjects
"""

import numpy as np
import pandas as pd
from sagittal_extractor_2d import MediaPipeSagittalExtractor
from improved_signal_processing import ImprovedSignalProcessor, process_angles_dataframe
from landmark_quality_filter import LandmarkQualityFilter
from kinematic_constraints import KinematicConstraintEnforcer
from typing import Dict, Tuple
import matplotlib.pyplot as plt


class ImprovedGaitAnalysisPipeline:
    """
    Enhanced gait analysis pipeline with all accuracy improvements.
    """

    def __init__(self, fps=30, subject_height=1.70):
        """
        Args:
            fps: Video frame rate
            subject_height: Subject height in meters (for biomechanical constraints)
        """
        self.fps = fps
        self.subject_height = subject_height

        # Initialize components
        self.base_extractor = MediaPipeSagittalExtractor()
        self.signal_processor = ImprovedSignalProcessor(fps=fps)
        self.landmark_qc = LandmarkQualityFilter()
        self.constraint_enforcer = KinematicConstraintEnforcer(fps=fps, subject_height=subject_height)

    def process_video(
        self,
        video_path: str,
        enable_landmark_qc: bool = True,
        enable_improved_filtering: bool = True,
        enable_kinematic_constraints: bool = True,
        visualize: bool = False
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Process video through improved pipeline.

        Args:
            video_path: Path to video file
            enable_landmark_qc: Apply landmark quality filtering
            enable_improved_filtering: Apply improved signal processing
            enable_kinematic_constraints: Apply biomechanical constraints
            visualize: Generate comparison plots

        Returns:
            angles_df: Processed joint angles
            quality_report: Comprehensive quality metrics
        """

        print("\n" + "="*70)
        print(f"IMPROVED GAIT ANALYSIS PIPELINE")
        print(f"Video: {video_path}")
        print("="*70)

        # Stage 1: MediaPipe Pose Extraction
        print("\n[1/5] Extracting MediaPipe landmarks...")
        landmarks, fps_detected = self.base_extractor.extract_pose_landmarks(video_path)
        if fps_detected:
            self.fps = fps_detected
            self.signal_processor.fps = fps_detected
            self.constraint_enforcer.fps = fps_detected

        # Stage 2: Landmark Quality Assessment (Optional)
        quality_report = {'landmark_qc': None}

        if enable_landmark_qc:
            print("[2/5] Assessing landmark quality...")
            landmarks_df = self._landmarks_to_dataframe(landmarks)
            lm_quality = self.landmark_qc.assess_landmark_quality(landmarks_df)

            # Check for critical failures
            critical_landmarks = ['RIGHT_KNEE', 'LEFT_KNEE', 'RIGHT_ANKLE', 'LEFT_ANKLE',
                                 'RIGHT_HIP', 'LEFT_HIP']
            critical_quality = {k: v for k, v in lm_quality.items() if k in critical_landmarks}

            avg_critical_quality = np.mean([q.overall_score for q in critical_quality.values()])
            quality_report['landmark_qc'] = {
                'avg_quality': avg_critical_quality,
                'details': {k: v.overall_score for k, v in critical_quality.items()}
            }

            if avg_critical_quality < 0.5:
                print(f"  ⚠️  WARNING: Low landmark quality ({avg_critical_quality:.2f})")
                print("  → Video may not be suitable for analysis")
            else:
                print(f"  ✓ Landmark quality acceptable ({avg_critical_quality:.2f})")
        else:
            print("[2/5] Skipping landmark QC...")

        # Stage 3: Joint Angle Calculation
        print("[3/5] Calculating joint angles...")
        angles_raw = self.base_extractor.calculate_joint_angles(landmarks)

        # Stage 4: Improved Signal Processing (Optional)
        if enable_improved_filtering:
            print("[4/5] Applying improved signal processing...")
            angles_processed, signal_quality = process_angles_dataframe(angles_raw, self.fps)

            quality_report['signal_processing'] = {
                joint: {
                    'snr': metrics['snr'],
                    'smoothness': metrics['smoothness'],
                    'overall_quality': metrics['overall_quality']
                }
                for joint, metrics in signal_quality.items()
            }

            # Report
            avg_quality = np.mean([m['overall_quality'] for m in signal_quality.values()])
            print(f"  ✓ Average signal quality: {avg_quality:.2f}")
        else:
            print("[4/5] Skipping improved filtering...")
            angles_processed = angles_raw

        # Stage 5: Kinematic Constraints (Optional)
        if enable_kinematic_constraints:
            print("[5/5] Enforcing kinematic constraints...")

            # Before constraints
            violations_before = self.constraint_enforcer.validate_angles(angles_processed)
            total_violations_before = sum(v['angle_violations'] + v['velocity_violations'] + v['acceleration_violations']
                                         for v in violations_before.values())

            # Apply constraints
            angles_final = self.constraint_enforcer.enforce_joint_angle_constraints(angles_processed)

            # After constraints
            violations_after = self.constraint_enforcer.validate_angles(angles_final)
            total_violations_after = sum(v['angle_violations'] + v['velocity_violations'] + v['acceleration_violations']
                                        for v in violations_after.values())

            violations_fixed = total_violations_before - total_violations_after

            quality_report['kinematic_constraints'] = {
                'violations_before': total_violations_before,
                'violations_after': total_violations_after,
                'violations_fixed': violations_fixed
            }

            print(f"  ✓ Fixed {violations_fixed} constraint violations")
        else:
            print("[5/5] Skipping kinematic constraints...")
            angles_final = angles_processed

        # Visualization
        if visualize:
            self._visualize_pipeline(angles_raw, angles_final, quality_report)

        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)

        return angles_final, quality_report

    def compare_with_baseline(self, video_path: str, ground_truth_df: pd.DataFrame = None):
        """
        Compare improved pipeline vs baseline on a single video.

        Args:
            video_path: Path to video
            ground_truth_df: Optional ground truth angles for correlation analysis
        """

        # Run baseline pipeline
        print("\n" + "="*70)
        print("BASELINE PIPELINE (Original)")
        print("="*70)

        landmarks, _ = self.base_extractor.extract_pose_landmarks(video_path)
        angles_baseline = self.base_extractor.calculate_joint_angles(landmarks)

        # Basic fill (old method)
        for col in angles_baseline.columns:
            angles_baseline[col] = angles_baseline[col].fillna(method='bfill').fillna(method='ffill')

        # Run improved pipeline
        print("\n" + "="*70)
        print("IMPROVED PIPELINE (All Enhancements)")
        print("="*70)

        angles_improved, quality_report = self.process_video(
            video_path,
            enable_landmark_qc=True,
            enable_improved_filtering=True,
            enable_kinematic_constraints=True,
            visualize=False
        )

        # Compare
        print("\n" + "="*70)
        print("COMPARISON: Baseline vs Improved")
        print("="*70)

        joints = ['right_knee_angle', 'right_hip_angle', 'right_ankle_angle']

        for joint in joints:
            if joint not in angles_baseline.columns or joint not in angles_improved.columns:
                continue

            # Compute metrics
            baseline_signal = angles_baseline[joint].values
            improved_signal = angles_improved[joint].values

            # Smoothness (inverse of jerk)
            baseline_jerk = np.std(np.diff(baseline_signal, n=2))
            improved_jerk = np.std(np.diff(improved_signal, n=2))

            smoothness_improvement = (baseline_jerk - improved_jerk) / baseline_jerk * 100

            # Range of motion
            baseline_rom = np.ptp(baseline_signal)
            improved_rom = np.ptp(improved_signal)

            print(f"\n{joint}:")
            print(f"  Jerk (lower is better):  {baseline_jerk:.2f} → {improved_jerk:.2f} ({smoothness_improvement:+.1f}%)")
            print(f"  Range of motion:         {baseline_rom:.1f}° → {improved_rom:.1f}°")

            # If ground truth available, compute correlation
            if ground_truth_df is not None and joint in ground_truth_df.columns:
                from scipy.stats import pearsonr

                # Resample to match length
                min_len = min(len(baseline_signal), len(improved_signal), len(ground_truth_df))
                baseline_corr = pearsonr(baseline_signal[:min_len], ground_truth_df[joint].values[:min_len])[0]
                improved_corr = pearsonr(improved_signal[:min_len], ground_truth_df[joint].values[:min_len])[0]

                print(f"  Correlation with GT:     {baseline_corr:.3f} → {improved_corr:.3f} ({improved_corr - baseline_corr:+.3f})")

        # Visualize comparison
        self._visualize_comparison(angles_baseline, angles_improved, joints)

    def _landmarks_to_dataframe(self, landmarks) -> pd.DataFrame:
        """Convert landmarks list to DataFrame for quality assessment."""
        landmarks_long = []

        landmark_names = {
            23: 'LEFT_HIP', 24: 'RIGHT_HIP',
            25: 'LEFT_KNEE', 26: 'RIGHT_KNEE',
            27: 'LEFT_ANKLE', 28: 'RIGHT_ANKLE',
            29: 'LEFT_HEEL', 30: 'RIGHT_HEEL',
            31: 'LEFT_FOOT_INDEX', 32: 'RIGHT_FOOT_INDEX'
        }

        for frame_idx, frame_data in enumerate(landmarks):
            if frame_data is not None:
                for lm_idx, lm in enumerate(frame_data.landmark):
                    if lm_idx in landmark_names:
                        landmarks_long.append({
                            'frame': frame_idx,
                            'landmark': landmark_names[lm_idx],
                            'x': lm.x,
                            'y': lm.y,
                            'z': lm.z,
                            'visibility': lm.visibility
                        })

        return pd.DataFrame(landmarks_long)

    def _visualize_pipeline(self, angles_raw: pd.DataFrame, angles_final: pd.DataFrame, quality_report: Dict):
        """Visualize before/after comparison."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        joints = ['right_knee_angle', 'right_hip_angle', 'right_ankle_angle']
        labels = ['Knee', 'Hip', 'Ankle']

        for ax, joint, label in zip(axes, joints, labels):
            if joint not in angles_raw.columns or joint not in angles_final.columns:
                continue

            ax.plot(angles_raw[joint], 'r-', alpha=0.3, label='Raw', linewidth=1)
            ax.plot(angles_final[joint], 'b-', label='Improved', linewidth=2)

            ax.set_title(f"{label} - Pipeline Comparison")
            ax.set_ylabel('Angle (deg)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Frame')
        plt.tight_layout()
        plt.savefig('/data/gait/2d_project/improved_pipeline_visualization.png', dpi=150)
        print("\n✓ Saved visualization to improved_pipeline_visualization.png")

    def _visualize_comparison(self, baseline: pd.DataFrame, improved: pd.DataFrame, joints: list):
        """Visualize baseline vs improved comparison."""
        fig, axes = plt.subplots(len(joints), 1, figsize=(14, 10))

        for ax, joint in zip(axes, joints):
            if joint not in baseline.columns or joint not in improved.columns:
                continue

            ax.plot(baseline[joint], 'gray', alpha=0.4, label='Baseline', linewidth=1)
            ax.plot(improved[joint], 'blue', label='Improved', linewidth=2)

            ax.set_title(f"{joint} - Baseline vs Improved")
            ax.set_ylabel('Angle (deg)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Frame')
        plt.tight_layout()
        plt.savefig('/data/gait/2d_project/baseline_vs_improved_comparison.png', dpi=150)
        print("\n✓ Saved comparison to baseline_vs_improved_comparison.png")


# Example usage
if __name__ == "__main__":
    import sys

    # Test on worst-performing subject from QC report
    test_subjects = {
        'S1_15': '/data/gait/data/15/15-2.mp4',  # Ankle r=0.316 (worst)
        'S1_08': '/data/gait/data/8/8-2.mp4',    # Knee r=0.443
        'S1_16': '/data/gait/data/16/16-2.mp4',  # Knee r=0.433
    }

    subject = sys.argv[1] if len(sys.argv) > 1 else 'S1_15'

    if subject not in test_subjects:
        print(f"Unknown subject: {subject}")
        print(f"Available: {list(test_subjects.keys())}")
        sys.exit(1)

    video_path = test_subjects[subject]

    print(f"\n{'='*70}")
    print(f"TESTING IMPROVED PIPELINE ON {subject}")
    print(f"Known issue: Poor tracking quality from QC analysis")
    print(f"{'='*70}")

    # Initialize pipeline
    pipeline = ImprovedGaitAnalysisPipeline(fps=30, subject_height=1.70)

    # Run comparison
    pipeline.compare_with_baseline(video_path)

    print("\n✓ Analysis complete!")
    print("\nTo test another subject:")
    print("  python improved_pipeline_integrated.py S1_08")
    print("  python improved_pipeline_integrated.py S1_16")
