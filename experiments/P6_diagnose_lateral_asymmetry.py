"""
P6: Lateral Asymmetry Diagnosis - Phase 1 & 2

Goal: Identify root cause of 6× right-side outlier asymmetry
      despite bidirectional walking (camera distance should equalize)

Hypotheses:
1. Left/right label inconsistency (MediaPipe vs Ground Truth)
2. Direction-dependent performance (outbound vs inbound)
3. MediaPipe model bias for lateral views

Author: Research Team
Date: 2025-10-22
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import seaborn as sns

sns.set_style("whitegrid")


class LateralAsymmetryDiagnostic:
    """Diagnose left/right asymmetry in lateral view gait analysis."""

    def __init__(self, data_root: str = "/data/gait/data"):
        self.data_root = Path(data_root)
        self.results = {}

    def _find_mediapipe_csv(self, subject_id: str) -> Path:
        """Find MediaPipe side view CSV for subject (same logic as V4)."""
        subject_num = subject_id.split('_')[-1]
        csv_files = []

        # V4 logic
        patterns = [f"{subject_num}/*_side_pose_fps*.csv"]

        try:
            numeric = int(subject_num)
            patterns.append(f"{numeric}/*_side_pose_fps*.csv")
            patterns.append(f"{numeric:02d}/*_side_pose_fps*.csv")
        except ValueError:
            pass

        for pattern in patterns:
            csv_files.extend(sorted(self.data_root.glob(pattern)))

        if not csv_files:
            csv_files.extend(sorted(self.data_root.glob(f"{subject_id}/*_side_pose_fps*.csv")))

        if csv_files:
            return csv_files[0]

        raise FileNotFoundError(f"No side view CSV found for {subject_id} in {self.data_root}")

    def load_subject_data(self, subject_id: str) -> Tuple[pd.DataFrame, Dict]:
        """Load pose data and ground truth for a subject."""
        # Find MediaPipe CSV
        csv_file = self._find_mediapipe_csv(subject_id)
        print(f"  Loading: {csv_file}")

        # Load CSV (already in long format)
        df_long = pd.read_csv(csv_file)

        # Convert to wide format with x/y/z columns per joint
        df_angles = self._convert_long_to_wide(df_long)

        # Load ground truth from evaluation report
        with open('tiered_evaluation_report_v52.json', 'r') as f:
            report = json.load(f)

        if subject_id not in report['subjects']:
            raise ValueError(f"Subject {subject_id} not in evaluation report")

        gt_info = report['subjects'][subject_id]['temporal']

        return df_angles, gt_info

    def _convert_long_to_wide(self, df_long: pd.DataFrame) -> pd.DataFrame:
        """Convert long format to wide with x/y/z columns per joint."""
        positions_needed = ['left_heel', 'right_heel', 'left_ankle', 'right_ankle',
                           'left_hip', 'right_hip', 'left_knee', 'right_knee']

        df_filtered = df_long[df_long['position'].isin(positions_needed)].copy()

        frames = []
        for frame_num in df_filtered['frame'].unique():
            frame_data = {'frame': frame_num}
            frame_subset = df_filtered[df_filtered['frame'] == frame_num]

            for _, row in frame_subset.iterrows():
                pos = row['position']
                frame_data[f'x_{pos}'] = row['x']
                frame_data[f'y_{pos}'] = row['y']
                frame_data[f'z_{pos}'] = row['z']

            frames.append(frame_data)

        return pd.DataFrame(frames)

    def analyze_depth_by_side(self, df: pd.DataFrame, subject_id: str) -> Dict:
        """
        Analyze Z-coordinate (camera depth) for left/right heels.

        In lateral view, Z represents distance from camera:
        - Z < 0: Closer to camera (front)
        - Z > 0: Further from camera (back)

        Expected: In bidirectional walking, both sides should be
                  front/back equally often.
        """
        left_z = df['z_left_heel'].dropna()
        right_z = df['z_right_heel'].dropna()

        result = {
            'subject_id': subject_id,
            'left_z_mean': float(left_z.mean()),
            'left_z_std': float(left_z.std()),
            'right_z_mean': float(right_z.mean()),
            'right_z_std': float(right_z.std()),
            'left_z_range': [float(left_z.min()), float(left_z.max())],
            'right_z_range': [float(right_z.min()), float(right_z.max())],
            # Count frames where each side is closer to camera
            'left_closer_count': int((left_z < right_z).sum()),
            'right_closer_count': int((left_z > right_z).sum()),
            'equal_depth_count': int((np.abs(left_z - right_z) < 0.05).sum()),
        }

        # Asymmetry metric: if one side is consistently closer, suggests labeling issue
        total_comparable = result['left_closer_count'] + result['right_closer_count']
        if total_comparable > 0:
            result['depth_asymmetry_ratio'] = (
                abs(result['left_closer_count'] - result['right_closer_count']) / total_comparable
            )
        else:
            result['depth_asymmetry_ratio'] = 0.0

        return result

    def detect_walking_direction(self, df: pd.DataFrame, fps: float = 30.0) -> pd.DataFrame:
        """
        Detect walking direction from hip trajectory.

        Returns:
            DataFrame with additional 'direction' column:
            - 'outbound': Walking in positive X direction
            - 'inbound': Walking in negative X direction
            - 'turn': Turning phase
        """
        df = df.copy()

        # Use left hip X coordinate (lateral position)
        hip_x = df['x_left_hip'].values

        # Calculate velocity (smoothed)
        window = int(fps * 0.5)  # 0.5 second window
        if window % 2 == 0:
            window += 1

        # Pad for savgol
        from scipy.signal import savgol_filter
        if len(hip_x) > window:
            hip_x_smooth = savgol_filter(
                pd.Series(hip_x).interpolate(method='linear').values,
                window, 2
            )
        else:
            hip_x_smooth = hip_x

        velocity_x = np.gradient(hip_x_smooth) * fps

        # Classify direction
        direction = np.full(len(df), 'unknown', dtype=object)
        direction[velocity_x > 0.1] = 'outbound'
        direction[velocity_x < -0.1] = 'inbound'
        direction[np.abs(velocity_x) <= 0.1] = 'turn'

        df['direction'] = direction
        df['velocity_x'] = velocity_x

        return df

    def analyze_strike_accuracy_by_direction(
        self,
        df: pd.DataFrame,
        gt_info: Dict,
        subject_id: str
    ) -> Dict:
        """
        Analyze heel strike detection accuracy separately for each direction.

        Key insight: If one side performs worse in one direction,
                     suggests direction-dependent labeling confusion.
        """
        df = self.detect_walking_direction(df)

        # Get detected strikes from ground truth info
        left_strikes = gt_info.get('left_heel_strikes', [])
        right_strikes = gt_info.get('right_heel_strikes', [])

        # Classify strikes by direction
        def classify_strikes(strike_frames: List[int]) -> Dict:
            classified = {'outbound': 0, 'inbound': 0, 'turn': 0}
            for frame in strike_frames:
                if frame < len(df):
                    direction = df.iloc[frame]['direction']
                    classified[direction] = classified.get(direction, 0) + 1
            return classified

        left_by_dir = classify_strikes(left_strikes)
        right_by_dir = classify_strikes(right_strikes)

        # Ground truth stride counts (per side)
        gt_left_strides = gt_info.get('left_gt_stride_count', 0)
        gt_right_strides = gt_info.get('right_gt_stride_count', 0)

        result = {
            'subject_id': subject_id,
            'gt_left_strides': gt_left_strides,
            'gt_right_strides': gt_right_strides,
            'detected_left_total': len(left_strikes),
            'detected_right_total': len(right_strikes),
            'left_by_direction': left_by_dir,
            'right_by_direction': right_by_dir,
            'direction_distribution': {
                'outbound_frames': int((df['direction'] == 'outbound').sum()),
                'inbound_frames': int((df['direction'] == 'inbound').sum()),
                'turn_frames': int((df['direction'] == 'turn').sum()),
            }
        }

        # Calculate detection ratios
        if gt_left_strides > 0:
            result['left_detection_ratio'] = len(left_strikes) / gt_left_strides
        if gt_right_strides > 0:
            result['right_detection_ratio'] = len(right_strikes) / gt_right_strides

        # Direction-specific ratios (assuming GT evenly split)
        if gt_left_strides > 0:
            expected_per_dir = gt_left_strides / 2
            result['left_outbound_ratio'] = left_by_dir['outbound'] / expected_per_dir
            result['left_inbound_ratio'] = left_by_dir['inbound'] / expected_per_dir

        if gt_right_strides > 0:
            expected_per_dir = gt_right_strides / 2
            result['right_outbound_ratio'] = right_by_dir['outbound'] / expected_per_dir
            result['right_inbound_ratio'] = right_by_dir['inbound'] / expected_per_dir

        return result

    def diagnose_label_swap_hypothesis(
        self,
        depth_result: Dict,
        direction_result: Dict
    ) -> Dict:
        """
        Test hypothesis: MediaPipe swaps left/right labels based on direction.

        Evidence for label swap:
        1. One side consistently closer to camera
        2. Detection ratio flips between outbound/inbound
        3. Right side has more outliers (due to mismatch with GT)
        """
        diagnosis = {
            'hypothesis': 'label_swap',
            'evidence': [],
            'confidence': 0.0
        }

        # Evidence 1: Depth asymmetry
        depth_asym = depth_result.get('depth_asymmetry_ratio', 0)
        if depth_asym > 0.6:  # One side closer >80% of time
            diagnosis['evidence'].append({
                'type': 'depth_asymmetry',
                'value': depth_asym,
                'interpretation': 'One side consistently closer to camera (suspicious)'
            })
            diagnosis['confidence'] += 0.3

        # Evidence 2: Direction-dependent detection ratios
        left_out = direction_result.get('left_outbound_ratio', 0)
        left_in = direction_result.get('left_inbound_ratio', 0)
        right_out = direction_result.get('right_outbound_ratio', 0)
        right_in = direction_result.get('right_inbound_ratio', 0)

        # If left performs better outbound and right performs better inbound
        # (or vice versa), suggests direction-dependent swap
        if (left_out > 0 and left_in > 0 and right_out > 0 and right_in > 0):
            ratio_flip = abs((left_out / left_in) - (right_in / right_out))
            if ratio_flip > 1.5:
                diagnosis['evidence'].append({
                    'type': 'direction_flip',
                    'left_ratio': left_out / left_in,
                    'right_ratio': right_out / right_in,
                    'interpretation': 'Detection quality flips between directions'
                })
                diagnosis['confidence'] += 0.4

        # Evidence 3: Overall detection asymmetry
        left_total_ratio = direction_result.get('left_detection_ratio', 0)
        right_total_ratio = direction_result.get('right_detection_ratio', 0)

        if left_total_ratio > 0 and right_total_ratio > 0:
            overall_asym = abs(left_total_ratio - right_total_ratio)
            if overall_asym > 0.5:
                diagnosis['evidence'].append({
                    'type': 'overall_asymmetry',
                    'left_ratio': left_total_ratio,
                    'right_ratio': right_total_ratio,
                    'interpretation': 'Large difference in overall detection ratio'
                })
                diagnosis['confidence'] += 0.3

        return diagnosis

    def run_full_diagnosis(self, subject_ids: List[str]) -> Dict:
        """Run comprehensive diagnosis on multiple subjects."""

        all_depth_results = []
        all_direction_results = []
        all_diagnoses = []

        for sid in subject_ids:
            print(f"\n{'='*60}")
            print(f"Analyzing {sid}...")
            print('='*60)

            try:
                df, gt_info = self.load_subject_data(sid)

                # Phase 1: Depth analysis
                print("\n[Phase 1] Depth-based label consistency check...")
                depth_result = self.analyze_depth_by_side(df, sid)
                all_depth_results.append(depth_result)

                print(f"  Left heel Z:  mean={depth_result['left_z_mean']:.3f}, "
                      f"std={depth_result['left_z_std']:.3f}")
                print(f"  Right heel Z: mean={depth_result['right_z_mean']:.3f}, "
                      f"std={depth_result['right_z_std']:.3f}")
                print(f"  Depth asymmetry: {depth_result['depth_asymmetry_ratio']:.2%}")

                # Phase 2: Direction-based analysis
                print("\n[Phase 2] Direction-dependent performance check...")
                direction_result = self.analyze_strike_accuracy_by_direction(df, gt_info, sid)
                all_direction_results.append(direction_result)

                print(f"  GT: Left={direction_result['gt_left_strides']}, "
                      f"Right={direction_result['gt_right_strides']}")
                print(f"  Detected: Left={direction_result['detected_left_total']}, "
                      f"Right={direction_result['detected_right_total']}")
                print(f"  Left by direction: {direction_result['left_by_direction']}")
                print(f"  Right by direction: {direction_result['right_by_direction']}")

                # Diagnosis
                print("\n[Diagnosis] Label swap hypothesis testing...")
                diagnosis = self.diagnose_label_swap_hypothesis(depth_result, direction_result)
                all_diagnoses.append(diagnosis)

                print(f"  Confidence: {diagnosis['confidence']:.1%}")
                for evidence in diagnosis['evidence']:
                    print(f"    - {evidence['type']}: {evidence['interpretation']}")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue

        # Aggregate results
        summary = {
            'subjects_analyzed': len(subject_ids),
            'depth_results': all_depth_results,
            'direction_results': all_direction_results,
            'diagnoses': all_diagnoses,
            'aggregate_confidence': np.mean([d['confidence'] for d in all_diagnoses])
        }

        return summary

    def visualize_results(self, summary: Dict, output_file: str = 'P6_asymmetry_diagnosis.png'):
        """Create diagnostic visualizations."""

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Lateral Asymmetry Diagnosis: Phase 1 & 2 Results',
                     fontsize=14, fontweight='bold')

        depth_results = summary['depth_results']
        direction_results = summary['direction_results']
        diagnoses = summary['diagnoses']

        # Plot 1: Depth asymmetry
        ax1 = axes[0, 0]
        subjects = [r['subject_id'] for r in depth_results]
        depth_asym = [r['depth_asymmetry_ratio'] for r in depth_results]

        colors = ['red' if d > 0.6 else 'orange' if d > 0.4 else 'green'
                  for d in depth_asym]
        ax1.barh(subjects, depth_asym, color=colors, alpha=0.7)
        ax1.axvline(0.6, color='red', linestyle='--', label='High asymmetry (>60%)')
        ax1.axvline(0.4, color='orange', linestyle='--', label='Moderate (>40%)')
        ax1.set_xlabel('Depth Asymmetry Ratio')
        ax1.set_title('Phase 1: Camera Depth Asymmetry')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)

        # Plot 2: Detection ratios by direction
        ax2 = axes[0, 1]
        x = np.arange(len(subjects))
        width = 0.2

        left_out = [r.get('left_outbound_ratio', 0) for r in direction_results]
        left_in = [r.get('left_inbound_ratio', 0) for r in direction_results]
        right_out = [r.get('right_outbound_ratio', 0) for r in direction_results]
        right_in = [r.get('right_inbound_ratio', 0) for r in direction_results]

        ax2.bar(x - 1.5*width, left_out, width, label='Left Outbound', color='blue', alpha=0.7)
        ax2.bar(x - 0.5*width, left_in, width, label='Left Inbound', color='lightblue', alpha=0.7)
        ax2.bar(x + 0.5*width, right_out, width, label='Right Outbound', color='red', alpha=0.7)
        ax2.bar(x + 1.5*width, right_in, width, label='Right Inbound', color='lightcoral', alpha=0.7)

        ax2.axhline(1.0, color='black', linestyle='--', linewidth=0.8, label='Perfect (1.0×)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(subjects, rotation=45, ha='right')
        ax2.set_ylabel('Detection Ratio (detected/GT)')
        ax2.set_title('Phase 2: Direction-Dependent Detection')
        ax2.legend(fontsize=8)
        ax2.grid(axis='y', alpha=0.3)

        # Plot 3: Diagnosis confidence
        ax3 = axes[1, 0]
        confidences = [d['confidence'] for d in diagnoses]
        colors_conf = ['red' if c > 0.7 else 'orange' if c > 0.4 else 'green'
                       for c in confidences]
        ax3.barh(subjects, confidences, color=colors_conf, alpha=0.7)
        ax3.axvline(0.7, color='red', linestyle='--', label='High confidence (>70%)')
        ax3.axvline(0.4, color='orange', linestyle='--', label='Moderate (>40%)')
        ax3.set_xlabel('Label Swap Hypothesis Confidence')
        ax3.set_title('Diagnosis: Label Swap Likelihood')
        ax3.legend()
        ax3.grid(axis='x', alpha=0.3)

        # Plot 4: Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Aggregate statistics
        high_asym_count = sum(1 for d in depth_asym if d > 0.6)
        high_conf_count = sum(1 for c in confidences if c > 0.7)
        avg_left_ratio = np.mean([r.get('left_detection_ratio', 0) for r in direction_results])
        avg_right_ratio = np.mean([r.get('right_detection_ratio', 0) for r in direction_results])

        summary_text = f"""
        DIAGNOSTIC SUMMARY
        ══════════════════════════════════

        Subjects Analyzed: {len(subjects)}

        Phase 1: Depth Analysis
          High asymmetry (>60%): {high_asym_count}/{len(subjects)}
          Avg asymmetry: {np.mean(depth_asym):.1%}

        Phase 2: Direction Analysis
          Avg Left ratio: {avg_left_ratio:.2f}×
          Avg Right ratio: {avg_right_ratio:.2f}×
          Asymmetry: {abs(avg_left_ratio - avg_right_ratio):.2f}×

        Diagnosis: Label Swap Hypothesis
          High confidence: {high_conf_count}/{len(subjects)}
          Avg confidence: {summary['aggregate_confidence']:.1%}

        ══════════════════════════════════
        INTERPRETATION:

        {"✓ LABEL SWAP LIKELY" if summary['aggregate_confidence'] > 0.6
         else "⚠ INCONCLUSIVE - Mixed Evidence"
         if summary['aggregate_confidence'] > 0.3
         else "✗ Label swap unlikely"}

        {"  → Recommend: Implement direction-based"
         if summary['aggregate_confidence'] > 0.6
         else "  → Recommend: Further investigation"}
        {"    label correction (Option A)" if summary['aggregate_confidence'] > 0.6 else ""}
        """

        ax4.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to {output_file}")

        return fig


def main():
    """Run Phase 1 & 2 diagnosis on sample subjects."""

    print("=" * 70)
    print("P6: LATERAL ASYMMETRY DIAGNOSIS")
    print("Phase 1 & 2: Label Consistency + Direction-Dependent Analysis")
    print("=" * 70)

    # Test on first 5 subjects
    sample_subjects = ['S1_01', 'S1_03', 'S1_08', 'S1_09', 'S1_10']

    diagnostic = LateralAsymmetryDiagnostic()

    print("\nRunning comprehensive diagnosis...")
    summary = diagnostic.run_full_diagnosis(sample_subjects)

    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    print(f"\nSubjects analyzed: {summary['subjects_analyzed']}")
    print(f"Average label swap confidence: {summary['aggregate_confidence']:.1%}")

    # Save results
    output_file = 'P6_asymmetry_diagnosis_results.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")

    # Visualize
    diagnostic.visualize_results(summary)

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if summary['aggregate_confidence'] > 0.6:
        print("\n✓ HIGH CONFIDENCE: Label swap hypothesis supported")
        print("\nNext steps:")
        print("  1. Implement direction-based label correction (Option A)")
        print("  2. Test on full cohort (21 subjects)")
        print("  3. Re-run V5.2 with corrected labels")
        print("  4. Expected: Right ICC 0.14 → 0.60+ (parity with left)")
    elif summary['aggregate_confidence'] > 0.3:
        print("\n⚠ INCONCLUSIVE: Mixed evidence")
        print("\nNext steps:")
        print("  1. Expand to full cohort analysis")
        print("  2. Visual inspection of videos")
        print("  3. Consider bilateral consensus approach (Option C)")
    else:
        print("\n✗ LOW CONFIDENCE: Label swap unlikely")
        print("\nAlternative hypotheses:")
        print("  1. MediaPipe model bias for lateral views")
        print("  2. True physiological asymmetry in cohort")
        print("  3. Consider Option C: Symmetric bilateral scaling")

    print("\n" + "=" * 70)
    print("Diagnosis complete. See P6_asymmetry_diagnosis.png for visualizations.")
    print("=" * 70)


if __name__ == "__main__":
    main()
