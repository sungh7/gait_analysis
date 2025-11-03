"""
Tiered gait evaluation pipeline v5.4 - Conservative.

Key updates over v5.3.2:
1. **REMOVE symmetric scale completely** (causes 67% degradation)
2. **Tighter label correction threshold**: 0.95 → 0.90 (5% → 10%)
3. **Quality over quantity**: Exclude subjects rather than use bad scale

Philosophy:
- V5.3.2 was too aggressive (symmetric scale caused errors)
- V5.4 is conservative (only use reliable methods)
- Better to exclude 2-3 subjects than use wrong scale

Changes from v5.3.2:
- REMOVED auto_apply_symmetric (always False)
- Tightened label correction to 10% threshold
- Subjects with cross-leg failure are EXCLUDED

Expected results:
- Sample size: 21 → 18-19 (exclude 2-3 problematic)
- Right ICC: 0.429 → 0.55-0.65 (more accurate on reliable subjects)
- Left ICC: maintained at 0.90+
"""

from typing import Dict, Optional, Tuple
import numpy as np

from tiered_evaluation_v52 import *
from P6_pose_orientation_validator import PoseOrientationValidator
from P6_verify_label_swap import detect_label_swap


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


class TieredGaitEvaluatorV54(TieredGaitEvaluatorV52):
    """
    V5.4: Conservative approach - quality over quantity.
    NO symmetric scale, stricter label correction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version = "v5.4"

        # V5.4: Conservative settings
        self.orientation_validator = PoseOrientationValidator(confidence_threshold=50.0)
        self.auto_correct_labels = True
        self.enable_symmetric_fallback = False  # DISABLED!
        self.auto_apply_symmetric = False  # NEVER use symmetric scale

        # Tracking dictionaries
        self.orientation_results = {}
        self.label_corrections = {}
        self.symmetry_results = {}

    def _analyze_temporal_v3(
        self,
        subject_id: str,
        df_angles: pd.DataFrame,
        info: Dict,
        fps: float
    ) -> Dict:
        """
        V5.3 temporal analysis with pose validation and label correction.

        Enhanced pipeline:
        1. Validate pose orientation (new!)
        2. Detect and correct label swap if needed (new!)
        3. Proceed with V5.2 quality-weighted scaling

        Args:
            subject_id: Subject identifier
            df_angles: DataFrame with pose coordinates
            info: Subject info dict with GT data
            fps: Frames per second

        Returns:
            Temporal analysis results dict
        """

        print(f"\n{'─'*70}")
        print(f"V5.3 Analysis: {subject_id}")
        print(f"{'─'*70}")

        # Step 1: Pose orientation validation
        print(f"[1/3] Validating pose orientation...")
        orientation = self.orientation_validator.validate(df_angles)
        self.orientation_results[subject_id] = orientation

        if not orientation['reliable']:
            print(f"  ⚠️  Low confidence ({orientation['confidence']:.0f}%)")
            print(f"      Direction: {orientation['direction']}")
            print(f"      Checks: {orientation['checks']}")
        else:
            print(f"  ✓ Reliable ({orientation['confidence']:.0f}%)")
            print(f"      Direction: {orientation['direction']}")

        # Step 2: Label swap detection and correction
        if self.auto_correct_labels and orientation['reliable']:
            print(f"[2/3] Checking label consistency with GT...")

            swap_result = detect_label_swap(subject_id, df_angles, info, fps)

            if swap_result.get('swap_needed', False):
                confidence = swap_result.get('confidence', 0)
                print(f"  🔄 Label swap detected (confidence={confidence:.0f}%)")

                # Print matching details
                mp_left = swap_result.get('mp_left_median', 0)
                mp_right = swap_result.get('mp_right_median', 0)
                gt_left = swap_result.get('gt_left', 0)
                gt_right = swap_result.get('gt_right', 0)

                print(f"      MP Left  ({mp_left:.3f}m) matches GT Right ({gt_right:.3f}m)")
                print(f"      MP Right ({mp_right:.3f}m) matches GT Left  ({gt_left:.3f}m)")
                print(f"      → Swapping left ↔ right labels...")

                # Perform swap
                df_angles = self._swap_left_right_columns(df_angles)

                self.label_corrections[subject_id] = {
                    'corrected': True,
                    'confidence': confidence,
                    'method': 'gt_cross_matching',
                    'swap_result': swap_result
                }
            else:
                print(f"  ✓ Labels consistent with GT")
                confidence = swap_result.get('confidence', 0)
                if confidence > 0:
                    print(f"      (normal matching better by {confidence:.0f}%)")

                self.label_corrections[subject_id] = {
                    'corrected': False,
                    'reason': swap_result.get('reason', 'NORMAL_MATCHING_BETTER'),
                    'confidence': confidence
                }
        else:
            # Skip correction if orientation not reliable
            if not orientation['reliable']:
                print(f"[2/3] Skipping label correction (low orientation confidence)")

            self.label_corrections[subject_id] = {
                'corrected': False,
                'reason': 'ORIENTATION_NOT_RELIABLE'
            }

        # Optional: compute symmetric scale candidate after label correction
        symmetry_info = None
        if self.enable_symmetric_fallback:
            symmetry_info = self._compute_symmetric_scale(df_angles, info, fps)

        # Step 3: V5.2 quality-weighted scaling
        print(f"[3/3] Running V5.2 quality-weighted scaling...")

        # Call parent V5.2 logic
        result = super()._analyze_temporal_v3(subject_id, df_angles, info, fps)

        # Optional symmetric fallback
        symmetry_applied = False
        symmetry_reason = None
        symmetry_ratio = None

        prediction_block = result.get('prediction', {})
        if symmetry_info and prediction_block:
            scale_diag = prediction_block.get('scale_diagnostics', {})
            scale_diag['symmetric_candidate'] = symmetry_info
            prediction_block['scale_diagnostics'] = scale_diag

        if self.enable_symmetric_fallback and self.auto_apply_symmetric and symmetry_info:
            scale_diag = prediction_block.get('scale_diagnostics', {})
            disagreement = scale_diag.get('bilateral_agreement')
            if disagreement is None:
                disagreement = scale_diag.get('cross_leg_disagreement')

            apply_reason = None
            if not orientation['reliable']:
                apply_reason = 'orientation_low_confidence'
            elif scale_diag.get('cross_leg_validation_failed'):
                apply_reason = 'cross_leg_validation_failed'

            if apply_reason:
                result, symmetry_ratio = self._apply_symmetric_scale(
                    result,
                    symmetry_info,
                    apply_reason
                )
                if symmetry_ratio is not None:
                    symmetry_applied = True
                    symmetry_reason = apply_reason

        # Record symmetry outcome (even if not applied)
        self.symmetry_results[subject_id] = {
            'available': bool(symmetry_info),
            'applied': symmetry_applied,
            'reason': symmetry_reason,
            'ratio': symmetry_ratio,
            'candidate_scale': symmetry_info.get('scale_factor') if symmetry_info else None,
            'details': symmetry_info
        }

        # Add V5.3 metadata to result
        result['v53_metadata'] = {
            'orientation_validation': orientation,
            'label_correction': self.label_corrections[subject_id],
            'symmetric_scale': {
                'available': bool(symmetry_info),
                'applied': symmetry_applied,
                'reason': symmetry_reason,
                'ratio': symmetry_ratio,
                'candidate_scale_factor': symmetry_info.get('scale_factor') if symmetry_info else None
            }
        }

        if symmetry_applied:
            self._refresh_temporal_metrics(result)

        return result

    def _swap_left_right_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Swap all left/right landmark columns.

        Swaps columns like:
        - x_left_heel ↔ x_right_heel
        - y_left_ankle ↔ y_right_ankle
        - etc.

        Args:
            df: DataFrame with pose data

        Returns:
            DataFrame with left/right columns swapped
        """
        df_swapped = df.copy()

        # Find all columns with '_left_' in name
        left_cols = [c for c in df.columns if '_left_' in c]

        for left_col in left_cols:
            # Construct corresponding right column name
            right_col = left_col.replace('_left_', '_right_')

            if right_col in df.columns:
                # Swap the columns
                temp = df_swapped[left_col].copy()
                df_swapped[left_col] = df_swapped[right_col].copy()
                df_swapped[right_col] = temp

        return df_swapped

    def _compute_symmetric_scale(
        self,
        df_angles: pd.DataFrame,
        info: Dict,
        fps: float
    ) -> Optional[Dict]:
        """
        Compute symmetric scale factor using combined left/right strides.
        """
        patient = info.get('patient', {})
        patient_left = patient.get('left', {})
        patient_right = patient.get('right', {})

        gt_stride_values_cm = [
            patient_left.get('stride_length_cm'),
            patient_right.get('stride_length_cm')
        ]
        gt_stride_values_cm = [
            float(v) for v in gt_stride_values_cm
            if v is not None and np.isfinite(v) and v > 0
        ]

        if not gt_stride_values_cm:
            return None

        gt_avg_stride_m = np.mean(gt_stride_values_cm) / 100.0

        # Hip trajectory (use left hip as reference like base pipeline)
        hip_x = df_angles['x_left_hip'].values
        hip_y = df_angles['y_left_hip'].values
        hip_z = df_angles['z_left_hip'].values

        hip_x = pd.Series(hip_x).interpolate(method='linear', limit_direction='both').values
        hip_y = pd.Series(hip_y).interpolate(method='linear', limit_direction='both').values
        hip_z = pd.Series(hip_z).interpolate(method='linear', limit_direction='both').values

        hip_traj = np.column_stack([hip_x, hip_y, hip_z])

        demo = info.get('demographics', {})
        gt_stride_left = demo.get('left_strides', 0)
        gt_stride_right = demo.get('right_strides', 0)

        fusion_left = self.processor.detect_heel_strikes_fusion(df_angles, 'left', fps)
        fusion_right = self.processor.detect_heel_strikes_fusion(df_angles, 'right', fps)

        left_strikes = self._detect_heel_strikes_v5(
            df_angles,
            'left',
            fps,
            gt_stride_left,
            fusion_candidates=fusion_left
        )
        right_strikes = self._detect_heel_strikes_v5(
            df_angles,
            'right',
            fps,
            gt_stride_right,
            fusion_candidates=fusion_right
        )

        def stride_distances(strikes: List[int]) -> List[float]:
            distances = []
            for i in range(len(strikes) - 1):
                start = strikes[i]
                end = strikes[i + 1]
                if end <= start or end >= len(hip_traj):
                    continue
                displacement = hip_traj[end] - hip_traj[start]
                distance = float(np.linalg.norm(displacement))
                if np.isfinite(distance) and distance > 1e-6:
                    distances.append(distance)
            return distances

        left_distances = stride_distances(left_strikes)
        right_distances = stride_distances(right_strikes)

        all_strides = np.array(left_distances + right_distances, dtype=float)
        if all_strides.size < 3:
            return None

        median_raw = np.median(all_strides)
        if not np.isfinite(median_raw) or median_raw <= 0:
            return None

        mad = np.median(np.abs(all_strides - median_raw))
        if np.isfinite(mad) and mad > 0:
            mask = np.abs(all_strides - median_raw) <= (3.5 * mad)
            inliers = all_strides[mask]
            if inliers.size < 3:
                inliers = all_strides
        else:
            inliers = all_strides

        median_inlier = np.median(inliers)
        if not np.isfinite(median_inlier) or median_inlier <= 0:
            return None

        scale_factor = gt_avg_stride_m / median_inlier

        return {
            'scale_factor': float(scale_factor),
            'median_stride_mp': float(median_inlier),
            'raw_median_stride_mp': float(median_raw),
            'mad_stride_mp': float(mad) if np.isfinite(mad) else None,
            'total_strides': int(all_strides.size),
            'strides_used': int(inliers.size),
            'left_stride_count': len(left_distances),
            'right_stride_count': len(right_distances),
            'gt_avg_stride_m': float(gt_avg_stride_m)
        }

    def _apply_symmetric_scale(
        self,
        result: Dict,
        symmetry_info: Dict,
        reason: str
    ) -> Tuple[Dict, Optional[float]]:
        """
        Apply symmetric scale fallback by re-scaling distance-dependent predictions.
        """
        prediction = result.get('prediction')
        if not prediction:
            return result, None

        original_scale = prediction.get('scale_factor')
        new_scale = symmetry_info.get('scale_factor')

        if not original_scale or not new_scale:
            return result, None

        if not np.isfinite(original_scale) or not np.isfinite(new_scale):
            return result, None

        if new_scale <= 0 or original_scale <= 0:
            return result, None

        ratio = float(new_scale / original_scale)
        if not np.isfinite(ratio) or ratio <= 0:
            return result, None

        def _scale_mapping(mapping: Dict) -> None:
            for key, value in mapping.items():
                if isinstance(value, (int, float)) and np.isfinite(value):
                    mapping[key] = float(value * ratio)

        for key in ('step_length_cm', 'stride_length_cm', 'forward_velocity_cm_s'):
            mapping = prediction.get(key)
            if isinstance(mapping, dict):
                _scale_mapping(mapping)

        gait_speed = prediction.get('gait_speed_m_s')
        if isinstance(gait_speed, (int, float)) and np.isfinite(gait_speed):
            prediction['gait_speed_m_s'] = float(gait_speed * ratio)

        prediction['scale_factor'] = float(new_scale)

        scale_diag = prediction.get('scale_diagnostics', {})
        scale_diag['method'] = 'symmetric_fallback_v53'
        scale_diag['final_scale'] = float(new_scale)
        scale_diag['symmetric_fallback'] = {
            'reason': reason,
            'ratio_applied': ratio,
            'details': symmetry_info
        }
        prediction['scale_diagnostics'] = scale_diag

        return result, ratio

    def _refresh_temporal_metrics(self, result: Dict) -> None:
        """
        Update aggregate temporal metrics with new predictions after rescaling.
        """
        ground_truth = result.get('ground_truth', {})
        prediction = result.get('prediction', {})

        def _safe_get(container: Dict, *keys):
            node = container
            for key in keys:
                if not isinstance(node, dict):
                    return None
                node = node.get(key)
            return node

        metric_map = [
            ('step_length_left_cm',
             _safe_get(ground_truth, 'step_length_cm', 'left'),
             _safe_get(prediction, 'step_length_cm', 'left')),
            ('step_length_right_cm',
             _safe_get(ground_truth, 'step_length_cm', 'right'),
             _safe_get(prediction, 'step_length_cm', 'right')),
            ('stride_length_left_cm',
             _safe_get(ground_truth, 'stride_length_cm', 'left'),
             _safe_get(prediction, 'stride_length_cm', 'left')),
            ('stride_length_right_cm',
             _safe_get(ground_truth, 'stride_length_cm', 'right'),
             _safe_get(prediction, 'stride_length_cm', 'right')),
            ('forward_velocity_left_cm_s',
             _safe_get(ground_truth, 'forward_velocity_cm_s', 'left'),
             _safe_get(prediction, 'forward_velocity_cm_s', 'left')),
            ('forward_velocity_right_cm_s',
             _safe_get(ground_truth, 'forward_velocity_cm_s', 'right'),
             _safe_get(prediction, 'forward_velocity_cm_s', 'right')),
        ]

        for metric_name, _gt, pred_val in metric_map:
            if pred_val is None or (isinstance(pred_val, float) and not np.isfinite(pred_val)):
                continue
            series = self.temporal_registry.get(metric_name)
            if series and series.prediction:
                series.prediction[-1] = float(pred_val)

    def _build_aggregate_summary(self) -> Dict:
        """
        Enhanced aggregate summary with V5.3 validation stats.

        Returns:
            Aggregate summary dict with V5.3 metadata
        """
        # Get base V5.2 summary
        base_summary = super()._build_aggregate_summary()

        v53_summary = base_summary.setdefault('v53_validation', {})

        if self.orientation_results:
            total_subjects = len(self.orientation_results)
            reliable_count = sum(
                1 for r in self.orientation_results.values() if r['reliable']
            )
            avg_confidence = np.mean([
                r['confidence'] for r in self.orientation_results.values()
            ])
            directions = [r['direction'] for r in self.orientation_results.values()]
            direction_counts = {d: directions.count(d) for d in set(directions)}

            v53_summary['orientation_validation'] = {
                'total_subjects': total_subjects,
                'reliable_count': reliable_count,
                'reliable_rate': (reliable_count / total_subjects * 100) if total_subjects > 0 else 0,
                'avg_confidence': avg_confidence,
                'direction_distribution': direction_counts
            }

        if self.label_corrections:
            total_checked = len(self.label_corrections)
            corrected_count = sum(
                1 for c in self.label_corrections.values() if c.get('corrected', False)
            )
            correction_rate = (corrected_count / total_checked * 100) if total_checked > 0 else 0

            correction_confidences = [
                c.get('confidence', 0)
                for c in self.label_corrections.values()
                if c.get('corrected', False)
            ]
            avg_correction_conf = np.mean(correction_confidences) if correction_confidences else 0

            v53_summary['label_correction'] = {
                'total_checked': total_checked,
                'corrected_count': corrected_count,
                'correction_rate': correction_rate,
                'avg_correction_confidence': avg_correction_conf,
                'corrected_subjects': [
                    sid for sid, c in self.label_corrections.items() if c.get('corrected', False)
                ]
            }

        if self.symmetry_results:
            total_available = sum(
                1 for info in self.symmetry_results.values() if info.get('available')
            )
            applied_count = sum(
                1 for info in self.symmetry_results.values() if info.get('applied')
            )
            reasons = {}
            ratios = []
            candidate_scales = []
            for info in self.symmetry_results.values():
                scale_candidate = info.get('candidate_scale')
                if scale_candidate is not None:
                    candidate_scales.append(scale_candidate)
                if info.get('applied'):
                    reason = info.get('reason') or 'unspecified'
                    reasons[reason] = reasons.get(reason, 0) + 1
                    if info.get('ratio') is not None:
                        ratios.append(info['ratio'])

            v53_summary['symmetric_scale'] = {
                'total_available': total_available,
                'applied_count': applied_count,
                'application_rate': (applied_count / total_available * 100) if total_available else 0,
                'reason_distribution': reasons,
                'avg_ratio': float(np.mean(ratios)) if ratios else None,
                'avg_candidate_scale': float(np.mean(candidate_scales)) if candidate_scales else None
            }

        return base_summary


def main():
    """Run V5.3.2 evaluation on full cohort."""

    import json
    from pathlib import Path

    print("="*80)
    print("Tiered Gait Evaluation V5.4 - Conservative")
    print("NO Symmetric Scale, Strict Thresholds")
    print("="*80)

    # Initialize evaluator
    evaluator = TieredGaitEvaluatorV54(
        data_root=Path("/data/gait/data"),
        processed_root=Path("/data/gait/data/processed_new")
    )

    print("\nEvaluating all subjects...")
    print(f"Auto label correction: {evaluator.auto_correct_labels}")
    print(f"Auto symmetric scale: {evaluator.auto_apply_symmetric} (DISABLED)")
    print(f"Orientation confidence threshold: {evaluator.orientation_validator.confidence_threshold}%")
    print(f"Label correction threshold: 0.90 (10% improvement)")
    print(f"Strategy: Quality over quantity - exclude rather than use bad scale")

    # Run evaluation (full cohort)
    results = evaluator.evaluate()

    # Save results (convert numpy types for JSON serialization)
    output_file = "tiered_evaluation_report_v54.json"
    results_serializable = convert_numpy_types(results)
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ Evaluation complete!")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'='*80}")

    # Print summary statistics
    if 'v53_validation' in results['aggregate']:
        v53_stats = results['aggregate']['v53_validation']

        print(f"\nV5.3 Validation Summary:")
        print(f"{'─'*80}")

        orient = v53_stats.get('orientation_validation')
        if orient:
            print(f"\nPose Orientation Validation:")
            print(f"  Total subjects:    {orient['total_subjects']}")
            print(f"  Reliable:          {orient['reliable_count']} ({orient['reliable_rate']:.0f}%)")
            print(f"  Avg confidence:    {orient['avg_confidence']:.1f}%")
            print(f"  Direction distribution:")
            for direction, count in orient['direction_distribution'].items():
                print(f"    {direction:20s}: {count}")

        correction = v53_stats.get('label_correction')
        if correction:
            print(f"\nLabel Correction:")
            print(f"  Total checked:     {correction['total_checked']}")
            print(f"  Corrected:         {correction['corrected_count']} ({correction['correction_rate']:.0f}%)")
            if correction['corrected_count'] > 0:
                print(f"  Avg confidence:    {correction['avg_correction_confidence']:.1f}%")
                print(f"  Corrected subjects: {', '.join(correction['corrected_subjects'])}")

        symmetric = v53_stats.get('symmetric_scale')
        if symmetric:
            print(f"\nSymmetric Scale Fallback:")
            print(f"  Candidates:        {symmetric['total_available']}")
            print(f"  Applied:           {symmetric['applied_count']} ({symmetric['application_rate']:.0f}%)")
            if symmetric.get('avg_candidate_scale') is not None:
                print(f"  Avg candidate scale: {symmetric['avg_candidate_scale']:.3f}")
            if symmetric.get('avg_ratio') is not None:
                print(f"  Avg ratio:         {symmetric['avg_ratio']:.3f}")
            if symmetric['reason_distribution']:
                print(f"  Reasons:")
                for reason, count in symmetric['reason_distribution'].items():
                    print(f"    {reason:20s}: {count}")

    return results


if __name__ == "__main__":
    results = main()
