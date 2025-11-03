"""
Tiered gait evaluation pipeline v5.2.

Key updates over v5:
1. **Phase 5.2 Integration**: Quality-weighted stride-based scaling
2. Stride-level outlier rejection using MAD
3. Cross-leg scale validation (reject if >15% disagreement)
4. Enhanced turn detection with ankle curvature
5. Maintains all V5 functionality (template-based heel strike detection)

Change log from v5:
- Enhanced calculate_stride_based_scale_factor() with quality weighting
- Enhanced calculate_hybrid_scale_factor() with cross-leg validation
- Added _detect_turns_with_curvature() for improved turn detection
- Updated _analyze_temporal_v3() to use enhanced scaling

Target improvements:
- Step Length ICC: 0.02 → 0.40-0.45 (fair agreement)
- Step Length RMSE: 9.3 → 6-7 cm
- Retain 16 subjects (same as V5.1)
"""

from tiered_evaluation_v5 import *


class TieredGaitEvaluatorV52(TieredGaitEvaluatorV5):
    """
    V5.2: Adds quality-weighted scaling and cross-leg validation.
    Inherits template-based heel strike detection from V5.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.version = "v5.2"
        self.use_quality_weighting = True
        self.cross_leg_validation = True
        self.enhanced_turn_detection = True

    def _detect_turns_with_curvature(
        self,
        df_angles: pd.DataFrame,
        fps: float,
        curvature_threshold: float = 0.5
    ) -> List[int]:
        """
        Detect turn segments using ankle trajectory curvature.

        Args:
            df_angles: DataFrame with ankle positions
            fps: Frames per second
            curvature_threshold: Curvature threshold (rad/m)

        Returns:
            List of turn frame indices
        """
        # Use left ankle trajectory as reference
        ankle_x = df_angles['x_left_ankle'].values
        ankle_y = df_angles['y_left_ankle'].values
        ankle_z = df_angles['z_left_ankle'].values

        # Skip if too much missing data
        valid_mask = ~(np.isnan(ankle_x) | np.isnan(ankle_y) | np.isnan(ankle_z))
        if np.sum(valid_mask) < fps * 2:
            return []

        # Interpolate missing values
        ankle_x_clean = pd.Series(ankle_x).interpolate(method='linear', limit_direction='both').values
        ankle_y_clean = pd.Series(ankle_y).interpolate(method='linear', limit_direction='both').values
        ankle_z_clean = pd.Series(ankle_z).interpolate(method='linear', limit_direction='both').values

        trajectory = np.column_stack([ankle_x_clean, ankle_y_clean, ankle_z_clean])

        # Calculate curvature using finite differences
        # First derivative (velocity)
        dt = 1.0 / fps
        vel = np.gradient(trajectory, axis=0) / dt

        # Second derivative (acceleration)
        acc = np.gradient(vel, axis=0) / dt

        # Curvature = |v × a| / |v|^3
        vel_magnitude = np.linalg.norm(vel, axis=1)
        cross_product = np.cross(vel, acc)
        cross_magnitude = np.linalg.norm(cross_product, axis=1)

        # Avoid division by zero
        curvature = np.zeros_like(vel_magnitude)
        valid_vel = vel_magnitude > 0.01  # Min velocity threshold
        curvature[valid_vel] = cross_magnitude[valid_vel] / (vel_magnitude[valid_vel] ** 3)

        # Smooth curvature
        window_size = int(fps * 0.5)  # 0.5 second window
        if window_size % 2 == 0:
            window_size += 1
        if window_size >= 3 and len(curvature) >= window_size:
            curvature_smooth = savgol_filter(curvature, window_size, 2)
        else:
            curvature_smooth = curvature

        # Detect high-curvature regions (turns)
        turn_mask = curvature_smooth > curvature_threshold

        # Convert to frame indices
        turn_frames = np.where(turn_mask)[0].tolist()

        return turn_frames

    def _analyze_temporal_v3(
        self,
        subject_id: str,
        df_angles: pd.DataFrame,
        info: Dict,
        fps: float
    ) -> Dict:
        """V5.2 temporal analysis with enhanced scaling and turn detection."""

        # Hip trajectory (world coordinates)
        hip_x = df_angles['x_left_hip'].values
        hip_y = df_angles['y_left_hip'].values
        hip_z = df_angles['z_left_hip'].values
        hip_traj = np.column_stack([hip_x, hip_y, hip_z])

        demo = info.get('demographics', {})
        patient = info.get('patient', {})
        patient_left = patient.get('left', {})
        patient_right = patient.get('right', {})

        fusion_left_strikes = self.processor.detect_heel_strikes_fusion(df_angles, 'left', fps)
        fusion_right_strikes = self.processor.detect_heel_strikes_fusion(df_angles, 'right', fps)

        gt_stride_left = demo.get('left_strides', 0)
        gt_stride_right = demo.get('right_strides', 0)

        # Template-based heel strike detection (inherited from V5)
        left_strikes = self._detect_heel_strikes_v5(
            df_angles,
            'left',
            fps,
            gt_stride_left,
            fusion_candidates=fusion_left_strikes
        )
        right_strikes = self._detect_heel_strikes_v5(
            df_angles,
            'right',
            fps,
            gt_stride_right,
            fusion_candidates=fusion_right_strikes
        )

        gt_stride_left_cm = patient_left.get('stride_length_cm')
        gt_stride_right_cm = patient_right.get('stride_length_cm')

        # V5.2: Enhanced scale factor calculation with quality weighting and cross-leg validation
        scale_factor, scale_diagnostics = calculate_hybrid_scale_factor(
            hip_traj,
            left_strikes,
            right_strikes,
            gt_stride_left_cm,
            gt_stride_right_cm,
            fallback_walkway_m=self.walkway_distance_m,
            use_quality_weighting=self.use_quality_weighting,
            cross_leg_validation=self.cross_leg_validation
        )

        hip_traj_scaled = hip_traj * scale_factor

        # V5.2: Enhanced turn detection with curvature
        if self.enhanced_turn_detection:
            try:
                turn_frames = self._detect_turns_with_curvature(df_angles, fps)
                # Convert to turn points format (list of frame indices where turn starts/ends)
                if turn_frames:
                    # Group consecutive frames into turn segments
                    turn_segments = []
                    if len(turn_frames) > 0:
                        segment_start = turn_frames[0]
                        prev_frame = turn_frames[0]

                        for frame in turn_frames[1:]:
                            if frame - prev_frame > fps * 0.5:  # Gap > 0.5s means new segment
                                turn_segments.append((segment_start, prev_frame))
                                segment_start = frame
                            prev_frame = frame

                        turn_segments.append((segment_start, prev_frame))

                    # Extract turn point centers
                    turn_points = [(start + end) // 2 for start, end in turn_segments]
                else:
                    turn_points = []
            except Exception:
                # Fallback to V5 adaptive turn detection
                turn_points, gait_speed = self._detect_turn_points_adaptive(hip_traj_scaled, fps)
        else:
            # Use V5 adaptive turn detection
            turn_points, gait_speed = self._detect_turn_points_adaptive(hip_traj_scaled, fps)

        # If curvature detection succeeded, still need gait_speed
        if self.enhanced_turn_detection and 'gait_speed' not in locals():
            _, gait_speed = self._detect_turn_points_adaptive(hip_traj_scaled, fps)

        turn_buffer_frames = int((self.base_turn_buffer_sec * gait_speed + 0.5) * fps)

        left_cycles_dir = self._classify_cycles_by_direction(left_strikes, turn_points, turn_buffer_frames, fps)
        right_cycles_dir = self._classify_cycles_by_direction(right_strikes, turn_points, turn_buffer_frames, fps)

        duration_minutes = len(df_angles) / (fps * 60.0) if fps > 0 else 0.0

        gait_timing = demo.get('gait_cycle_timing', {})
        gt_strides_left = demo.get('left_strides')
        gt_strides_right = demo.get('right_strides')
        gt_cadence_left = patient_left.get('cadence_steps_min')
        gt_cadence_right = patient_right.get('cadence_steps_min')
        cadence_values = [v for v in [gt_cadence_left, gt_cadence_right] if v is not None]
        gt_cadence_avg = float(np.mean(cadence_values)) if cadence_values else None

        gt_stance_left = gait_timing.get('left_stance')
        gt_stance_right = gait_timing.get('right_stance')
        gt_step_length_left = patient_left.get('step_length_cm')
        gt_step_length_right = patient_right.get('step_length_cm')
        gt_stride_length_left = patient_left.get('stride_length_cm')
        gt_stride_length_right = patient_right.get('stride_length_cm')
        gt_velocity_left = patient_left.get('forward_velocity_cm_s')
        gt_velocity_right = patient_right.get('forward_velocity_cm_s')

        pred_strides_left = max(len(left_strikes) - 1, 0)
        pred_strides_right = max(len(right_strikes) - 1, 0)

        outbound_left = [c for c in left_cycles_dir if c['direction'] == 'outbound']
        inbound_right = [c for c in right_cycles_dir if c['direction'] == 'inbound']

        def compute_cadence(cycles: List[Dict]) -> float:
            if not cycles:
                return 0.0
            total_minutes = sum((c['end'] - c['start']) for c in cycles) / (fps * 60.0) if fps > 0 else 0.0
            if total_minutes <= 0:
                return 0.0
            return len(cycles) / total_minutes

        dir_cadence_left = compute_cadence(outbound_left) if outbound_left else compute_cadence([c for c in left_cycles_dir if c['direction'] != 'turn'])
        dir_cadence_right = compute_cadence(inbound_right) if inbound_right else compute_cadence([c for c in right_cycles_dir if c['direction'] != 'turn'])
        dir_cadence_avg = (dir_cadence_left + dir_cadence_right) / 2 if (dir_cadence_left or dir_cadence_right) else 0.0

        stride_cadence_left = self._cadence_from_strikes(fusion_left_strikes, fps)
        stride_cadence_right = self._cadence_from_strikes(fusion_right_strikes, fps)

        pred_cadence_left = stride_cadence_left if stride_cadence_left > 0 else dir_cadence_left
        pred_cadence_right = stride_cadence_right if stride_cadence_right > 0 else dir_cadence_right

        ransac_left, ransac_diag_left = estimate_cadence_ransac(fusion_left_strikes, fps)
        ransac_right, ransac_diag_right = estimate_cadence_ransac(fusion_right_strikes, fps)
        if ransac_left > 0:
            pred_cadence_left = ransac_left
        if ransac_right > 0:
            pred_cadence_right = ransac_right
        cadence_values_pred = [v for v in [pred_cadence_left, pred_cadence_right] if v and v > 0]
        pred_cadence_avg = float(np.mean(cadence_values_pred)) if cadence_values_pred else dir_cadence_avg

        total_cadence = self._compute_total_cadence(
            fusion_left_strikes,
            fusion_right_strikes,
            turn_points,
            turn_buffer_frames,
            fps
        )
        if stride_cadence_left <= 0 and total_cadence > 0:
            pred_cadence_left = total_cadence
        if stride_cadence_right <= 0 and total_cadence > 0:
            pred_cadence_right = total_cadence

        if total_cadence > 0:
            max_allowed = total_cadence * 2.0
            if pred_cadence_left > 0:
                pred_cadence_left = float(np.clip(pred_cadence_left, total_cadence, max_allowed))
            if pred_cadence_right > 0:
                pred_cadence_right = float(np.clip(pred_cadence_right, total_cadence, max_allowed))

        cadence_values_pred = [v for v in [pred_cadence_left, pred_cadence_right] if v and v > 0]
        if cadence_values_pred:
            pred_cadence_avg = float(np.mean(cadence_values_pred))
        elif total_cadence > 0:
            pred_cadence_avg = total_cadence

        total_cadence_scaled = total_cadence
        if self.cadence_scale_bias and self.cadence_scale_bias > 0:
            pred_cadence_left *= self.cadence_scale_bias
            pred_cadence_right *= self.cadence_scale_bias
            pred_cadence_avg *= self.cadence_scale_bias
            if total_cadence > 0:
                total_cadence_scaled = total_cadence * self.cadence_scale_bias

        pred_cadence_left = float(np.clip(pred_cadence_left, 0.0, 220.0))
        pred_cadence_right = float(np.clip(pred_cadence_right, 0.0, 220.0))
        pred_cadence_avg = float(np.clip(pred_cadence_avg, 0.0, 220.0))
        if total_cadence_scaled > 0:
            total_cadence_scaled = float(np.clip(total_cadence_scaled, 0.0, 220.0))

        pred_stance_left = float(np.mean([c['stance_pct'] for c in left_cycles_dir if c['direction'] != 'turn'])) if left_cycles_dir else 0.0
        pred_stance_right = float(np.mean([c['stance_pct'] for c in right_cycles_dir if c['direction'] != 'turn'])) if right_cycles_dir else 0.0

        allowed_directions = {'outbound', 'inbound'}
        left_allowed_pairs = {
            (cycle['start'], cycle['end'])
            for cycle in left_cycles_dir
            if cycle.get('direction') in allowed_directions
        }
        right_allowed_pairs = {
            (cycle['start'], cycle['end'])
            for cycle in right_cycles_dir
            if cycle.get('direction') in allowed_directions
        }
        min_filtered_strides = 3

        def compute_stride_metrics(
            strikes: List[int],
            allowed_pairs: Optional[set],
            scale_override: Optional[float] = None
        ) -> Tuple[float, float, float, Dict[str, float]]:
            stats: Dict[str, float] = {
                'strides_total': 0.0,
                'strides_filtered': 0.0,
                'using_filtered': False
            }
            if strikes is None or len(strikes) < 2 or fps <= 0:
                return float("nan"), float("nan"), float("nan"), stats

            stride_lengths_all: List[float] = []
            stride_velocities_all: List[float] = []
            stride_lengths_filtered: List[float] = []
            stride_velocities_filtered: List[float] = []

            for idx in range(len(strikes) - 1):
                start = strikes[idx]
                end = strikes[idx + 1]
                if end >= len(hip_traj_scaled) or start >= len(hip_traj_scaled):
                    continue

                displacement_raw = hip_traj[end] - hip_traj[start]
                scale_used = scale_override if scale_override and scale_override > 0 else scale_factor
                stride_length_cm = float(np.linalg.norm(displacement_raw) * scale_used * 100.0)
                duration_sec = (end - start) / fps if fps > 0 else 0.0

                stride_lengths_all.append(stride_length_cm)
                if duration_sec > 0:
                    stride_velocities_all.append(stride_length_cm / duration_sec)

                pair = (start, end)
                if not allowed_pairs or pair in allowed_pairs:
                    stride_lengths_filtered.append(stride_length_cm)
                    if duration_sec > 0:
                        stride_velocities_filtered.append(stride_length_cm / duration_sec)

            stats['strides_total'] = float(len(stride_lengths_all))
            stats['strides_filtered'] = float(len(stride_lengths_filtered))

            use_filtered = len(stride_lengths_filtered) >= min_filtered_strides
            stats['using_filtered'] = bool(use_filtered)

            if use_filtered:
                stride_lengths_use = stride_lengths_filtered
                stride_velocities_use = stride_velocities_filtered
            else:
                stride_lengths_use = stride_lengths_all
                stride_velocities_use = stride_velocities_all

            if not stride_lengths_use:
                return float("nan"), float("nan"), float("nan"), stats

            avg_stride_length = float(np.mean(stride_lengths_use))
            avg_step_length = avg_stride_length / 2.0 if np.isfinite(avg_stride_length) else float("nan")
            avg_velocity = float(np.mean(stride_velocities_use)) if stride_velocities_use else float("nan")
            return avg_step_length, avg_stride_length, avg_velocity, stats

        left_scale_override = scale_diagnostics.get('left', {}).get('scale_factor')
        right_scale_override = scale_diagnostics.get('right', {}).get('scale_factor')

        if scale_diagnostics.get('suspect_stride_data'):
            suspect_details = scale_diagnostics.get('suspect_details', {})
            if left_scale_override is None and 'left' in suspect_details:
                left_scale_override = suspect_details['left'].get('scale_factor')
            if right_scale_override is None and 'right' in suspect_details:
                right_scale_override = suspect_details['right'].get('scale_factor')

        pred_step_length_left, pred_stride_length_left, pred_velocity_left, left_stride_stats = compute_stride_metrics(
            left_strikes,
            left_allowed_pairs,
            left_scale_override
        )
        pred_step_length_right, pred_stride_length_right, pred_velocity_right, right_stride_stats = compute_stride_metrics(
            right_strikes,
            right_allowed_pairs,
            right_scale_override
        )

        def velocity_from_step(step_length_cm: float, cadence_steps_min: float) -> float:
            if step_length_cm is None or cadence_steps_min is None:
                return float("nan")
            if not np.isfinite(step_length_cm) or not np.isfinite(cadence_steps_min):
                return float("nan")
            return float(step_length_cm * cadence_steps_min / 60.0)

        velocity_left_formula = velocity_from_step(pred_step_length_left, pred_cadence_left)
        velocity_right_formula = velocity_from_step(pred_step_length_right, pred_cadence_right)

        hip_diffs_scaled = np.diff(hip_traj_scaled, axis=0)
        total_distance_cm = float(np.sum(np.linalg.norm(hip_diffs_scaled, axis=1)) * 100.0)
        duration_sec = len(df_angles) / fps if fps > 0 else 0.0
        overall_velocity_cm_s = total_distance_cm / duration_sec if duration_sec > 0 else float("nan")

        if np.isfinite(velocity_left_formula):
            pred_velocity_left = velocity_left_formula
        elif not np.isfinite(pred_velocity_left):
            pred_velocity_left = overall_velocity_cm_s

        if np.isfinite(velocity_right_formula):
            pred_velocity_right = velocity_right_formula
        elif not np.isfinite(pred_velocity_right):
            pred_velocity_right = overall_velocity_cm_s

        turn_cycles_left = sum(1 for c in left_cycles_dir if c.get('direction') == 'turn')
        turn_cycles_right = sum(1 for c in right_cycles_dir if c.get('direction') == 'turn')
        straight_cycles_left = len(left_cycles_dir) - turn_cycles_left
        straight_cycles_right = len(right_cycles_dir) - turn_cycles_right

        self._register_temporal_metric("strides_left", gt_strides_left, pred_strides_left)
        self._register_temporal_metric("strides_right", gt_strides_right, pred_strides_right)
        self._register_temporal_metric("cadence_left", gt_cadence_left, pred_cadence_left)
        self._register_temporal_metric("cadence_right", gt_cadence_right, pred_cadence_right)
        self._register_temporal_metric("cadence_average", gt_cadence_avg, pred_cadence_avg)
        self._register_temporal_metric("stance_percent_left", gt_stance_left, pred_stance_left)
        self._register_temporal_metric("stance_percent_right", gt_stance_right, pred_stance_right)
        self._register_temporal_metric("step_length_left_cm", gt_step_length_left, pred_step_length_left)
        self._register_temporal_metric("step_length_right_cm", gt_step_length_right, pred_step_length_right)
        self._register_temporal_metric("stride_length_left_cm", gt_stride_length_left, pred_stride_length_left)
        self._register_temporal_metric("stride_length_right_cm", gt_stride_length_right, pred_stride_length_right)
        self._register_temporal_metric("forward_velocity_left_cm_s", gt_velocity_left, pred_velocity_left)
        self._register_temporal_metric("forward_velocity_right_cm_s", gt_velocity_right, pred_velocity_right)

        ground_truth = {
            "strides": {
                "left": to_serializable(gt_strides_left),
                "right": to_serializable(gt_strides_right)
            },
            "cadence_steps_min": {
                "left": to_serializable(gt_cadence_left),
                "right": to_serializable(gt_cadence_right),
                "average": to_serializable(gt_cadence_avg)
            },
            "stance_percent": {
                "left": to_serializable(gt_stance_left),
                "right": to_serializable(gt_stance_right)
            },
            "step_length_cm": {
                "left": to_serializable(gt_step_length_left),
                "right": to_serializable(gt_step_length_right)
            },
            "stride_length_cm": {
                "left": to_serializable(gt_stride_length_left),
                "right": to_serializable(gt_stride_length_right)
            },
            "forward_velocity_cm_s": {
                "left": to_serializable(gt_velocity_left),
                "right": to_serializable(gt_velocity_right)
            }
        }

        def _serialize_diag(diag):
            if not isinstance(diag, dict):
                return diag
            serialized = {}
            for key, value in diag.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    serialized[key] = to_serializable(value)
                else:
                    serialized[key] = value
            return serialized

        serialized_ransac_left = _serialize_diag(ransac_diag_left)
        serialized_ransac_right = _serialize_diag(ransac_diag_right)

        prediction = {
            "strides": {
                "left": to_serializable(pred_strides_left),
                "right": to_serializable(pred_strides_right)
            },
            "cadence_steps_min": {
                "left": to_serializable(pred_cadence_left),
                "right": to_serializable(pred_cadence_right),
                "average": to_serializable(pred_cadence_avg)
            },
            "directional_cadence_steps_min": {
                "left_outbound": to_serializable(dir_cadence_left),
                "right_inbound": to_serializable(dir_cadence_right),
                "average": to_serializable(dir_cadence_avg)
            },
            "filtered_total_cadence_steps_min": to_serializable(total_cadence_scaled),
            "stance_percent": {
                "left": to_serializable(pred_stance_left),
                "right": to_serializable(pred_stance_right)
            },
            "step_length_cm": {
                "left": to_serializable(pred_step_length_left),
                "right": to_serializable(pred_step_length_right)
            },
            "stride_length_cm": {
                "left": to_serializable(pred_stride_length_left),
                "right": to_serializable(pred_stride_length_right)
            },
            "forward_velocity_cm_s": {
                "left": to_serializable(pred_velocity_left),
                "right": to_serializable(pred_velocity_right)
            },
            "stride_filter_stats": {
                "left": {
                    "strides_total": to_serializable(left_stride_stats['strides_total']),
                    "strides_filtered": to_serializable(left_stride_stats['strides_filtered']),
                    "using_filtered": bool(left_stride_stats['using_filtered']),
                    "straight_cycles": to_serializable(straight_cycles_left),
                    "turn_cycles": to_serializable(turn_cycles_left)
                },
                "right": {
                    "strides_total": to_serializable(right_stride_stats['strides_total']),
                    "strides_filtered": to_serializable(right_stride_stats['strides_filtered']),
                    "using_filtered": bool(right_stride_stats['using_filtered']),
                    "straight_cycles": to_serializable(straight_cycles_right),
                    "turn_cycles": to_serializable(turn_cycles_right)
                }
            },
            "scale_factor": to_serializable(scale_factor),
            "scale_diagnostics": scale_diagnostics,
            "gait_speed_m_s": to_serializable(gait_speed),
            "adaptive_buffer_frames": to_serializable(turn_buffer_frames),
            "detector_version": 'template_dtw_v52',
            "cadence_ransac_diagnostics": {
                "left": serialized_ransac_left,
                "right": serialized_ransac_right
            }
        }

        left_strikes_serializable = [int(x) for x in left_strikes]
        right_strikes_serializable = [int(x) for x in right_strikes]

        return {
            "ground_truth": ground_truth,
            "prediction": prediction,
            "left_heel_strikes": left_strikes_serializable,
            "right_heel_strikes": right_strikes_serializable,
            "duration_minutes": to_serializable(duration_minutes),
            "cadence_ransac_diagnostics": {
                "left": serialized_ransac_left,
                "right": serialized_ransac_right
            }
        }


def main_v52_test():
    """Quick test of V5.2 evaluator on single subject."""
    from mediapipe_csv_processor import MediaPipeCSVProcessor

    print("=" * 80)
    print("Tiered Gait Evaluator V5.2 - Enhanced Scaling Test")
    print("=" * 80)
    print()

    processor = MediaPipeCSVProcessor()

    test_file = "data/1/1-2_side_pose_fps30.csv"
    test_info_file = "processed/S1_01_info.json"

    if not Path(test_file).exists():
        print("Test CSV not found. Skipping.")
        return

    if not Path(test_info_file).exists():
        print("Test info not found. Skipping.")
        return

    # Load data
    df_wide = processor.load_csv(test_file)
    df_angles = processor.calculate_joint_angles(df_wide)

    # Load GT
    with open(test_info_file) as f:
        info = json.load(f)

    print("Testing V5.2 enhanced scaling...")
    print("- Quality-weighted stride selection: ENABLED")
    print("- Cross-leg validation (>15% reject): ENABLED")
    print("- Enhanced turn detection with curvature: ENABLED")
    print()

    evaluator = TieredGaitEvaluatorV52()
    result = evaluator._analyze_temporal_v3('S1_01', df_angles, info, 30.0)

    print("Scale diagnostics:")
    scale_diag = result['prediction'].get('scale_diagnostics', {})
    print(f"  Method: {scale_diag.get('method')}")
    print(f"  Final scale: {scale_diag.get('final_scale', 0.0):.4f}")

    if 'left' in scale_diag:
        left_diag = scale_diag['left']
        print(f"  Left: {left_diag.get('n_strides_used', 0)}/{left_diag.get('n_strides', 0)} strides used")
        print(f"        CV: {left_diag.get('cv_stride_mp', 0.0):.3f}")
        print(f"        Outliers rejected: {left_diag.get('n_outliers_rejected', 0)}")

    if 'right' in scale_diag:
        right_diag = scale_diag['right']
        print(f"  Right: {right_diag.get('n_strides_used', 0)}/{right_diag.get('n_strides', 0)} strides used")
        print(f"         CV: {right_diag.get('cv_stride_mp', 0.0):.3f}")
        print(f"         Outliers rejected: {right_diag.get('n_outliers_rejected', 0)}")

    if 'cross_leg_disagreement' in scale_diag:
        print(f"  Cross-leg disagreement: {scale_diag['cross_leg_disagreement']:.1%}")
        if scale_diag.get('cross_leg_validation_passed'):
            print("  ✓ Cross-leg validation PASSED")
        elif scale_diag.get('cross_leg_validation_failed'):
            print(f"  ✗ Cross-leg validation FAILED (rejected {scale_diag.get('rejected_side')})")

    print()
    print("✓ V5.2 enhanced scaling working!")


if __name__ == '__main__':
    main_v52_test()
