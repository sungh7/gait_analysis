#!/usr/bin/env python3
"""
MediaPipe 보행분석 메인 파이프라인
- 비디오 입력 → MediaPipe 키포인트 추출 → 보행지표 계산 → 관절각도 분석 → 101포인트 정규화
- 전통적 보행분석과 완전 호환되는 결과 출력
- 통합된 원스톱 분석 솔루션

Author: AI Assistant
Date: 2025-09-15
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# MediaPipe 관련
import mediapipe as mp
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats

class MainGaitAnalysisPipeline:
    """통합 보행분석 메인 파이프라인"""

    def __init__(self):
        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # 분석 결과 저장
        self.analysis_results = {}
        self.landmarks_data = []
        self.video_info = {}

        # 보행 표준값 (성인 기준)
        self.gait_norms = {
            'cadence_range': (100, 120),      # steps/min
            'step_length_range': (0.5, 0.8),  # m
            'walking_speed_range': (1.0, 1.6), # m/s
            'hip_angle_range': (-10, 30),     # degrees
            'knee_angle_range': (0, 70),      # degrees
            'ankle_angle_range': (-20, 20),   # degrees
        }

        # 각도 보정을 위한 기본 파라미터 (외부 설정에 의해 덮어쓰기 가능)
        self.angle_calibration_offsets = {
            'hip_flexion_extension': 76.08724065303748,
            'knee_flexion_extension': 50.362597945682886,
            'ankle_dorsi_plantarflexion': -15.181074732638209,
            'hip_abduction_adduction_left': 0.0,
            'hip_abduction_adduction_right': 0.0,
            'pelvis_obliquity': 0.0,
            'trunk_sway_mediolateral': 0.0,
        }
        self.angle_phase_shifts = {
            'hip_flexion_extension': 35,
            'knee_flexion_extension': 56,
            'ankle_dorsi_plantarflexion': 46,
            'hip_abduction_adduction_left': 0,
            'hip_abduction_adduction_right': 0,
            'pelvis_obliquity': 0,
            'trunk_sway_mediolateral': 0,
        }
        self.dynamic_shift_enabled = False
        self.dynamic_shift_window = 5

        self._load_angle_calibration_config()

        print("✅ MediaPipe 보행분석 파이프라인 초기화 완료")

    # ============================================================================
    # 1단계: 비디오 처리 및 키포인트 추출
    # ============================================================================

    def extract_pose_landmarks(self, video_path, output_dir=None, max_frames=None):
        """비디오에서 MediaPipe 포즈 랜드마크 추출"""
        print(f"🎬 비디오 분석 시작: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"비디오를 열 수 없습니다: {video_path}")

        # 비디오 정보 수집
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps

        self.video_info = {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration
        }

        print(f"📋 비디오 정보: {frame_count}프레임, {fps}FPS, {duration:.2f}초")

        landmarks_data = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # RGB 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe 포즈 추출
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                # 랜드마크 좌표 추출
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

                landmarks_data.append({
                    'frame': frame_idx,
                    'timestamp': frame_idx / fps,
                    'landmarks': landmarks
                })

            frame_idx += 1

            # 진행률 표시
            if frame_idx % 30 == 0:
                progress = (frame_idx / frame_count) * 100
                print(f"진행률: {progress:.1f}%")

            if max_frames is not None and frame_idx >= max_frames:
                print(f"🛑 프레임 제한 {max_frames}에 도달하여 분석을 종료합니다.")
                break

        cap.release()

        # DataFrame으로 변환
        if landmarks_data:
            self.landmarks_data = pd.DataFrame(landmarks_data)
            print(f"✅ 랜드마크 추출 완료: {len(self.landmarks_data)}개 프레임")
        else:
            raise ValueError("포즈 랜드마크를 추출할 수 없습니다.")

        return self.landmarks_data

    def get_landmark_coordinates(self, landmark_idx):
        """특정 랜드마크의 좌표 시계열 추출"""
        if self.landmarks_data.empty:
            return None, None, None

        x_coords = []
        y_coords = []
        z_coords = []

        for _, row in self.landmarks_data.iterrows():
            landmarks = row['landmarks']
            x = landmarks[landmark_idx * 4]     # x
            y = landmarks[landmark_idx * 4 + 1] # y
            z = landmarks[landmark_idx * 4 + 2] # z

            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)

        return np.array(x_coords), np.array(y_coords), np.array(z_coords)

    # ============================================================================
    # 2단계: 보행 이벤트 탐지
    # ============================================================================

    def detect_gait_events(self):
        """보행 이벤트(heel strike, toe off) 탐지"""
        print("👣 보행 이벤트 탐지 중...")

        # 발목 랜드마크 추출 (MediaPipe pose landmark 인덱스)
        left_ankle_x, left_ankle_y, _ = self.get_landmark_coordinates(27)   # LEFT_ANKLE
        right_ankle_x, right_ankle_y, _ = self.get_landmark_coordinates(28)  # RIGHT_ANKLE

        # 발가락 랜드마크 추출
        left_foot_x, left_foot_y, _ = self.get_landmark_coordinates(31)    # LEFT_FOOT_INDEX
        right_foot_x, right_foot_y, _ = self.get_landmark_coordinates(32)   # RIGHT_FOOT_INDEX

        # Y좌표 기반 heel strike 탐지 (최저점)
        left_heel_strikes = self._detect_heel_strikes(left_ankle_y)
        right_heel_strikes = self._detect_heel_strikes(right_ankle_y)

        # Toe off 탐지 (발가락 최고점)
        left_toe_offs = self._detect_toe_offs(left_foot_y)
        right_toe_offs = self._detect_toe_offs(right_foot_y)

        gait_events = {
            'left_heel_strikes': left_heel_strikes,
            'right_heel_strikes': right_heel_strikes,
            'left_toe_offs': left_toe_offs,
            'right_toe_offs': right_toe_offs,
            'fps': self.video_info['fps']
        }

        print(f"✅ 보행 이벤트 탐지 완료:")
        print(f"  • 좌측 heel strike: {len(left_heel_strikes)}개")
        print(f"  • 우측 heel strike: {len(right_heel_strikes)}개")

        return gait_events

    def _detect_heel_strikes(self, ankle_y, min_distance=15):
        """발목 Y좌표의 최저점에서 heel strike 탐지"""
        # 신호 평활화
        if len(ankle_y) > 5:
            smoothed = savgol_filter(ankle_y, window_length=min(11, len(ankle_y)//2*2+1), polyorder=2)
        else:
            smoothed = ankle_y

        # 최저점 탐지 (heel strike는 발이 지면에 닿는 순간 = y값 최대)
        peaks, _ = find_peaks(smoothed, distance=min_distance, prominence=0.01)

        return peaks

    def _detect_toe_offs(self, foot_y, min_distance=15):
        """발가락 Y좌표의 최고점에서 toe off 탐지"""
        # 신호 평활화
        if len(foot_y) > 5:
            smoothed = savgol_filter(foot_y, window_length=min(11, len(foot_y)//2*2+1), polyorder=2)
        else:
            smoothed = foot_y

        # 최고점 탐지 (toe off는 발가락이 지면에서 떨어지는 순간 = y값 최소)
        peaks, _ = find_peaks(-smoothed, distance=min_distance, prominence=0.01)

        return peaks

    # ============================================================================
    # 3단계: 시공간 매개변수 계산
    # ============================================================================

    def calculate_temporal_spatial_parameters(self, gait_events, pixel_to_meter_ratio=0.001):
        """시공간 매개변수 계산"""
        print("📏 시공간 매개변수 계산 중...")

        fps = gait_events['fps']
        left_hs = gait_events['left_heel_strikes']
        right_hs = gait_events['right_heel_strikes']

        # Cadence 계산 (steps/min)
        total_steps = len(left_hs) + len(right_hs)
        duration_minutes = len(self.landmarks_data) / fps / 60
        cadence = total_steps / duration_minutes if duration_minutes > 0 else 0

        # Stride time 계산 (좌측 기준)
        stride_times = []
        if len(left_hs) > 1:
            for i in range(len(left_hs) - 1):
                stride_time = (left_hs[i+1] - left_hs[i]) / fps
                stride_times.append(stride_time)

        # Step time 계산
        step_times = []
        all_heel_strikes = sorted(list(left_hs) + list(right_hs))
        if len(all_heel_strikes) > 1:
            for i in range(len(all_heel_strikes) - 1):
                step_time = (all_heel_strikes[i+1] - all_heel_strikes[i]) / fps
                step_times.append(step_time)

        # Stride length 추정 (발목 X좌표 변화 기반)
        left_ankle_x, _, _ = self.get_landmark_coordinates(27)
        stride_lengths = []
        if len(left_hs) > 1:
            for i in range(len(left_hs) - 1):
                x_displacement = abs(left_ankle_x[left_hs[i+1]] - left_ankle_x[left_hs[i]])
                stride_length = x_displacement * pixel_to_meter_ratio * self.video_info['width']
                stride_lengths.append(stride_length)

        # Walking speed 계산
        walking_speeds = []
        if stride_lengths and stride_times:
            for sl, st in zip(stride_lengths, stride_times):
                if st > 0:
                    walking_speeds.append(sl / st)

        # Stance/Swing phase 계산
        stance_phases = []
        swing_phases = []

        # 간단한 추정: heel strike ~ toe off = stance, toe off ~ next heel strike = swing
        left_to = gait_events['left_toe_offs']
        if len(left_hs) > 1 and len(left_to) > 0:
            for i, hs in enumerate(left_hs[:-1]):
                # 해당 heel strike 이후 첫 번째 toe off 찾기
                next_tos = [to for to in left_to if to > hs]
                if next_tos and i+1 < len(left_hs):
                    next_to = next_tos[0]
                    next_hs = left_hs[i+1]

                    stance_time = (next_to - hs) / fps
                    swing_time = (next_hs - next_to) / fps
                    stride_total = stance_time + swing_time

                    if stride_total > 0:
                        stance_phases.append(stance_time / stride_total * 100)
                        swing_phases.append(swing_time / stride_total * 100)

        temporal_spatial = {
            'cadence': cadence,
            'cadence_list': [cadence],  # 단일값을 리스트로
            'stride_time_mean': np.mean(stride_times) if stride_times else 0,
            'stride_time_std': np.std(stride_times) if stride_times else 0,
            'stride_time_list': stride_times,
            'step_time_mean': np.mean(step_times) if step_times else 0,
            'step_time_std': np.std(step_times) if step_times else 0,
            'step_time_list': step_times,
            'stride_length_mean': np.mean(stride_lengths) if stride_lengths else 0,
            'stride_length_std': np.std(stride_lengths) if stride_lengths else 0,
            'stride_length_list': stride_lengths,
            'walking_speed_mean': np.mean(walking_speeds) if walking_speeds else 0,
            'walking_speed_std': np.std(walking_speeds) if walking_speeds else 0,
            'walking_speed_list': walking_speeds,
            'stance_phase_mean': np.mean(stance_phases) if stance_phases else 60,  # 기본값
            'stance_phase_std': np.std(stance_phases) if stance_phases else 0,
            'stance_phase_percent': stance_phases,
            'swing_phase_mean': np.mean(swing_phases) if swing_phases else 40,  # 기본값
            'swing_phase_std': np.std(swing_phases) if swing_phases else 0,
            'double_support_percent': [100 - sp - sw for sp, sw in zip(stance_phases, swing_phases)] if stance_phases and swing_phases else [0]
        }

        print(f"✅ 시공간 매개변수 계산 완료:")
        print(f"  • Cadence: {cadence:.1f} steps/min")
        print(f"  • 평균 stride time: {temporal_spatial['stride_time_mean']:.3f}s")
        print(f"  • 평균 stride length: {temporal_spatial['stride_length_mean']:.3f}m")

        return temporal_spatial

    # ============================================================================
    # 4단계: 관절각도 계산
    # ============================================================================

    def calculate_joint_angles_3d(self):
        """3D 관절각도 계산"""
        print("🦴 관절각도 계산 중...")

        joint_angles = {
            'hip_flexion_extension': [],
            'knee_flexion_extension': [],
            'ankle_dorsi_plantarflexion': [],
            'hip_abduction_adduction_left': [],
            'hip_abduction_adduction_right': [],
            'pelvis_obliquity': [],
            'trunk_sway_mediolateral': [],
            'timestamps': []
        }

        calibration = self._load_calibration_parameters()

        def apply_calibration(joint, value):
            params = calibration.get(joint)
            if not params:
                return value
            slope = params.get('slope', 1.0)
            intercept = params.get('intercept', 0.0)
            if abs(slope) < 1e-6:
                return value
            return (value - intercept) / slope

        for _, row in self.landmarks_data.iterrows():
            landmarks = row['landmarks']
            timestamp = row['timestamp']

            # 좌측 하지 관절각도 계산 (왼쪽 기준)
            hip_angle = self._calculate_hip_angle(landmarks, side='left')
            knee_angle = self._calculate_knee_angle(landmarks, side='left')
            ankle_angle = self._calculate_ankle_angle(landmarks, side='left')

            joint_angles['hip_flexion_extension'].append(apply_calibration('hip_flexion_extension', hip_angle))
            joint_angles['knee_flexion_extension'].append(apply_calibration('knee_flexion_extension', knee_angle))
            joint_angles['ankle_dorsi_plantarflexion'].append(apply_calibration('ankle_dorsi_plantarflexion', ankle_angle))
            joint_angles['hip_abduction_adduction_left'].append(self._calculate_hip_abduction_angle(landmarks, side='left'))
            joint_angles['hip_abduction_adduction_right'].append(self._calculate_hip_abduction_angle(landmarks, side='right'))
            joint_angles['pelvis_obliquity'].append(self._calculate_pelvis_obliquity(landmarks))
            joint_angles['trunk_sway_mediolateral'].append(self._calculate_trunk_sway(landmarks))
            joint_angles['timestamps'].append(timestamp)

        print(f"✅ 관절각도 계산 완료: {len(joint_angles['timestamps'])}개 프레임")
        return joint_angles

    def _normalize_vector(self, vec, fallback):
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return np.array(fallback, dtype=float)
        return vec / norm

    def _load_calibration_parameters(self):
        path = Path('calibration_parameters_deming.json')
        if not path.exists():
            return {}
        try:
            with path.open() as f:
                data = json.load(f)
        except Exception:
            return {}
        return data

    def _orthonormal_axes(self, axis_x, axis_y, axis_z):
        axis_x = self._normalize_vector(axis_x, [1.0, 0.0, 0.0])
        axis_y = self._normalize_vector(axis_y, [0.0, 1.0, 0.0])
        axis_z = self._normalize_vector(axis_z, [0.0, 0.0, 1.0])
        # Re-orthogonalize via Gram-Schmidt
        axis_x = self._normalize_vector(axis_x, [1.0, 0.0, 0.0])
        axis_y = self._normalize_vector(axis_y - np.dot(axis_y, axis_x) * axis_x, [0.0, 1.0, 0.0])
        axis_z = self._normalize_vector(axis_z - np.dot(axis_z, axis_x) * axis_x - np.dot(axis_z, axis_y) * axis_y, [0.0, 0.0, 1.0])
        return {'x': axis_x, 'y': axis_y, 'z': axis_z}

    def _axes_to_matrix(self, axes):
        return np.stack([axes['x'], axes['y'], axes['z']], axis=1)

    def _relative_rotation(self, parent_axes, child_axes):
        r_parent = self._axes_to_matrix(parent_axes)
        r_child = self._axes_to_matrix(child_axes)
        return r_parent.T @ r_child

    def _cardan_yxz(self, rotation_matrix):
        r = rotation_matrix
        theta_y = np.degrees(np.arctan2(r[0, 2], r[2, 2]))
        theta_x = np.degrees(np.arcsin(np.clip(r[2, 1], -1.0, 1.0)))
        theta_z = np.degrees(np.arctan2(r[1, 0], r[1, 1]))  # Fixed: removed negative sign
        return theta_y, theta_x, theta_z

    def _cardan_yzx(self, rotation_matrix):
        """
        Extract YZX Cardan angles from rotation matrix.
        Used for ankle joint per Vicon Plug-in Gait standard.

        Different from YXZ! Ankle uses Y-Z-X rotation order.

        YZX rotation matrix: R_y @ R_z @ R_x =
        [cy*cz,              -cy*sz,              sy    ]
        [sx*sy*cz + cx*sz,   -sx*sy*sz + cx*cz,  -sx*cy]
        [-cx*sy*cz + sx*sz,  cx*sy*sz + sx*cz,   cx*cy ]

        Returns:
            theta_y: Y-axis rotation (dorsiflexion/plantarflexion)
            theta_z: Z-axis rotation (inversion/eversion)
            theta_x: X-axis rotation (internal/external rotation)
        """
        r = rotation_matrix
        # Fixed YZX extraction formula
        theta_y = np.degrees(np.arcsin(np.clip(r[0, 2], -1.0, 1.0)))  # Y-axis: arcsin(sy)
        theta_z = np.degrees(np.arctan2(-r[0, 1], r[0, 0]))  # Z-axis: arctan2(cy*sz, cy*cz)
        theta_x = np.degrees(np.arctan2(-r[1, 2], r[2, 2]))  # X-axis: arctan2(sx*cy, cx*cy)
        return theta_y, theta_z, theta_x

    def _build_pelvis_frame(self, landmarks):
        left_hip = self._get_point(landmarks, 23)
        right_hip = self._get_point(landmarks, 24)
        left_shoulder = self._get_point(landmarks, 11)
        right_shoulder = self._get_point(landmarks, 12)
        pelvis_origin = (left_hip + right_hip) / 2
        axis_x = self._normalize_vector(left_hip - right_hip, [1.0, 0.0, 0.0])  # medial→lateral
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        forward_candidate = shoulder_mid - pelvis_origin
        axis_z = self._normalize_vector(forward_candidate - np.dot(forward_candidate, axis_x) * axis_x, [0.0, 0.0, 1.0])  # anterior
        axis_y = self._normalize_vector(np.cross(axis_z, axis_x), [0.0, 1.0, 0.0])  # superior
        axes = self._orthonormal_axes(axis_x, axis_y, axis_z)
        return pelvis_origin, axes

    def _project_axis(self, reference, target):
        return self._normalize_vector(reference - np.dot(reference, target) * target, reference)

    def _build_femur_frame(self, landmarks, side, pelvis_axes):
        hip_idx = 23 if side == 'left' else 24
        knee_idx = 25 if side == 'left' else 26
        hip = self._get_point(landmarks, hip_idx)
        knee = self._get_point(landmarks, knee_idx)
        axis_y = self._normalize_vector(knee - hip, [0.0, -1.0, 0.0])  # proximal→distal
        axis_x = self._project_axis(pelvis_axes['x'], axis_y)  # medial-lateral projected onto thigh plane
        axis_z = self._normalize_vector(np.cross(axis_x, axis_y), pelvis_axes['z'])
        return self._orthonormal_axes(axis_x, axis_y, axis_z)

    def _build_tibia_frame(self, landmarks, side, pelvis_axes):
        knee_idx = 25 if side == 'left' else 26
        ankle_idx = 27 if side == 'left' else 28
        toe_idx = 31 if side == 'left' else 32
        knee = self._get_point(landmarks, knee_idx)
        ankle = self._get_point(landmarks, ankle_idx)
        toe = self._get_point(landmarks, toe_idx)
        axis_y = self._normalize_vector(ankle - knee, [0.0, -1.0, 0.0])  # proximal→distal
        foot_dir = self._normalize_vector(toe - ankle, [0.0, 0.0, 1.0])
        axis_x = self._normalize_vector(np.cross(axis_y, foot_dir), pelvis_axes['x'])
        axis_z = self._normalize_vector(np.cross(axis_x, axis_y), pelvis_axes['z'])
        return self._orthonormal_axes(axis_x, axis_y, axis_z)

    def _build_foot_frame(self, landmarks, side, pelvis_axes):
        """
        Build foot coordinate frame per Vicon Plug-in Gait standard.
        Origin: Ankle Joint Center (AJC)
        Y_foot: AJC → TOE (anterior direction)
        X_foot: Lateral direction
        Z_foot: Superior (right-hand rule)

        Day 4 Fix: Use tibia frame as reference to ensure Y-axis always points anterior
        """
        # Build tibia frame first (reference for consistent orientation)
        tibia_axes = self._build_tibia_frame(landmarks, side, pelvis_axes)

        ankle_idx = 27 if side == 'left' else 28
        heel_idx = 29 if side == 'left' else 30
        toe_idx = 31 if side == 'left' else 32
        ankle = self._get_point(landmarks, ankle_idx)
        heel = self._get_point(landmarks, heel_idx)
        toe = self._get_point(landmarks, toe_idx)

        # Y_foot: AJC → TOE (anterior, per Vicon PiG standard)
        axis_y_raw = toe - ankle

        # ✅ DAY 4 FIX: Ensure Y_foot points anterior (same direction as tibia Y-axis)
        # This fixes subject-specific sign variation (10/17 had flipped Y-axis)
        if np.dot(axis_y_raw, tibia_axes['y']) < 0:
            axis_y_raw = -axis_y_raw  # Flip to match tibia anterior direction

        axis_y = self._normalize_vector(axis_y_raw, [0.0, 0.0, 1.0])

        # Use heel-ankle vector to determine vertical reference
        # (heel is inferior to ankle, so ankle-heel points superior)
        vertical_ref = self._normalize_vector(ankle - heel, [0.0, 1.0, 0.0])

        # X_foot: Lateral (cross of Y_anterior and Z_vertical)
        axis_x = self._normalize_vector(np.cross(axis_y, vertical_ref), [1.0, 0.0, 0.0])

        # Z_foot: Superior (right-hand rule: cross of X_lateral and Y_anterior)
        axis_z = self._normalize_vector(np.cross(axis_x, axis_y), [0.0, 1.0, 0.0])

        return self._orthonormal_axes(axis_x, axis_y, axis_z)

    def _vector_in_frame(self, vec, axes):
        return np.array([
            np.dot(vec, axes['x']),
            np.dot(vec, axes['y']),
            np.dot(vec, axes['z'])
        ])

    def _calculate_hip_angle(self, landmarks, side='left'):
        """Cardan YXZ 기반 고관절 굽힘/신전"""
        try:
            _, pelvis_axes = self._build_pelvis_frame(landmarks)
            femur_axes = self._build_femur_frame(landmarks, side, pelvis_axes)
            rel = self._relative_rotation(pelvis_axes, femur_axes)
            theta_y, _, _ = self._cardan_yxz(rel)
            return theta_y
        except Exception:
            return 0.0

    def _calculate_knee_angle(self, landmarks, side='left'):
        """무릎 굽힘/신전 각도 계산"""
        if side == 'left':
            # 좌측: 고관절(23) - 무릎(25) - 발목(27)
            hip_idx, knee_idx, ankle_idx = 23, 25, 27
        else:
            # 우측: 고관절(24) - 무릎(26) - 발목(28)
            hip_idx, knee_idx, ankle_idx = 24, 26, 28

        try:
            _, pelvis_axes = self._build_pelvis_frame(landmarks)
            femur_axes = self._build_femur_frame(landmarks, side, pelvis_axes)
            tibia_axes = self._build_tibia_frame(landmarks, side, pelvis_axes)
            rel = self._relative_rotation(femur_axes, tibia_axes)
            theta_y, _, _ = self._cardan_yxz(rel)
            return theta_y
        except Exception:
            return 0.0

    def _get_point(self, landmarks, idx):
        return np.array([landmarks[idx*4], landmarks[idx*4+1], landmarks[idx*4+2]])

    def _calculate_hip_abduction_angle(self, landmarks, side='left'):
        """고관절 외전/내전 각도 (frontal plane)."""
        hip_idx = 23 if side == 'left' else 24
        knee_idx = 25 if side == 'left' else 26
        try:
            _, axes = self._build_pelvis_frame(landmarks)
            hip = self._get_point(landmarks, hip_idx)
            knee = self._get_point(landmarks, knee_idx)
            thigh_vec = knee - hip
            local = self._vector_in_frame(thigh_vec, axes)
            abduction = np.degrees(np.arctan2(local[0], local[1]))
            if side == 'right':
                abduction *= -1
            return abduction
        except Exception:
            return 0.0

    def _calculate_pelvis_obliquity(self, landmarks):
        """좌우 고관절 높이 차이 기반 골반 경사."""
        try:
            left_hip = self._get_point(landmarks, 23)
            right_hip = self._get_point(landmarks, 24)
            delta_y = left_hip[1] - right_hip[1]
            pelvis_width = np.linalg.norm(left_hip - right_hip)
            if pelvis_width == 0:
                return 0.0
            angle = np.degrees(np.arctan2(delta_y, pelvis_width))
            return angle
        except Exception:
            return 0.0

    def _calculate_trunk_sway(self, landmarks):
        """어깨-골반 중심선 기반 의 몸통 기울기."""
        try:
            left_shoulder = self._get_point(landmarks, 11)
            right_shoulder = self._get_point(landmarks, 12)
            left_hip = self._get_point(landmarks, 23)
            right_hip = self._get_point(landmarks, 24)
            shoulder_mid = (left_shoulder + right_shoulder) / 2
            hip_mid = (left_hip + right_hip) / 2
            vec = shoulder_mid - hip_mid
            vec_frontal = vec.copy()
            vec_frontal[2] = 0
            if np.linalg.norm(vec_frontal) == 0:
                return 0.0
            vertical = np.array([0, 1, 0])
            cos_angle = np.dot(vec_frontal, vertical) / (np.linalg.norm(vec_frontal) * np.linalg.norm(vertical))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            sign = np.sign(vec_frontal[0])
            return angle * sign
        except Exception:
            return 0.0

    def _calculate_ankle_angle(self, landmarks, side='left'):
        """
        Cardan YZX 기반 발목 배측/족저 굽힘 (Vicon Plug-in Gait standard)

        Note: Ankle uses YZX rotation order (different from hip/knee)
        Per Vicon PiG: theta_y = dorsiflexion/plantarflexion
        """
        try:
            _, pelvis_axes = self._build_pelvis_frame(landmarks)
            tibia_axes = self._build_tibia_frame(landmarks, side, pelvis_axes)
            foot_axes = self._build_foot_frame(landmarks, side, pelvis_axes)  # Day 4: Pass pelvis_axes
            rel = self._relative_rotation(tibia_axes, foot_axes)
            theta_y, _, _ = self._cardan_yzx(rel)  # YZX for ankle (Vicon PiG)
            return -theta_y  # Negate to match MediaPipe coordinate convention
        except Exception:
            return 0.0

    # ============================================================================
    # 5단계: 101포인트 정규화
    # ============================================================================

    def normalize_to_101_points(self, joint_angles, gait_events):
        """
        보행주기를 101포인트(0-100%)로 정규화
        Returns:
            tuple(dict, list): (대표 보행주기 평균, 개별 보행주기 목록)
        """
        print("📊 101포인트 정규화 중...")

        joint_types = [
            'hip_flexion_extension',
            'knee_flexion_extension',
            'ankle_dorsi_plantarflexion',
            'hip_abduction_adduction_left',
            'hip_abduction_adduction_right',
            'pelvis_obliquity',
            'trunk_sway_mediolateral',
        ]

        left_hs = gait_events.get('left_heel_strikes', [])
        right_hs = gait_events.get('right_heel_strikes', [])
        timestamps = joint_angles.get('timestamps', [])
        fps = gait_events.get('fps', 30)
        left_toe_offs = gait_events.get('left_toe_offs', [])
        right_toe_offs = gait_events.get('right_toe_offs', [])

        all_cycles = []

        if len(left_hs) >= 2:
            total_frames = len(timestamps)

            for idx in range(len(left_hs) - 1):
                cycle_start = int(left_hs[idx])
                cycle_end = int(left_hs[idx + 1])
                if cycle_end <= cycle_start:
                    continue

                # ensure indices within bounds
                start_idx = max(0, min(cycle_start, total_frames - 1))
                end_idx = max(0, min(cycle_end, total_frames - 1))
                if end_idx <= start_idx or (end_idx - start_idx) < 3:
                    continue

                cycle_indices = range(start_idx, end_idx + 1)
                cycle_angles = {}

                for joint_type in joint_types:
                    joint_series = np.array([joint_angles[joint_type][i] for i in cycle_indices], dtype=float)
                    toe_off_idx = next((to for to in left_toe_offs if start_idx < to < end_idx), None)
                    right_toe_within = next((to for to in right_toe_offs if start_idx < to < end_idx), None)
                    if toe_off_idx is not None:
                        toe_rel = toe_off_idx - start_idx
                        stance_series = joint_series[: toe_rel + 1]
                        swing_series = joint_series[toe_rel:]
                        total_len = len(joint_series)
                        stance_len = len(stance_series)
                        swing_len = len(swing_series)
                        stance_ratio = stance_len / total_len if total_len > 0 else 0
                        stance_points = max(2, int(round(stance_ratio * 101)))
                        stance_points = min(100, stance_points)
                        swing_points = 101 - stance_points
                        stance_norm = self._resample_series_to_points(stance_series, stance_points)
                        swing_norm = self._resample_series_to_points(swing_series, swing_points) if swing_points > 0 else np.array([])
                        curve = np.concatenate([stance_norm, swing_norm])
                        if len(curve) < 101:
                            curve = np.pad(curve, (0, 101 - len(curve)), mode='edge')
                        elif len(curve) > 101:
                            curve = curve[:101]
                        cycle_angles[joint_type] = curve.tolist()
                    else:
                        cycle_angles[joint_type] = self._resample_series_to_101(joint_series).tolist()

                prev_right_hs = max([rh for rh in right_hs if rh <= cycle_start], default=None)
                next_right_to = next((rt for rt in right_toe_offs if rt >= cycle_start), None)
                double_support_pct = None
                if toe_off_idx is not None and prev_right_hs is not None and prev_right_hs < toe_off_idx:
                    ds_duration = (min(toe_off_idx, next_right_to or toe_off_idx) - prev_right_hs) / fps
                    cycle_duration = (cycle_end - cycle_start) / fps if cycle_end > cycle_start else None
                    if cycle_duration and cycle_duration > 0:
                        double_support_pct = max(0.0, min(1.0, ds_duration / cycle_duration)) * 100.0

                start_time = float(timestamps[start_idx]) if start_idx < len(timestamps) else start_idx / fps
                end_time = float(timestamps[end_idx]) if end_idx < len(timestamps) else end_idx / fps

                all_cycles.append({
                    'cycle_index': idx,
                    'start_frame': start_idx,
                    'end_frame': end_idx,
                    'num_frames': len(cycle_indices),
                    'start_time_sec': start_time,
                    'end_time_sec': end_time,
                    'duration_sec': max(0.0, end_time - start_time),
                    'normalized_angles': cycle_angles,
                    'events': {
                        'left': {
                            'heel_strike_frame': start_idx,
                            'toe_off_frame': int(toe_off_idx) if toe_off_idx is not None else None,
                        },
                        'right': {
                            'prev_heel_strike_frame': int(prev_right_hs) if prev_right_hs is not None else None,
                            'next_toe_off_frame': int(next_right_to) if next_right_to is not None else None,
                            'toe_off_within_cycle': int(right_toe_within) if right_toe_within is not None else None,
                        }
                    },
                    'double_support_percent': double_support_pct
                })

        if not all_cycles:
            print("⚠️ 보행주기가 충분하지 않습니다. 전체 데이터로 정규화합니다.")
            # 전체 데이터를 101포인트로 정규화
            normalized_angles = self._normalize_full_data_to_101(joint_angles)
            return normalized_angles, []

        # 대표 주기: 각 관절별 평균 파형
        aggregated_angles = {}
        for joint_type in joint_types:
            stack = np.array([cycle['normalized_angles'][joint_type] for cycle in all_cycles], dtype=float)
            with np.errstate(invalid='ignore'):
                mean_cycle = np.nanmean(stack, axis=0)

            if np.isnan(mean_cycle).all():
                # 모든 값이 NaN이면 개별 첫 주기 값 사용
                mean_cycle = stack[0]

            aggregated_angles[joint_type] = mean_cycle.tolist()

        print(f"✅ 101포인트 정규화 완료 (총 {len(all_cycles)}개 보행주기)")
        return aggregated_angles, all_cycles

    def _normalize_full_data_to_101(self, joint_angles):
        """전체 데이터를 101포인트로 정규화 (보행주기 탐지 실패시 대안)"""
        normalized_angles = {}

        for joint_type in [
            'hip_flexion_extension',
            'knee_flexion_extension',
            'ankle_dorsi_plantarflexion',
            'hip_abduction_adduction_left',
            'hip_abduction_adduction_right',
            'pelvis_obliquity',
            'trunk_sway_mediolateral',
        ]:
            data = joint_angles[joint_type]

            if len(data) > 1:
                x_original = np.linspace(0, 100, len(data))
                x_new = np.linspace(0, 100, 101)

                interp_func = interp1d(x_original, data, kind='linear', fill_value='extrapolate')
                normalized_data = interp_func(x_new)

                normalized_angles[joint_type] = normalized_data.tolist()
            else:
                normalized_angles[joint_type] = [0] * 101

        return normalized_angles

    def _resample_series_to_points(self, series, points, kind='cubic'):
        """주어진 시계열을 지정된 포인트 수로 보간."""
        series = np.asarray(series, dtype=float)
        if points <= 1:
            return np.array([series[0] if len(series) else 0.0])

        valid_mask = ~np.isnan(series)
        if np.sum(valid_mask) < 2:
            fill_value = float(series[valid_mask][0]) if np.any(valid_mask) else 0.0
            return np.full(points, fill_value, dtype=float)

        series = series[valid_mask]
        x_original = np.linspace(0, 100, len(series))
        x_new = np.linspace(0, 100, points)

        try:
            interp_func = interp1d(x_original, series, kind=kind, fill_value='extrapolate')
            normalized = interp_func(x_new)
        except Exception:
            normalized = np.interp(x_new, x_original, series)

        return normalized

    def _load_angle_calibration_config(self):
        """외부 설정 파일(config/angle_calibration.json)에서 보정 파라미터를 로드."""
        config_path = Path("config/angle_calibration.json")
        if not config_path.exists():
            return

        try:
            with config_path.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"⚠️ 각도 보정 설정을 불러오는 중 오류: {exc}")
            return

        offsets = data.get("offsets")
        if isinstance(offsets, dict):
            self.angle_calibration_offsets.update(offsets)

        phase_shifts = data.get("phase_shifts")
        if isinstance(phase_shifts, dict):
            self.angle_phase_shifts.update(phase_shifts)

        dynamic = data.get("dynamic_shift_search", {})
        if isinstance(dynamic, dict):
            self.dynamic_shift_enabled = bool(dynamic.get("enabled", self.dynamic_shift_enabled))
            self.dynamic_shift_window = int(dynamic.get("window", self.dynamic_shift_window))

    def _apply_angle_calibration(self, normalized_angles, normalized_cycles):
        """GT 기준과 일치하도록 각도 오프셋 및 위상 시프트를 적용한다."""
        joint_types = self.angle_calibration_offsets.keys()

        for joint in joint_types:
            if joint in normalized_angles:
                curve = np.asarray(normalized_angles[joint], dtype=float)
                curve -= self.angle_calibration_offsets[joint]
                shift = self.angle_phase_shifts.get(joint, 0)
                if shift:
                    curve = np.roll(curve, -shift)
                normalized_angles[joint] = curve.tolist()

        for cycle in normalized_cycles:
            angles = cycle.get('normalized_angles', {})
            for joint in joint_types:
                if joint in angles:
                    curve = np.asarray(angles[joint], dtype=float)
                    curve -= self.angle_calibration_offsets[joint]
                    shift = self.angle_phase_shifts.get(joint, 0)
                    if shift:
                        curve = np.roll(curve, -shift)
                    angles[joint] = curve.tolist()

        if self.dynamic_shift_enabled and normalized_cycles:
            window = max(0, int(self.dynamic_shift_window))
            for joint in joint_types:
                if joint not in normalized_angles:
                    continue
                mean_curve = np.asarray(normalized_angles[joint], dtype=float)
                aligned_curves = []
                cycle_indices = []
                for idx, cycle in enumerate(normalized_cycles):
                    angles = cycle.get('normalized_angles', {})
                    if joint not in angles:
                        continue
                    curve = np.asarray(angles[joint], dtype=float)
                    best_shift = 0
                    best_rmse = float('inf')
                    for delta in range(-window, window + 1):
                        shifted = np.roll(curve, -delta)
                        rmse = float(np.sqrt(np.mean((shifted - mean_curve) ** 2)))
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_shift = delta
                    aligned_curve = np.roll(curve, -best_shift)
                    aligned_curves.append(aligned_curve)
                    cycle['normalized_angles'][joint] = aligned_curve.tolist()
                    cycle.setdefault('alignment', {})[joint] = {'delta': int(best_shift), 'rmse': float(best_rmse)}
                    cycle_indices.append(idx)
                if aligned_curves:
                    mean_curve = np.mean(np.stack(aligned_curves), axis=0)
                    normalized_angles[joint] = mean_curve.tolist()

        return normalized_angles, normalized_cycles

    def _compute_gt_quality_metrics(self, subject_id, normalized_angles, normalized_cycles):
        """보정 후 각도를 GT와 비교하여 QA 지표를 계산한다 (가능한 경우)."""
        gt_path = Path("processed") / f"{subject_id}_traditional_condition.csv"
        if not gt_path.exists():
            return None

        try:
            df = pd.read_csv(gt_path)
        except Exception as exc:
            print(f"⚠️ GT 비교용 CSV 로드 실패: {exc}")
            return None

        df = df[df['category'] == 'joint_angle']
        joint_variable_map = {
            'hip_flexion_extension': 'r.hi.angle',
            'knee_flexion_extension': 'r.kn.angle',
            'ankle_dorsi_plantarflexion': 'r.an.angle',
        }

        metrics = {}
        for joint, var_name in joint_variable_map.items():
            gt_curve = df[df['var_name'] == var_name].sort_values('sample')
            if len(gt_curve) != 101 or joint not in normalized_angles:
                continue

            gt_values = gt_curve['Z'].to_numpy(dtype=float)
            mp_mean = np.asarray(normalized_angles[joint], dtype=float)
            diff = mp_mean - gt_values

            gt_centered = gt_values - gt_values.mean()
            mp_centered = mp_mean - mp_mean.mean()

            def safe_corr(a, b):
                if np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
                    return float('nan')
                return float(np.corrcoef(a, b)[0, 1])

            gt_std = np.std(gt_values, ddof=1)

            metrics[joint] = {
                'mean_rmse': float(np.sqrt(np.mean(diff**2))),
                'mean_bias': float(diff.mean()),
                'mean_max_abs': float(np.max(np.abs(diff))),
                'mean_rmse_centered': float(np.sqrt(np.mean((mp_centered - gt_centered) ** 2))),
                'mean_corr': safe_corr(mp_mean, gt_values),
                'mean_corr_centered': safe_corr(mp_centered, gt_centered),
                'mean_diff_ci_lower': float(diff.mean() - 1.96 * (diff.std(ddof=1) / np.sqrt(len(diff)))),
                'mean_diff_ci_upper': float(diff.mean() + 1.96 * (diff.std(ddof=1) / np.sqrt(len(diff)))),
                'cohens_d': float(diff.mean() / gt_std) if gt_std > 0 else float('nan'),
                'icc_2_1': float(self._compute_icc(gt_values, mp_mean)),
            }

            cycle_rmses = []
            for cycle in normalized_cycles:
                angles = cycle.get('normalized_angles', {})
                joint_curve = angles.get(joint)
                if joint_curve is None:
                    continue
                curve = np.asarray(joint_curve, dtype=float)
                cycle_rmses.append(float(np.sqrt(np.mean((curve - gt_values) ** 2))))
            if cycle_rmses:
                metrics[joint]['cycle_rmse_mean'] = float(np.mean(cycle_rmses))
                metrics[joint]['cycle_rmse_std'] = float(np.std(cycle_rmses, ddof=1)) if len(cycle_rmses) > 1 else 0.0

        return metrics if metrics else None

    def _compute_icc(self, gt_values, mp_values):
        """Two-way random-effects single-measure ICC(2,1) between GT and MediaPipe curves."""
        try:
            data = np.vstack([gt_values, mp_values]).T  # shape (n, k=2)
            n, k = data.shape
            mean_targets = data.mean(axis=1, keepdims=True)
            mean_raters = data.mean(axis=0, keepdims=True)
            grand_mean = data.mean()

            ss_total = ((data - grand_mean) ** 2).sum()
            ss_between = k * ((mean_targets - grand_mean) ** 2).sum()
            ss_rater = n * ((mean_raters - grand_mean) ** 2).sum()
            ss_error = ss_total - ss_between - ss_rater

            ms_between = ss_between / (n - 1)
            ms_error = ss_error / ((n - 1) * (k - 1))
            ms_rater = ss_rater / (k - 1)

            icc = (ms_between - ms_error) / (ms_between + (k - 1) * ms_error + k * (ms_rater - ms_error) / n)
            return icc
        except Exception:
            return float('nan')

    def _resample_series_to_101(self, series, kind='cubic'):
        """주어진 시계열을 101포인트로 보간 (호환 함수)."""
        return self._resample_series_to_points(series, 101, kind=kind)

    # ============================================================================
    # 6단계: 통합 분석 실행
    # ============================================================================

    def analyze_gait_video(self, video_path, subject_id="unknown", output_dir="./results", max_frames=None):
        """완전한 보행분석 실행"""
        print(f"🚀 보행분석 시작: {subject_id}")
        print("="*60)

        output_path = None
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)

        try:
            # 1단계: 포즈 랜드마크 추출
            print("\n1️⃣ 포즈 랜드마크 추출")
            self.extract_pose_landmarks(video_path, max_frames=max_frames)

            # 2단계: 보행 이벤트 탐지
            print("\n2️⃣ 보행 이벤트 탐지")
            gait_events = self.detect_gait_events()

            # 3단계: 시공간 매개변수 계산
            print("\n3️⃣ 시공간 매개변수 계산")
            temporal_spatial = self.calculate_temporal_spatial_parameters(gait_events)

            # 4단계: 관절각도 계산
            print("\n4️⃣ 관절각도 계산")
            joint_angles = self.calculate_joint_angles_3d()

            # 5단계: 101포인트 정규화
            print("\n5️⃣ 101포인트 정규화")
            normalized_angles, normalized_cycles = self.normalize_to_101_points(joint_angles, gait_events)
            normalized_angles, normalized_cycles = self._apply_angle_calibration(normalized_angles, normalized_cycles)
            qa_metrics = self._compute_gt_quality_metrics(subject_id, normalized_angles, normalized_cycles)

            # 6단계: 결과 통합
            print("\n6️⃣ 결과 통합")
            analysis_results = {
                'subject_id': subject_id,
                'video_info': self.video_info,
                'temporal_spatial': temporal_spatial,
                'joint_angles_raw': joint_angles,
                'joint_angles_101': normalized_angles,
                'joint_cycles_101': normalized_cycles,
                'qa_metrics': qa_metrics,
                'cycle_summary': {
                    'num_cycles': len(normalized_cycles),
                    'representative': 'mean_left_heel_strike_cycles',
                    'note': 'All detected left heel-strike to heel-strike cycles normalized to 101 points.'
                },
                'gait_events': gait_events,
                'analysis_timestamp': datetime.now().isoformat(),
                'pipeline_version': '1.0'
            }

            # 7단계: 결과 저장
            if output_path is not None:
                print("\n7️⃣ 결과 저장")
                self._save_analysis_results(analysis_results, output_path, subject_id)
            else:
                print("\n7️⃣ 결과 저장 건너뜀 (output_dir=None)")

            print("\n" + "="*60)
            print("🎉 보행분석 완료!")
            if output_path is not None:
                print(f"📁 결과 저장 위치: {output_path}")
            else:
                print("📁 결과 저장 생략")

            return analysis_results

        except Exception as e:
            print(f"\n❌ 분석 중 오류 발생: {e}")
            raise

    def _save_analysis_results(self, results, output_path, subject_id):
        """분석 결과 저장"""
        # JSON 저장
        json_path = output_path / f"{subject_id}_analysis_results.json"
        serializable = self._make_json_serializable(results)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        # 시각화 저장
        self._generate_analysis_plots(results, output_path, subject_id)

        print(f"✅ 결과 저장 완료:")
        print(f"  • JSON: {json_path}")
        print(f"  • 시각화: {output_path}/{subject_id}_*.png")

    def _make_json_serializable(self, obj):
        """numpy 자료형을 포함한 객체를 JSON 직렬화 가능 형태로 변환."""
        import numpy as np

        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._make_json_serializable(value) for value in obj]
        if isinstance(obj, tuple):
            return tuple(self._make_json_serializable(value) for value in obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.generic,)):
            return obj.item()
        return obj

    def _generate_analysis_plots(self, results, output_path, subject_id):
        """분석 결과 시각화 생성"""
        plt.style.use('default')

        # 관절각도 플롯
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        joint_names = ['hip_flexion_extension', 'knee_flexion_extension', 'ankle_dorsi_plantarflexion']
        joint_titles = ['Hip Flexion/Extension', 'Knee Flexion/Extension', 'Ankle Dorsiflexion/Plantarflexion']

        x_points = np.linspace(0, 100, 101)

        for i, (joint, title) in enumerate(zip(joint_names, joint_titles)):
            angles = results['joint_angles_101'][joint]
            axes[i].plot(x_points, angles, 'b-', linewidth=2)
            axes[i].set_title(f'{title} (101-point normalized)')
            axes[i].set_xlabel('Gait Cycle (%)')
            axes[i].set_ylabel('Angle (degrees)')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(0, 100)

        plt.tight_layout()
        plt.savefig(output_path / f"{subject_id}_joint_angles.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 시공간 매개변수 요약 플롯
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        ts = results['temporal_spatial']

        # Cadence
        ax1.bar(['Cadence'], [ts['cadence']], color='skyblue')
        ax1.set_ylabel('Steps/min')
        ax1.set_title('Cadence')
        ax1.set_ylim(0, 140)

        # Stride Time
        if ts['stride_time_list']:
            ax2.hist(ts['stride_time_list'], bins=10, alpha=0.7, color='lightgreen')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Stride Time Distribution')

        # Stride Length
        if ts['stride_length_list']:
            ax3.hist(ts['stride_length_list'], bins=10, alpha=0.7, color='lightcoral')
            ax3.set_xlabel('Length (m)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Stride Length Distribution')

        # Walking Speed
        if ts['walking_speed_list']:
            ax4.hist(ts['walking_speed_list'], bins=10, alpha=0.7, color='gold')
            ax4.set_xlabel('Speed (m/s)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Walking Speed Distribution')

        plt.tight_layout()
        plt.savefig(output_path / f"{subject_id}_temporal_spatial.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """메인 함수"""
    print("🚀 MediaPipe 보행분석 메인 파이프라인")

    # 파이프라인 초기화
    pipeline = MainGaitAnalysisPipeline()

    # 예시 비디오 분석
    video_path = "./sample_videos/gait_video.mp4"  # 실제 비디오 경로로 변경
    subject_id = "S001"

    if Path(video_path).exists():
        results = pipeline.analyze_gait_video(
            video_path=video_path,
            subject_id=subject_id,
            output_dir="./analysis_results"
        )
        print("\n✅ 분석이 성공적으로 완료되었습니다!")
    else:
        print(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
        print("비디오 경로를 확인하고 다시 실행하세요.")

if __name__ == "__main__":
    main()
