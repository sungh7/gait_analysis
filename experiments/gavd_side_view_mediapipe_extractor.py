#!/usr/bin/env python3
"""
GAVD Side View MediaPipe Gait Analysis Extractor
Enhanced MediaPipe Gait Analysis System v3.0 - Side View 전용

실제 GAVD 임상 동영상(side view)에서 MediaPipe를 사용한 보행 분석

Author: Research Team
Date: 2025-09-22
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class GAVDSideViewMediaPipeExtractor:
    """GAVD Side View 전용 MediaPipe 보행 분석 추출기"""

    def __init__(self, gavd_analyzer=None):
        """
        GAVD Side View MediaPipe 추출기 초기화

        Args:
            gavd_analyzer: GAVDDatasetAnalyzer 인스턴스
        """
        self.gavd_analyzer = gavd_analyzer

        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Side view 전용 랜드마크 인덱스
        self.side_view_landmarks = {
            # 하체 주요 관절 (side view에서 중요)
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_heel': 29,
            'right_heel': 30,
            'left_foot_index': 31,
            'right_foot_index': 32,

            # 상체 (자세 참조용)
            'left_shoulder': 11,
            'right_shoulder': 12,
            'nose': 0
        }

        # 분석 결과 저장
        self.side_view_pairs = []
        self.extraction_results = []
        self.processing_stats = {
            'total_videos': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'side_view_only': True
        }

        print(f"📐 GAVD Side View MediaPipe 추출기 초기화")
        print(f"🎥 지원 뷰: left_side, right_side")

    def load_side_view_pairs(self):
        """Side view 비디오-주석 쌍 로드"""
        print(f"\n📐 Side view 비디오-주석 쌍 로드 중...")

        if self.gavd_analyzer is None:
            from gavd_dataset_analyzer import GAVDDatasetAnalyzer
            self.gavd_analyzer = GAVDDatasetAnalyzer()
            self.gavd_analyzer.load_clinical_annotations()

        # Side view 전용 매칭
        self.side_view_pairs = self.gavd_analyzer.match_videos_with_annotations(side_view_only=True)

        print(f"✅ Side view 쌍 로드 완료: {len(self.side_view_pairs)}개")

        # 뷰별 분포 확인
        view_counts = {}
        for pair in self.side_view_pairs:
            view = pair['camera_view']
            view_counts[view] = view_counts.get(view, 0) + 1

        print(f"📷 Side view 분포:")
        for view, count in view_counts.items():
            print(f"   {view}: {count}개")

        return self.side_view_pairs

    def extract_pose_landmarks(self, video_path):
        """단일 비디오에서 pose landmarks 추출"""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return None, "비디오 열기 실패"

        landmarks_sequence = []
        frame_count = 0
        successful_frames = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # BGR을 RGB로 변환
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # MediaPipe로 pose 검출
                results = self.pose.process(rgb_frame)

                if results.pose_landmarks:
                    successful_frames += 1

                    # 랜드마크 좌표 추출
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        })

                    landmarks_sequence.append({
                        'frame_number': frame_count,
                        'landmarks': landmarks,
                        'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS)
                    })

        except Exception as e:
            return None, f"처리 중 오류: {e}"

        finally:
            cap.release()

        if len(landmarks_sequence) == 0:
            return None, "유효한 pose 검출 없음"

        success_rate = successful_frames / frame_count if frame_count > 0 else 0

        return {
            'landmarks_sequence': landmarks_sequence,
            'total_frames': frame_count,
            'successful_frames': successful_frames,
            'success_rate': success_rate,
            'fps': cap.get(cv2.CAP_PROP_FPS)
        }, "성공"

    def extract_gait_cycle_features(self, landmarks_data, camera_view):
        """Side view 랜드마크에서 보행 주기 특징 추출"""
        landmarks_sequence = landmarks_data['landmarks_sequence']

        # 발목과 발가락 궤적 추출 (side view 최적화)
        ankle_trajectory = []
        heel_trajectory = []
        knee_trajectory = []
        hip_trajectory = []

        for frame_data in landmarks_sequence:
            landmarks = frame_data['landmarks']

            # Side view에서는 카메라 측면에 따라 적절한 다리 선택
            if camera_view == 'left_side':
                # 왼쪽에서 촬영 -> 오른쪽 다리가 더 잘 보임
                ankle_idx = self.side_view_landmarks['right_ankle']
                heel_idx = self.side_view_landmarks['right_heel']
                knee_idx = self.side_view_landmarks['right_knee']
                hip_idx = self.side_view_landmarks['right_hip']
            else:  # right_side
                # 오른쪽에서 촬영 -> 왼쪽 다리가 더 잘 보임
                ankle_idx = self.side_view_landmarks['left_ankle']
                heel_idx = self.side_view_landmarks['left_heel']
                knee_idx = self.side_view_landmarks['left_knee']
                hip_idx = self.side_view_landmarks['left_hip']

            if (ankle_idx < len(landmarks) and heel_idx < len(landmarks) and
                knee_idx < len(landmarks) and hip_idx < len(landmarks)):

                ankle = landmarks[ankle_idx]
                heel = landmarks[heel_idx]
                knee = landmarks[knee_idx]
                hip = landmarks[hip_idx]

                # Y 좌표 (수직 위치) - side view에서 중요
                ankle_trajectory.append({
                    'x': ankle['x'],
                    'y': ankle['y'],
                    'visibility': ankle['visibility'],
                    'timestamp': frame_data['timestamp']
                })

                heel_trajectory.append({
                    'x': heel['x'],
                    'y': heel['y'],
                    'visibility': heel['visibility'],
                    'timestamp': frame_data['timestamp']
                })

                knee_trajectory.append({
                    'x': knee['x'],
                    'y': knee['y'],
                    'visibility': knee['visibility'],
                    'timestamp': frame_data['timestamp']
                })

                hip_trajectory.append({
                    'x': hip['x'],
                    'y': hip['y'],
                    'visibility': hip['visibility'],
                    'timestamp': frame_data['timestamp']
                })

        return {
            'ankle_trajectory': ankle_trajectory,
            'heel_trajectory': heel_trajectory,
            'knee_trajectory': knee_trajectory,
            'hip_trajectory': hip_trajectory,
            'camera_view': camera_view,
            'primary_limb': 'right' if camera_view == 'left_side' else 'left'
        }

    def detect_gait_events(self, trajectories):
        """Side view에서 보행 이벤트 (heel strike, toe off) 검출"""
        heel_trajectory = trajectories['heel_trajectory']
        ankle_trajectory = trajectories['ankle_trajectory']

        if len(heel_trajectory) < 10:
            return {'heel_strikes': [], 'toe_offs': [], 'gait_cycles': []}

        # Y 좌표 (수직) 변화로 heel strike 검출
        heel_y = [point['y'] for point in heel_trajectory]
        ankle_y = [point['y'] for point in ankle_trajectory]

        # 발가락-발목 높이 차이로 toe off 검출
        heel_ankle_diff = [abs(h - a) for h, a in zip(heel_y, ankle_y)]

        # Simple peak detection for heel strikes (local minima in Y)
        heel_strikes = []
        toe_offs = []

        # 최저점 (heel strike) 찾기
        for i in range(1, len(heel_y) - 1):
            if heel_y[i] < heel_y[i-1] and heel_y[i] < heel_y[i+1]:
                if len(heel_strikes) == 0 or i - heel_strikes[-1] > 10:  # 최소 간격
                    heel_strikes.append(i)

        # 최고점 (toe off) 찾기
        for i in range(1, len(heel_ankle_diff) - 1):
            if heel_ankle_diff[i] > heel_ankle_diff[i-1] and heel_ankle_diff[i] > heel_ankle_diff[i+1]:
                if len(toe_offs) == 0 or i - toe_offs[-1] > 10:  # 최소 간격
                    toe_offs.append(i)

        # 보행 주기 구성 (heel strike to heel strike)
        gait_cycles = []
        for i in range(len(heel_strikes) - 1):
            start_frame = heel_strikes[i]
            end_frame = heel_strikes[i + 1]

            # 이 주기 내의 toe off 찾기
            cycle_toe_offs = [to for to in toe_offs if start_frame < to < end_frame]

            gait_cycles.append({
                'start_frame': start_frame,
                'end_frame': end_frame,
                'duration_frames': end_frame - start_frame,
                'heel_strike_frame': start_frame,
                'toe_off_frames': cycle_toe_offs,
                'start_timestamp': heel_trajectory[start_frame]['timestamp'],
                'end_timestamp': heel_trajectory[end_frame]['timestamp'],
                'duration_seconds': heel_trajectory[end_frame]['timestamp'] - heel_trajectory[start_frame]['timestamp']
            })

        return {
            'heel_strikes': heel_strikes,
            'toe_offs': toe_offs,
            'gait_cycles': gait_cycles,
            'total_cycles': len(gait_cycles)
        }

    def calculate_gait_parameters(self, trajectories, gait_events, video_info):
        """Side view에서 임상적 보행 파라미터 계산"""
        gait_cycles = gait_events['gait_cycles']

        if len(gait_cycles) < 2:
            return None

        # 기본 통계
        cycle_durations = [cycle['duration_seconds'] for cycle in gait_cycles]
        avg_cycle_duration = np.mean(cycle_durations)
        cycle_variability = np.std(cycle_durations) / avg_cycle_duration if avg_cycle_duration > 0 else 0

        # 케이던스 계산 (steps/minute)
        cadence = 60 / avg_cycle_duration if avg_cycle_duration > 0 else 0

        # 발목 높이 변화 분석
        ankle_trajectory = trajectories['ankle_trajectory']
        ankle_y_values = [point['y'] for point in ankle_trajectory]
        foot_clearance = max(ankle_y_values) - min(ankle_y_values) if ankle_y_values else 0

        # 무릎 각도 변화 (간접 추정)
        knee_trajectory = trajectories['knee_trajectory']
        hip_trajectory = trajectories['hip_trajectory']

        knee_hip_distances = []
        for i in range(min(len(knee_trajectory), len(hip_trajectory))):
            knee = knee_trajectory[i]
            hip = hip_trajectory[i]
            distance = np.sqrt((knee['x'] - hip['x'])**2 + (knee['y'] - hip['y'])**2)
            knee_hip_distances.append(distance)

        knee_flexion_range = max(knee_hip_distances) - min(knee_hip_distances) if knee_hip_distances else 0

        # 자세 안정성 (궤적의 부드러움)
        ankle_x_values = [point['x'] for point in ankle_trajectory]
        step_width_variability = np.std(ankle_x_values) if len(ankle_x_values) > 1 else 0

        # 평균 가시성 (landmark 검출 품질)
        all_visibilities = []
        for traj in [ankle_trajectory, knee_trajectory, hip_trajectory]:
            for point in traj:
                all_visibilities.append(point['visibility'])
        avg_visibility = np.mean(all_visibilities) if all_visibilities else 0

        return {
            'cadence': cadence,
            'avg_cycle_duration': avg_cycle_duration,
            'cycle_variability': cycle_variability,
            'foot_clearance': foot_clearance,
            'knee_flexion_range': knee_flexion_range,
            'step_width_variability': step_width_variability,
            'avg_visibility': avg_visibility,
            'total_gait_cycles': len(gait_cycles),
            'gait_symmetry': 1.0 - cycle_variability,  # 높을수록 대칭적
            'postural_stability': 1.0 - step_width_variability,  # 높을수록 안정적
            'movement_efficiency': avg_visibility * (1.0 - cycle_variability),
            'primary_limb': trajectories['primary_limb'],
            'camera_view': trajectories['camera_view']
        }

    def process_single_video(self, video_pair):
        """단일 side view 비디오 처리"""
        video_path = video_pair['video_file']
        video_id = video_pair['video_id']
        camera_view = video_pair['camera_view']
        gait_pattern = video_pair['gait_pattern']

        print(f"   📐 처리 중: {video_id} ({camera_view}) - {gait_pattern}")

        try:
            # 1. Pose landmarks 추출
            landmarks_data, status = self.extract_pose_landmarks(video_path)

            if landmarks_data is None:
                return {
                    'video_info': video_pair,
                    'success': False,
                    'error': status,
                    'processing_time': None
                }

            # 2. 보행 궤적 추출
            trajectories = self.extract_gait_cycle_features(landmarks_data, camera_view)

            # 3. 보행 이벤트 검출
            gait_events = self.detect_gait_events(trajectories)

            # 4. 임상 파라미터 계산
            gait_parameters = self.calculate_gait_parameters(trajectories, gait_events, video_pair)

            if gait_parameters is None:
                return {
                    'video_info': video_pair,
                    'success': False,
                    'error': "충분한 보행 주기 검출 실패",
                    'processing_time': None
                }

            # 성공적인 처리 결과
            return {
                'video_info': video_pair,
                'success': True,
                'landmarks_data': {
                    'total_frames': landmarks_data['total_frames'],
                    'successful_frames': landmarks_data['successful_frames'],
                    'success_rate': landmarks_data['success_rate'],
                    'fps': landmarks_data['fps']
                },
                'trajectories': {
                    'ankle_points': len(trajectories['ankle_trajectory']),
                    'heel_points': len(trajectories['heel_trajectory']),
                    'knee_points': len(trajectories['knee_trajectory']),
                    'hip_points': len(trajectories['hip_trajectory']),
                    'primary_limb': trajectories['primary_limb']
                },
                'gait_events': gait_events,
                'gait_parameters': gait_parameters,
                'error': None
            }

        except Exception as e:
            return {
                'video_info': video_pair,
                'success': False,
                'error': f"처리 중 예외 발생: {e}",
                'processing_time': None
            }

    def process_all_side_view_videos(self, max_videos=None):
        """모든 side view 비디오 처리"""
        print(f"\n🎥 Side view 비디오 일괄 처리 시작...")

        if not self.side_view_pairs:
            self.load_side_view_pairs()

        videos_to_process = self.side_view_pairs[:max_videos] if max_videos else self.side_view_pairs
        self.processing_stats['total_videos'] = len(videos_to_process)

        print(f"   처리할 비디오 수: {len(videos_to_process)}개")

        for i, video_pair in enumerate(videos_to_process, 1):
            print(f"\n📐 [{i}/{len(videos_to_process)}]", end=" ")

            result = self.process_single_video(video_pair)
            self.extraction_results.append(result)

            if result['success']:
                self.processing_stats['successful_extractions'] += 1
                print(f"✅ 성공")
            else:
                self.processing_stats['failed_extractions'] += 1
                print(f"❌ 실패: {result['error']}")

        # 최종 통계
        success_rate = (self.processing_stats['successful_extractions'] /
                       self.processing_stats['total_videos']) * 100

        print(f"\n📊 Side view 처리 완료:")
        print(f"   총 비디오: {self.processing_stats['total_videos']}개")
        print(f"   성공: {self.processing_stats['successful_extractions']}개")
        print(f"   실패: {self.processing_stats['failed_extractions']}개")
        print(f"   성공률: {success_rate:.1f}%")

        return self.extraction_results

    def save_results(self, output_file=None):
        """처리 결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if output_file is None:
            output_file = f"gavd_side_view_mediapipe_results_{timestamp}.json"

        results_data = {
            'extraction_info': {
                'timestamp': timestamp,
                'side_view_only': True,
                'processing_stats': self.processing_stats,
                'mediapipe_config': {
                    'model_complexity': 1,
                    'min_detection_confidence': 0.5,
                    'min_tracking_confidence': 0.5
                }
            },
            'results': self.extraction_results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 결과 저장 완료: {output_file}")
        return output_file

    def generate_analysis_report(self):
        """Side view 분석 보고서 생성"""
        if not self.extraction_results:
            print("❌ 분석할 결과가 없습니다.")
            return

        successful_results = [r for r in self.extraction_results if r['success']]

        print(f"\n📋 GAVD Side View MediaPipe 분석 보고서")
        print(f"{'='*60}")

        # 기본 통계
        print(f"\n📊 처리 통계:")
        print(f"   총 비디오: {len(self.extraction_results)}개")
        print(f"   성공적 처리: {len(successful_results)}개")
        print(f"   실패: {len(self.extraction_results) - len(successful_results)}개")

        if not successful_results:
            print("❌ 성공적인 처리 결과가 없습니다.")
            return

        # 카메라 뷰 분석
        view_stats = {}
        for result in successful_results:
            view = result['video_info']['camera_view']
            view_stats[view] = view_stats.get(view, 0) + 1

        print(f"\n📷 카메라 뷰별 성공:")
        for view, count in view_stats.items():
            print(f"   {view}: {count}개")

        # 보행 패턴 분석
        pattern_stats = {}
        for result in successful_results:
            pattern = result['video_info']['gait_pattern']
            pattern_stats[pattern] = pattern_stats.get(pattern, 0) + 1

        print(f"\n🦴 보행 패턴별 성공:")
        for pattern, count in pattern_stats.items():
            print(f"   {pattern}: {count}개")

        # 임상 파라미터 통계
        cadences = []
        cycle_variabilities = []
        foot_clearances = []
        gait_symmetries = []

        for result in successful_results:
            if 'gait_parameters' in result and result['gait_parameters']:
                params = result['gait_parameters']
                cadences.append(params['cadence'])
                cycle_variabilities.append(params['cycle_variability'])
                foot_clearances.append(params['foot_clearance'])
                gait_symmetries.append(params['gait_symmetry'])

        if cadences:
            print(f"\n🚶 보행 파라미터 통계:")
            print(f"   케이던스: {np.mean(cadences):.1f} ± {np.std(cadences):.1f} steps/min")
            print(f"   주기 변동성: {np.mean(cycle_variabilities):.3f} ± {np.std(cycle_variabilities):.3f}")
            print(f"   발 높이 변화: {np.mean(foot_clearances):.3f} ± {np.std(foot_clearances):.3f}")
            print(f"   보행 대칭성: {np.mean(gait_symmetries):.3f} ± {np.std(gait_symmetries):.3f}")

def main():
    """메인 실행 함수"""
    print("📐 GAVD Side View MediaPipe 보행 분석기")
    print("=" * 50)

    # GAVDDatasetAnalyzer 초기화
    from gavd_dataset_analyzer import GAVDDatasetAnalyzer
    gavd_analyzer = GAVDDatasetAnalyzer()
    gavd_analyzer.load_clinical_annotations()

    # Side View MediaPipe 추출기 초기화
    extractor = GAVDSideViewMediaPipeExtractor(gavd_analyzer)

    try:
        # 1. Side view 쌍 로드
        extractor.load_side_view_pairs()

        # 2. 테스트 처리 (처음 10개)
        print(f"\n🧪 테스트 처리 (최대 10개 비디오)")
        extractor.process_all_side_view_videos(max_videos=10)

        # 3. 결과 저장
        output_file = extractor.save_results()

        # 4. 분석 보고서 생성
        extractor.generate_analysis_report()

        print(f"\n🎉 Side view MediaPipe 분석 완료!")
        print(f"💾 결과 파일: {output_file}")

    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()