#!/usr/bin/env python3
"""
GAVD Multi-View Gait Analysis Integration
Enhanced MediaPipe Gait Analysis System v2.0 - GAVD Integration

다중 카메라 뷰 통합 보행 분석 시스템

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
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict

class GAVDMultiViewGaitAnalyzer:
    """GAVD 다중 뷰 보행 분석 시스템"""

    def __init__(self, gavd_path="/data/datasets/GAVD", gavd_analysis_file=None):
        """
        다중 뷰 보행 분석기 초기화

        Args:
            gavd_path: GAVD 데이터셋 경로
            gavd_analysis_file: GAVD 분석 결과 파일
        """
        self.gavd_path = Path(gavd_path)
        self.videos_path = self.gavd_path / "videos_cut_by_view"
        self.data_path = self.gavd_path / "data"
        self.gavd_analysis_file = gavd_analysis_file

        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # 다중 뷰 데이터
        self.multi_view_videos = {}
        self.view_features = {}
        self.integrated_features = {}

        # 카메라 뷰 설정
        self.camera_views = {
            'front': {'weight': 1.0, 'primary_joints': ['hip', 'knee', 'ankle']},
            'back': {'weight': 0.8, 'primary_joints': ['hip', 'spine']},
            'left_side': {'weight': 1.2, 'primary_joints': ['hip', 'knee', 'ankle', 'stride']},
            'right_side': {'weight': 1.2, 'primary_joints': ['hip', 'knee', 'ankle', 'stride']}
        }

        print(f"🎥 GAVD 다중 뷰 보행 분석기 초기화")
        print(f"📁 비디오 경로: {self.videos_path}")
        print(f"🔍 지원 카메라 뷰: {list(self.camera_views.keys())}")

    def load_multi_view_data(self):
        """다중 뷰 비디오 데이터 로드"""
        print(f"\n📖 다중 뷰 데이터 로드 중...")

        # GAVD 분석 결과에서 다중 뷰 비디오 찾기
        if self.gavd_analysis_file and Path(self.gavd_analysis_file).exists():
            with open(self.gavd_analysis_file, 'r', encoding='utf-8') as f:
                gavd_analysis = json.load(f)

            # 다중 뷰 비디오 추출
            unique_videos = gavd_analysis.get('pathological_patterns', {}).get('unique_videos', [])

            for video_info in unique_videos:
                video_id = video_info['id']
                cam_views = video_info['cam_view']
                gait_pattern = video_info['gait_pat']

                if len(cam_views) > 1:  # 다중 뷰만 선택
                    self.multi_view_videos[video_id] = {
                        'views': cam_views,
                        'gait_pattern': gait_pattern,
                        'dataset_type': video_info.get('dataset', 'Unknown'),
                        'video_files': {}
                    }

                    # 실제 비디오 파일 매칭
                    for view in cam_views:
                        video_pattern = f"{video_id}_{view}_*.mp4"
                        matching_files = list(self.videos_path.glob(video_pattern))

                        if matching_files:
                            self.multi_view_videos[video_id]['video_files'][view] = str(matching_files[0])

        else:
            # GAVD 분석 파일이 없으면 직접 스캔
            print(f"⚠️  GAVD 분석 파일이 없어 직접 비디오 스캔")
            self.scan_multi_view_videos()

        print(f"✅ 다중 뷰 비디오 {len(self.multi_view_videos)}개 발견")

        # 다중 뷰 통계
        view_count_dist = defaultdict(int)
        pattern_dist = defaultdict(int)

        for video_id, video_info in self.multi_view_videos.items():
            view_count_dist[len(video_info['views'])] += 1
            pattern_dist[video_info['gait_pattern']] += 1

        print(f"\n📊 다중 뷰 통계:")
        for view_count, count in sorted(view_count_dist.items()):
            print(f"   {view_count}개 뷰: {count}개 비디오")

        print(f"\n🦴 패턴별 분포:")
        for pattern, count in pattern_dist.items():
            print(f"   {pattern}: {count}개")

        return self.multi_view_videos

    def scan_multi_view_videos(self):
        """비디오 파일을 직접 스캔하여 다중 뷰 찾기"""
        video_files = list(self.videos_path.glob("*.mp4"))
        video_groups = defaultdict(list)

        # 비디오 ID별로 그룹화
        for video_file in video_files:
            filename = video_file.stem
            parts = filename.split('_')

            if len(parts) >= 3:
                video_id = parts[0]
                view = '_'.join(parts[1:-1])
                frame_range = parts[-1]

                video_groups[video_id].append({
                    'view': view,
                    'file': str(video_file),
                    'frame_range': frame_range
                })

        # 다중 뷰만 필터링
        for video_id, files in video_groups.items():
            if len(files) > 1:
                views = [f['view'] for f in files]
                video_files_dict = {f['view']: f['file'] for f in files}

                self.multi_view_videos[video_id] = {
                    'views': views,
                    'gait_pattern': 'unknown',
                    'dataset_type': 'Unknown',
                    'video_files': video_files_dict
                }

    def extract_view_features(self, video_path, view_type, max_frames=200):
        """특정 뷰에서 특징 추출"""
        features = {
            'success': False,
            'view_type': view_type,
            'frame_count': 0,
            'landmarks': [],
            'joint_angles': [],
            'view_specific_features': {},
            'error_message': None
        }

        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                features['error_message'] = f"비디오 파일을 열 수 없습니다: {video_path}"
                return features

            with self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as pose:

                frame_idx = 0

                while frame_idx < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # BGR을 RGB로 변환
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb_frame)

                    if results.pose_landmarks:
                        # 랜드마크 추출
                        landmarks = []
                        for landmark in results.pose_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

                        features['landmarks'].append(landmarks)

                        # 뷰별 특징 계산
                        view_features = self.compute_view_specific_features(
                            results.pose_landmarks.landmark, view_type
                        )
                        features['joint_angles'].append(view_features)

                    frame_idx += 1

                cap.release()

                features['frame_count'] = frame_idx
                features['success'] = len(features['landmarks']) > 0

                # 뷰별 특화 특징 계산
                if features['success']:
                    features['view_specific_features'] = self.analyze_view_specific_patterns(
                        features['landmarks'], features['joint_angles'], view_type
                    )

        except Exception as e:
            features['error_message'] = str(e)

        return features

    def compute_view_specific_features(self, landmarks, view_type):
        """뷰별 특화 특징 계산"""
        # MediaPipe 랜드마크 인덱스
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28

        features = {}

        try:
            if view_type in ['front', 'back']:
                # 정면/후면 뷰: 좌우 대칭성, 균형 분석
                left_hip = landmarks[LEFT_HIP]
                right_hip = landmarks[RIGHT_HIP]
                left_knee = landmarks[LEFT_KNEE]
                right_knee = landmarks[RIGHT_KNEE]
                left_ankle = landmarks[LEFT_ANKLE]
                right_ankle = landmarks[RIGHT_ANKLE]

                # 좌우 대칭성
                hip_symmetry = abs(left_hip.y - right_hip.y)
                knee_symmetry = abs(left_knee.y - right_knee.y)
                ankle_symmetry = abs(left_ankle.y - right_ankle.y)

                features.update({
                    'hip_symmetry': hip_symmetry,
                    'knee_symmetry': knee_symmetry,
                    'ankle_symmetry': ankle_symmetry,
                    'lateral_balance': abs(left_hip.x - right_hip.x)
                })

            elif view_type in ['left_side', 'right_side']:
                # 측면 뷰: 관절 각도, 보행 주기 분석
                hip = landmarks[LEFT_HIP if view_type == 'left_side' else RIGHT_HIP]
                knee = landmarks[LEFT_KNEE if view_type == 'left_side' else RIGHT_KNEE]
                ankle = landmarks[LEFT_ANKLE if view_type == 'left_side' else RIGHT_ANKLE]

                # 무릎 각도 계산
                knee_angle = self.calculate_joint_angle(hip, knee, ankle)

                # 보행 단계 추정 (발목 높이 기반)
                gait_phase = self.estimate_gait_phase(ankle.y)

                features.update({
                    'knee_angle': knee_angle,
                    'ankle_height': ankle.y,
                    'gait_phase': gait_phase,
                    'trunk_lean': abs(hip.x - 0.5)  # 몸통 기울기
                })

        except Exception as e:
            features['calculation_error'] = str(e)

        return features

    def calculate_joint_angle(self, point1, point2, point3):
        """세 점으로 관절 각도 계산"""
        try:
            # 벡터 계산
            v1 = np.array([point1.x - point2.x, point1.y - point2.y])
            v2 = np.array([point3.x - point2.x, point3.y - point2.y])

            # 코사인 값 계산
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)

            # 각도 변환 (라디안 -> 도)
            angle = np.arccos(cos_angle) * 180 / np.pi
            return angle
        except:
            return 0.0

    def estimate_gait_phase(self, ankle_y, threshold=0.1):
        """발목 높이로 보행 단계 추정"""
        # 간단한 임계값 기반 보행 단계 분류
        if ankle_y > 0.8:  # 높은 위치
            return 'swing_high'
        elif ankle_y > 0.7:
            return 'swing_mid'
        elif ankle_y > 0.6:
            return 'stance'
        else:
            return 'contact'

    def analyze_view_specific_patterns(self, landmarks_sequence, joint_angles_sequence, view_type):
        """뷰별 특화 패턴 분석"""
        if not landmarks_sequence or not joint_angles_sequence:
            return {}

        patterns = {}

        try:
            # 시간에 따른 변화 분석
            if view_type in ['left_side', 'right_side']:
                # 측면 뷰: 보행 주기 분석
                ankle_heights = []
                knee_angles = []

                for angles in joint_angles_sequence:
                    if 'ankle_height' in angles:
                        ankle_heights.append(angles['ankle_height'])
                    if 'knee_angle' in angles:
                        knee_angles.append(angles['knee_angle'])

                if ankle_heights:
                    patterns['ankle_height_range'] = max(ankle_heights) - min(ankle_heights)
                    patterns['ankle_height_std'] = np.std(ankle_heights)

                if knee_angles:
                    patterns['knee_angle_range'] = max(knee_angles) - min(knee_angles)
                    patterns['knee_angle_mean'] = np.mean(knee_angles)

            elif view_type in ['front', 'back']:
                # 정면/후면 뷰: 대칭성 변화 분석
                symmetry_scores = []

                for angles in joint_angles_sequence:
                    if 'hip_symmetry' in angles and 'knee_symmetry' in angles:
                        symmetry_score = 1.0 - (angles['hip_symmetry'] + angles['knee_symmetry']) / 2
                        symmetry_scores.append(symmetry_score)

                if symmetry_scores:
                    patterns['symmetry_mean'] = np.mean(symmetry_scores)
                    patterns['symmetry_std'] = np.std(symmetry_scores)

        except Exception as e:
            patterns['analysis_error'] = str(e)

        return patterns

    def integrate_multi_view_features(self, video_id):
        """다중 뷰 특징 통합"""
        if video_id not in self.multi_view_videos:
            return None

        video_info = self.multi_view_videos[video_id]
        views = video_info['views']
        video_files = video_info['video_files']

        print(f"🎬 {video_id} 다중 뷰 특징 통합 중... ({len(views)}개 뷰)")

        # 각 뷰별 특징 추출
        view_results = {}

        for view in views:
            if view in video_files:
                video_path = video_files[view]
                print(f"   📹 {view} 뷰 처리...")

                features = self.extract_view_features(video_path, view)
                view_results[view] = features

                if features['success']:
                    print(f"      ✅ {features['frame_count']}프레임 처리 성공")
                else:
                    print(f"      ❌ 처리 실패: {features.get('error_message', 'Unknown error')}")

        # 뷰 통합 알고리즘
        integrated_features = self.fuse_multi_view_features(view_results, video_info)

        self.integrated_features[video_id] = {
            'video_info': video_info,
            'view_results': view_results,
            'integrated_features': integrated_features,
            'processing_timestamp': datetime.now().isoformat()
        }

        return self.integrated_features[video_id]

    def fuse_multi_view_features(self, view_results, video_info):
        """다중 뷰 특징 융합"""
        fused_features = {
            'gait_symmetry': 0.0,
            'gait_stability': 0.0,
            'joint_coordination': 0.0,
            'movement_fluidity': 0.0,
            'pathological_indicators': {},
            'confidence_score': 0.0,
            'view_contributions': {}
        }

        successful_views = {view: result for view, result in view_results.items() if result['success']}

        if not successful_views:
            return fused_features

        # 가중 평균 기반 특징 융합
        total_weight = 0
        weighted_features = defaultdict(float)

        for view, result in successful_views.items():
            view_weight = self.camera_views.get(view, {}).get('weight', 1.0)
            total_weight += view_weight

            # 뷰별 특화 특징 추출
            view_specific = result.get('view_specific_features', {})

            if view in ['front', 'back']:
                # 정면/후면: 대칭성 정보
                if 'symmetry_mean' in view_specific:
                    weighted_features['symmetry'] += view_specific['symmetry_mean'] * view_weight

            elif view in ['left_side', 'right_side']:
                # 측면: 관절 각도 및 보행 주기 정보
                if 'knee_angle_mean' in view_specific:
                    weighted_features['knee_angle'] += view_specific['knee_angle_mean'] * view_weight
                if 'ankle_height_range' in view_specific:
                    weighted_features['ankle_range'] += view_specific['ankle_height_range'] * view_weight

            # 뷰 기여도 기록
            fused_features['view_contributions'][view] = {
                'weight': view_weight,
                'frame_count': result['frame_count'],
                'features_extracted': len(view_specific)
            }

        # 가중 평균 계산
        if total_weight > 0:
            for feature, value in weighted_features.items():
                weighted_features[feature] = value / total_weight

        # 최종 특징 계산
        fused_features['gait_symmetry'] = weighted_features.get('symmetry', 0.5)
        fused_features['joint_coordination'] = np.mean([
            weighted_features.get('knee_angle', 45) / 90,  # 정규화
            weighted_features.get('ankle_range', 0.1) * 10   # 정규화
        ])

        # 안정성 및 유동성 계산
        fused_features['gait_stability'] = 1.0 - np.std([
            result.get('view_specific_features', {}).get('ankle_height_std', 0.1)
            for result in successful_views.values()
        ])

        fused_features['movement_fluidity'] = min(1.0, np.mean([
            len(result['landmarks']) / max(result['frame_count'], 1)
            for result in successful_views.values()
        ]))

        # 병적 지표 계산
        pathological_score = 0.0

        if fused_features['gait_symmetry'] < 0.7:
            pathological_score += 0.3
        if fused_features['joint_coordination'] < 0.6:
            pathological_score += 0.25
        if fused_features['gait_stability'] < 0.5:
            pathological_score += 0.25
        if fused_features['movement_fluidity'] < 0.7:
            pathological_score += 0.2

        fused_features['pathological_indicators'] = {
            'overall_score': pathological_score,
            'risk_level': 'high' if pathological_score > 0.6 else 'medium' if pathological_score > 0.3 else 'low'
        }

        # 신뢰도 점수 (사용된 뷰 수와 성공률 기반)
        fused_features['confidence_score'] = min(1.0, len(successful_views) / 4.0 * 0.8 +
                                                 fused_features['movement_fluidity'] * 0.2)

        return fused_features

    def process_multi_view_batch(self, limit_videos=10):
        """다중 뷰 비디오 배치 처리"""
        print(f"\n🚀 다중 뷰 배치 처리 시작")
        print("=" * 60)

        if not self.multi_view_videos:
            self.load_multi_view_data()

        # 처리할 비디오 선택
        video_ids = list(self.multi_view_videos.keys())[:limit_videos]

        print(f"🎬 {len(video_ids)}개 다중 뷰 비디오 처리 예정")

        results = []

        for i, video_id in enumerate(video_ids, 1):
            print(f"\n📹 [{i}/{len(video_ids)}] {video_id} 처리 중...")

            try:
                result = self.integrate_multi_view_features(video_id)
                if result:
                    results.append(result)
                    print(f"   ✅ 통합 완료")
                else:
                    print(f"   ❌ 통합 실패")

            except Exception as e:
                print(f"   ❌ 오류: {e}")

        print(f"\n🎉 다중 뷰 배치 처리 완료!")
        print(f"   성공: {len(results)}/{len(video_ids)}개")

        return results

    def analyze_multi_view_performance(self):
        """다중 뷰 성능 분석"""
        if not self.integrated_features:
            print("❌ 통합된 특징이 없습니다.")
            return {}

        print(f"\n📊 다중 뷰 성능 분석...")

        analysis = {
            'total_videos': len(self.integrated_features),
            'view_coverage': defaultdict(int),
            'performance_metrics': {},
            'pathological_detection': {'high': 0, 'medium': 0, 'low': 0}
        }

        # 뷰 커버리지 분석
        for video_id, result in self.integrated_features.items():
            view_results = result['view_results']
            successful_views = [view for view, res in view_results.items() if res['success']]

            for view in successful_views:
                analysis['view_coverage'][view] += 1

            # 병적 위험도 분석
            integrated = result['integrated_features']
            risk_level = integrated.get('pathological_indicators', {}).get('risk_level', 'low')
            analysis['pathological_detection'][risk_level] += 1

        # 성능 메트릭 계산
        confidence_scores = []
        symmetry_scores = []
        stability_scores = []

        for result in self.integrated_features.values():
            integrated = result['integrated_features']
            confidence_scores.append(integrated.get('confidence_score', 0))
            symmetry_scores.append(integrated.get('gait_symmetry', 0))
            stability_scores.append(integrated.get('gait_stability', 0))

        analysis['performance_metrics'] = {
            'average_confidence': np.mean(confidence_scores),
            'average_symmetry': np.mean(symmetry_scores),
            'average_stability': np.mean(stability_scores),
            'high_confidence_ratio': sum(1 for s in confidence_scores if s > 0.8) / len(confidence_scores)
        }

        print(f"📈 분석 결과:")
        print(f"   총 비디오: {analysis['total_videos']}")
        print(f"   평균 신뢰도: {analysis['performance_metrics']['average_confidence']:.3f}")
        print(f"   평균 대칭성: {analysis['performance_metrics']['average_symmetry']:.3f}")
        print(f"   평균 안정성: {analysis['performance_metrics']['average_stability']:.3f}")

        print(f"\n🎥 뷰별 커버리지:")
        for view, count in analysis['view_coverage'].items():
            percentage = count / analysis['total_videos'] * 100
            print(f"   {view}: {count}개 ({percentage:.1f}%)")

        print(f"\n🦴 위험도 분포:")
        for risk, count in analysis['pathological_detection'].items():
            percentage = count / analysis['total_videos'] * 100
            print(f"   {risk}: {count}개 ({percentage:.1f}%)")

        return analysis

    def save_multi_view_results(self, output_file=None):
        """다중 뷰 결과 저장"""
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"gavd_multiview_results_{timestamp}.json"

        # 성능 분석
        performance_analysis = self.analyze_multi_view_performance()

        # 전체 결과 구조
        full_results = {
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'total_multi_view_videos': len(self.multi_view_videos),
                'processed_videos': len(self.integrated_features),
                'camera_views_supported': list(self.camera_views.keys())
            },
            'performance_analysis': performance_analysis,
            'integrated_features': self.integrated_features,
            'multi_view_videos_info': self.multi_view_videos
        }

        # JSON 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)

        file_size = Path(output_file).stat().st_size / (1024*1024)  # MB
        print(f"\n💾 다중 뷰 결과 저장: {output_file}")
        print(f"   파일 크기: {file_size:.1f} MB")

        return output_file

def main():
    """메인 실행 함수"""
    print("🎥 GAVD 다중 뷰 보행 분석 통합 시스템")
    print("=" * 60)

    # 기존 GAVD 분석 파일 찾기
    gavd_analysis_files = list(Path(".").glob("gavd_dataset_analysis_*.json"))
    gavd_file = gavd_analysis_files[0] if gavd_analysis_files else None

    # 다중 뷰 분석기 초기화
    analyzer = GAVDMultiViewGaitAnalyzer(gavd_analysis_file=gavd_file)

    try:
        # 1. 다중 뷰 데이터 로드
        analyzer.load_multi_view_data()

        # 2. 다중 뷰 배치 처리 (테스트: 5개 비디오)
        print(f"\n🧪 테스트 모드: 5개 다중 뷰 비디오 처리")
        results = analyzer.process_multi_view_batch(limit_videos=5)

        # 3. 성능 분석
        analysis = analyzer.analyze_multi_view_performance()

        # 4. 결과 저장
        output_file = analyzer.save_multi_view_results()

        print(f"\n🎉 GAVD 다중 뷰 분석 완료!")
        print(f"📁 결과 파일: {output_file}")
        print(f"🎬 {len(results)}개 비디오 성공적 처리")

        if analysis['performance_metrics']['average_confidence'] > 0.7:
            print(f"\n✨ 평균 신뢰도 {analysis['performance_metrics']['average_confidence']:.1%}로 높습니다!")
            print(f"💡 전체 다중 뷰 비디오 처리를 권장합니다.")

    except Exception as e:
        print(f"❌ 다중 뷰 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()