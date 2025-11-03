#!/usr/bin/env python3
"""
GAVD MediaPipe Feature Extractor
Enhanced MediaPipe Gait Analysis System v2.0 - GAVD Integration

GAVD 데이터셋의 510개 비디오 클립에서 MediaPipe 특징 추출

Author: Research Team
Date: 2025-09-22
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
from pathlib import Path
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict
import ast

class GAVDMediaPipeExtractor:
    """GAVD 비디오에서 MediaPipe 특징 추출기"""

    def __init__(self, gavd_path="/data/datasets/GAVD", max_workers=4):
        """
        GAVD MediaPipe 추출기 초기화

        Args:
            gavd_path: GAVD 데이터셋 경로
            max_workers: 병렬 처리 워커 수
        """
        self.gavd_path = Path(gavd_path)
        self.videos_path = self.gavd_path / "videos_cut_by_view"
        self.data_path = self.gavd_path / "data"
        self.max_workers = max_workers

        # MediaPipe 초기화
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # 결과 저장
        self.extracted_features = []
        self.processing_stats = {
            'total_videos': 0,
            'successful': 0,
            'failed': 0,
            'total_frames': 0,
            'processing_time': 0
        }

        # 스레드 안전 락
        self.lock = threading.Lock()

        print(f"🎬 GAVD MediaPipe Feature Extractor 초기화")
        print(f"📁 비디오 경로: {self.videos_path}")
        print(f"⚡ 워커 수: {max_workers}")

    def load_clinical_annotations(self):
        """임상 주석 데이터 로드"""
        print(f"\n📖 임상 주석 데이터 로드...")

        annotation_files = list(self.data_path.glob("GAVD_Clinical_Annotations_*.csv"))
        all_annotations = []

        for file_path in annotation_files:
            df = pd.read_csv(file_path)
            all_annotations.append(df)

        combined_annotations = pd.concat(all_annotations, ignore_index=True)
        print(f"✅ {len(combined_annotations):,}개 주석 데이터 로드 완료")

        return combined_annotations

    def get_video_annotation_pairs(self):
        """비디오-주석 쌍 생성"""
        print(f"\n🔗 비디오-주석 쌍 생성...")

        annotations = self.load_clinical_annotations()
        video_files = list(self.videos_path.glob("*.mp4"))

        pairs = []

        for video_file in video_files:
            filename = video_file.stem
            parts = filename.split('_')

            if len(parts) >= 3:
                video_id = parts[0]
                view = '_'.join(parts[1:-1])
                frame_range = parts[-1]

                # 해당 비디오의 주석 찾기
                video_annotations = annotations[
                    (annotations['id'] == video_id) &
                    (annotations['cam_view'] == view)
                ]

                if not video_annotations.empty:
                    gait_pattern = video_annotations['gait_pat'].iloc[0]
                    dataset_type = video_annotations['dataset'].iloc[0]

                    pairs.append({
                        'video_file': str(video_file),
                        'video_id': video_id,
                        'camera_view': view,
                        'frame_range': frame_range,
                        'gait_pattern': gait_pattern,
                        'dataset_type': dataset_type,
                        'annotation_count': len(video_annotations)
                    })

        print(f"✅ {len(pairs)}개 비디오-주석 쌍 생성 완료")
        return pairs

    def extract_mediapipe_features(self, video_path, max_frames=None):
        """단일 비디오에서 MediaPipe 특징 추출"""
        features = {
            'success': False,
            'frame_count': 0,
            'processing_time': 0,
            'pose_landmarks': [],
            'world_landmarks': [],
            'visibility_scores': [],
            'error_message': None
        }

        start_time = time.time()

        try:
            # 비디오 캡처 초기화
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                features['error_message'] = "비디오 파일을 열 수 없습니다"
                return features

            # MediaPipe Pose 초기화
            with self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            ) as pose:

                frame_idx = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if max_frames and frame_idx >= max_frames:
                        break

                    # BGR을 RGB로 변환
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # MediaPipe 처리
                    results = pose.process(rgb_frame)

                    if results.pose_landmarks:
                        # 2D 랜드마크 추출
                        landmarks_2d = []
                        visibility_scores = []

                        for landmark in results.pose_landmarks.landmark:
                            landmarks_2d.extend([landmark.x, landmark.y])
                            visibility_scores.append(landmark.visibility)

                        features['pose_landmarks'].append(landmarks_2d)
                        features['visibility_scores'].append(visibility_scores)

                        # 3D 월드 랜드마크 추출
                        if results.pose_world_landmarks:
                            landmarks_3d = []
                            for landmark in results.pose_world_landmarks.landmark:
                                landmarks_3d.extend([landmark.x, landmark.y, landmark.z])
                            features['world_landmarks'].append(landmarks_3d)

                    frame_idx += 1

                cap.release()

                features['frame_count'] = frame_idx
                features['processing_time'] = time.time() - start_time
                features['success'] = len(features['pose_landmarks']) > 0

        except Exception as e:
            features['error_message'] = str(e)
            features['processing_time'] = time.time() - start_time

        return features

    def compute_gait_features(self, pose_landmarks, visibility_scores):
        """보행 특징 계산"""
        if not pose_landmarks or len(pose_landmarks) < 10:
            return {}

        # NumPy 배열로 변환
        landmarks_array = np.array(pose_landmarks)
        visibility_array = np.array(visibility_scores)

        # 키포인트 인덱스 (MediaPipe Pose)
        left_hip = 23 * 2  # x,y 좌표이므로 *2
        right_hip = 24 * 2
        left_knee = 25 * 2
        right_knee = 26 * 2
        left_ankle = 27 * 2
        right_ankle = 28 * 2
        left_heel = 29 * 2
        right_heel = 30 * 2
        left_foot_index = 31 * 2
        right_foot_index = 32 * 2

        gait_features = {}

        try:
            # 평균 visibility 점수
            gait_features['avg_visibility'] = np.mean(visibility_array, axis=0).tolist()

            # 관절 각도 계산 (프레임별)
            hip_angles = []
            knee_angles = []
            ankle_angles = []

            for frame_landmarks in landmarks_array:
                if len(frame_landmarks) >= 66:  # 33개 랜드마크 * 2 (x,y)
                    # 왼쪽 무릎 각도 (대퇴-정강이)
                    left_knee_angle = self.calculate_angle(
                        [frame_landmarks[left_hip], frame_landmarks[left_hip+1]],
                        [frame_landmarks[left_knee], frame_landmarks[left_knee+1]],
                        [frame_landmarks[left_ankle], frame_landmarks[left_ankle+1]]
                    )

                    # 오른쪽 무릎 각도
                    right_knee_angle = self.calculate_angle(
                        [frame_landmarks[right_hip], frame_landmarks[right_hip+1]],
                        [frame_landmarks[right_knee], frame_landmarks[right_knee+1]],
                        [frame_landmarks[right_ankle], frame_landmarks[right_ankle+1]]
                    )

                    knee_angles.append([left_knee_angle, right_knee_angle])

            gait_features['knee_angles'] = knee_angles

            # 발목 높이 변화 (수직 이동)
            left_ankle_y = landmarks_array[:, left_ankle+1] if landmarks_array.shape[1] > left_ankle+1 else []
            right_ankle_y = landmarks_array[:, right_ankle+1] if landmarks_array.shape[1] > right_ankle+1 else []

            gait_features['left_ankle_trajectory'] = left_ankle_y.tolist() if len(left_ankle_y) > 0 else []
            gait_features['right_ankle_trajectory'] = right_ankle_y.tolist() if len(right_ankle_y) > 0 else []

            # 보행 주기 검출 (발목 높이 기반 피크 검출)
            if len(left_ankle_y) > 10:
                from scipy.signal import find_peaks
                peaks_left, _ = find_peaks(-left_ankle_y, distance=10)  # 음수로 하여 최솟값 찾기
                peaks_right, _ = find_peaks(-right_ankle_y, distance=10)

                gait_features['left_foot_strikes'] = peaks_left.tolist()
                gait_features['right_foot_strikes'] = peaks_right.tolist()
                gait_features['estimated_cadence'] = len(peaks_left) + len(peaks_right)

        except Exception as e:
            gait_features['calculation_error'] = str(e)

        return gait_features

    def calculate_angle(self, point1, point2, point3):
        """세 점으로 각도 계산"""
        try:
            # 벡터 계산
            v1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
            v2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])

            # 코사인 값 계산
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 수치 안정성

            # 각도 변환 (라디안 -> 도)
            angle = np.arccos(cos_angle) * 180 / np.pi
            return angle
        except:
            return 0.0

    def process_single_video(self, video_pair):
        """단일 비디오 처리"""
        video_path = video_pair['video_file']
        video_name = Path(video_path).name

        print(f"🎬 처리 중: {video_name}")

        # MediaPipe 특징 추출
        mp_features = self.extract_mediapipe_features(video_path, max_frames=300)

        # 보행 특징 계산
        gait_features = {}
        if mp_features['success'] and mp_features['pose_landmarks']:
            gait_features = self.compute_gait_features(
                mp_features['pose_landmarks'],
                mp_features['visibility_scores']
            )

        # 결과 구조화
        result = {
            'video_info': video_pair,
            'mediapipe_features': {
                'success': mp_features['success'],
                'frame_count': mp_features['frame_count'],
                'processing_time': mp_features['processing_time'],
                'error_message': mp_features.get('error_message'),
                'landmark_count': len(mp_features['pose_landmarks']),
                'world_landmark_count': len(mp_features['world_landmarks'])
            },
            'gait_features': gait_features,
            'extraction_timestamp': datetime.now().isoformat()
        }

        # 스레드 안전 통계 업데이트
        with self.lock:
            self.processing_stats['total_videos'] += 1
            if mp_features['success']:
                self.processing_stats['successful'] += 1
            else:
                self.processing_stats['failed'] += 1
            self.processing_stats['total_frames'] += mp_features['frame_count']
            self.processing_stats['processing_time'] += mp_features['processing_time']

        print(f"✅ {video_name}: {mp_features['frame_count']}프레임, "
              f"{mp_features['processing_time']:.1f}초, "
              f"성공: {mp_features['success']}")

        return result

    def extract_features_batch(self, limit_videos=None):
        """배치 특징 추출"""
        print(f"\n🚀 GAVD MediaPipe 특징 추출 시작")
        print("=" * 60)

        # 비디오-주석 쌍 가져오기
        video_pairs = self.get_video_annotation_pairs()

        if limit_videos:
            video_pairs = video_pairs[:limit_videos]
            print(f"📝 제한: {limit_videos}개 비디오만 처리")

        self.processing_stats['total_videos'] = 0  # 리셋
        start_time = time.time()

        print(f"🎬 총 {len(video_pairs)}개 비디오 처리 예정")
        print(f"⚡ {self.max_workers}개 워커로 병렬 처리")

        # 병렬 처리
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 모든 작업 제출
            future_to_video = {
                executor.submit(self.process_single_video, video_pair): video_pair
                for video_pair in video_pairs
            }

            # 결과 수집
            for future in as_completed(future_to_video):
                video_pair = future_to_video[future]
                try:
                    result = future.result()
                    self.extracted_features.append(result)
                except Exception as e:
                    print(f"❌ {video_pair['video_file']} 처리 실패: {e}")

        total_time = time.time() - start_time

        # 최종 통계
        print(f"\n🎉 GAVD MediaPipe 특징 추출 완료!")
        print("=" * 60)
        print(f"📊 처리 통계:")
        print(f"   총 비디오: {len(video_pairs)}")
        print(f"   성공: {self.processing_stats['successful']}")
        print(f"   실패: {self.processing_stats['failed']}")
        print(f"   성공률: {self.processing_stats['successful']/len(video_pairs)*100:.1f}%")
        print(f"   총 프레임: {self.processing_stats['total_frames']:,}")
        print(f"   총 처리 시간: {total_time:.1f}초")
        print(f"   평균 처리 속도: {self.processing_stats['total_frames']/total_time:.1f} FPS")

        return self.extracted_features

    def analyze_extracted_features(self):
        """추출된 특징 분석"""
        if not self.extracted_features:
            print("❌ 추출된 특징이 없습니다.")
            return {}

        print(f"\n📈 추출된 특징 분석...")

        # 기본 통계
        successful_extractions = [f for f in self.extracted_features if f['mediapipe_features']['success']]

        # 패턴별 분석
        pattern_stats = defaultdict(list)
        for feature in successful_extractions:
            pattern = feature['video_info']['gait_pattern']
            pattern_stats[pattern].append(feature)

        analysis = {
            'total_videos': len(self.extracted_features),
            'successful_extractions': len(successful_extractions),
            'success_rate': len(successful_extractions) / len(self.extracted_features) * 100,
            'pattern_distribution': {pattern: len(features) for pattern, features in pattern_stats.items()},
            'average_frames_per_video': np.mean([f['mediapipe_features']['frame_count'] for f in successful_extractions]),
            'average_processing_time': np.mean([f['mediapipe_features']['processing_time'] for f in successful_extractions])
        }

        print(f"📊 특징 추출 분석 결과:")
        print(f"   총 비디오: {analysis['total_videos']}")
        print(f"   성공적 추출: {analysis['successful_extractions']}")
        print(f"   성공률: {analysis['success_rate']:.1f}%")
        print(f"   평균 프레임/비디오: {analysis['average_frames_per_video']:.1f}")
        print(f"   평균 처리시간: {analysis['average_processing_time']:.1f}초")

        print(f"\n🦴 패턴별 분포:")
        for pattern, count in analysis['pattern_distribution'].items():
            percentage = count / len(successful_extractions) * 100
            print(f"   {pattern}: {count}개 ({percentage:.1f}%)")

        return analysis

    def save_extracted_features(self, output_file=None):
        """추출된 특징 저장"""
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"gavd_mediapipe_features_{timestamp}.json"

        # 전체 결과 구조
        full_results = {
            'extraction_info': {
                'timestamp': datetime.now().isoformat(),
                'total_videos_processed': len(self.extracted_features),
                'processing_stats': self.processing_stats,
                'gavd_dataset_path': str(self.gavd_path)
            },
            'analysis_summary': self.analyze_extracted_features(),
            'extracted_features': self.extracted_features
        }

        # JSON 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)

        file_size = Path(output_file).stat().st_size / (1024*1024)  # MB
        print(f"\n💾 특징 추출 결과 저장: {output_file}")
        print(f"   파일 크기: {file_size:.1f} MB")

        return output_file

def main():
    """메인 실행 함수"""
    print("🎬 GAVD MediaPipe 특징 추출 시작")
    print("=" * 60)

    # 추출기 초기화
    extractor = GAVDMediaPipeExtractor(max_workers=6)

    try:
        # 특징 추출 (처음에는 50개 비디오로 테스트)
        print(f"🧪 테스트 모드: 50개 비디오로 시작")
        extracted_features = extractor.extract_features_batch(limit_videos=50)

        # 분석
        analysis = extractor.analyze_extracted_features()

        # 저장
        output_file = extractor.save_extracted_features()

        print(f"\n🎉 GAVD MediaPipe 특징 추출 완료!")
        print(f"📁 결과 파일: {output_file}")
        print(f"🦴 {analysis['successful_extractions']}개 비디오에서 특징 추출 성공")

        # 성공률이 좋다면 전체 처리 제안
        if analysis['success_rate'] > 80:
            print(f"\n✨ 성공률이 {analysis['success_rate']:.1f}%로 높습니다!")
            print(f"💡 전체 {extractor.get_video_annotation_pairs().__len__()}개 비디오 처리를 권장합니다.")

    except Exception as e:
        print(f"❌ 특징 추출 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()