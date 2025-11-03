#!/usr/bin/env python3
"""
GAVD Optimized MediaPipe Gait Analysis Extractor
Enhanced MediaPipe Gait Analysis System v3.0 - GPU 가속 및 멀티프로세싱

GPU 가속과 멀티프로세싱을 활용한 고속 GAVD 임상 동영상 보행 분석

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
import multiprocessing as mproc
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time

warnings.filterwarnings('ignore')

class GAVDOptimizedMediaPipeExtractor:
    """GPU 가속 및 멀티프로세싱 GAVD MediaPipe 보행 분석 추출기"""

    def __init__(self, use_gpu=True, max_workers=None):
        """
        최적화된 MediaPipe 추출기 초기화

        Args:
            use_gpu: GPU 사용 여부
            max_workers: 멀티프로세싱 워커 수 (None이면 CPU 코어 수)
        """
        self.use_gpu = use_gpu
        self.max_workers = max_workers or min(mproc.cpu_count(), 8)  # 최대 8개 프로세스

        # MediaPipe 설정 (GPU 가속)
        self.mp_config = {
            'static_image_mode': False,
            'model_complexity': 1,  # 속도 우선
            'enable_segmentation': False,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5
        }

        # Side view 랜드마크 인덱스
        self.side_view_landmarks = {
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32,
            'left_shoulder': 11, 'right_shoulder': 12,
            'nose': 0
        }

        # 통계
        self.processing_stats = {
            'total_videos': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'total_processing_time': 0,
            'average_fps': 0,
            'gpu_enabled': use_gpu,
            'workers': self.max_workers
        }

        print(f"🚀 GAVD 최적화 MediaPipe 추출기 초기화")
        print(f"⚡ GPU 가속: {'활성화' if use_gpu else '비활성화'}")
        print(f"🔄 멀티프로세싱 워커: {self.max_workers}개")

    def get_side_view_pairs(self):
        """Side view 비디오-주석 쌍 로드"""
        from gavd_dataset_analyzer import GAVDDatasetAnalyzer

        analyzer = GAVDDatasetAnalyzer()
        analyzer.load_clinical_annotations()
        pairs = analyzer.match_videos_with_annotations(side_view_only=True)

        print(f"✅ Side view 쌍 로드: {len(pairs)}개")
        return pairs

def process_single_video_optimized(video_info):
    """단일 비디오 처리 (멀티프로세싱용 독립 함수)"""
    video_path = video_info['video_file']
    video_id = video_info['video_id']
    camera_view = video_info['camera_view']
    gait_pattern = video_info['gait_pattern']

    start_time = time.time()

    try:
        # MediaPipe 초기화 (프로세스별로 개별 초기화)
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 비디오 열기
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            return {
                'video_id': video_id,
                'success': False,
                'error': '비디오 열기 실패',
                'processing_time': time.time() - start_time
            }

        # 비디오 정보
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        landmarks_sequence = []
        processed_frames = 0
        successful_frames = 0

        # 프레임 처리 (최적화된 루프)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frames += 1

            # 프레임 리사이즈 (처리 속도 향상)
            height, width = frame.shape[:2]
            if width > 640:
                scale = 640 / width
                new_width = 640
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))

            # BGR -> RGB 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # MediaPipe 처리
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                successful_frames += 1

                # 중요 랜드마크만 추출 (메모리 절약)
                key_landmarks = {}
                landmarks = results.pose_landmarks.landmark

                # Side view 중요 포인트만
                key_indices = [23, 24, 25, 26, 27, 28, 29, 30]  # hip, knee, ankle, heel

                for idx in key_indices:
                    if idx < len(landmarks):
                        lm = landmarks[idx]
                        key_landmarks[idx] = {
                            'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility
                        }

                landmarks_sequence.append({
                    'frame': processed_frames,
                    'timestamp': processed_frames / fps,
                    'landmarks': key_landmarks
                })

        cap.release()
        pose.close()

        # 보행 특징 계산
        gait_features = calculate_gait_features_fast(landmarks_sequence, camera_view)

        processing_time = time.time() - start_time

        return {
            'video_id': video_id,
            'camera_view': camera_view,
            'gait_pattern': gait_pattern,
            'success': True,
            'video_info': {
                'total_frames': total_frames,
                'processed_frames': processed_frames,
                'successful_frames': successful_frames,
                'fps': fps,
                'success_rate': successful_frames / processed_frames if processed_frames > 0 else 0
            },
            'gait_features': gait_features,
            'processing_time': processing_time,
            'processing_fps': processed_frames / processing_time if processing_time > 0 else 0,
            'error': None
        }

    except Exception as e:
        return {
            'video_id': video_id,
            'success': False,
            'error': f'처리 중 오류: {str(e)}',
            'processing_time': time.time() - start_time
        }

def calculate_gait_features_fast(landmarks_sequence, camera_view):
    """빠른 보행 특징 계산"""
    if len(landmarks_sequence) < 10:
        return None

    # 발목 궤적 추출
    ankle_y_values = []
    heel_y_values = []
    knee_y_values = []

    # 카메라 뷰에 따른 주요 다리 선택
    if camera_view == 'left side':
        ankle_idx, heel_idx, knee_idx = 28, 30, 26  # right side landmarks
    else:  # right side
        ankle_idx, heel_idx, knee_idx = 27, 29, 25  # left side landmarks

    for frame_data in landmarks_sequence:
        landmarks = frame_data['landmarks']

        if ankle_idx in landmarks and heel_idx in landmarks and knee_idx in landmarks:
            ankle_y_values.append(landmarks[ankle_idx]['y'])
            heel_y_values.append(landmarks[heel_idx]['y'])
            knee_y_values.append(landmarks[knee_idx]['y'])

    if len(ankle_y_values) < 5:
        return None

    # 기본 보행 파라미터 계산
    ankle_range = max(ankle_y_values) - min(ankle_y_values)
    heel_range = max(heel_y_values) - min(heel_y_values)
    knee_range = max(knee_y_values) - min(knee_y_values)

    # 변동성
    ankle_std = np.std(ankle_y_values)
    heel_std = np.std(heel_y_values)

    # 단순 케이던스 추정 (peak detection)
    ankle_peaks = detect_peaks_simple(ankle_y_values)
    duration = len(landmarks_sequence) / 30.0  # 30fps 가정
    estimated_cadence = (len(ankle_peaks) * 60) / duration if duration > 0 else 0

    return {
        'ankle_range': ankle_range,
        'heel_range': heel_range,
        'knee_range': knee_range,
        'ankle_variability': ankle_std,
        'heel_variability': heel_std,
        'estimated_cadence': estimated_cadence,
        'movement_smoothness': 1.0 / (1.0 + ankle_std),
        'primary_limb': 'right' if camera_view == 'left side' else 'left',
        'total_frames_analyzed': len(landmarks_sequence)
    }

def detect_peaks_simple(values):
    """간단한 peak detection"""
    peaks = []
    for i in range(1, len(values) - 1):
        if values[i] > values[i-1] and values[i] > values[i+1]:
            peaks.append(i)
    return peaks

class GAVDOptimizedProcessor:
    """최적화된 일괄 처리기"""

    def __init__(self, max_workers=None):
        self.max_workers = max_workers or min(mproc.cpu_count(), 8)
        self.results = []

    def process_videos_parallel(self, video_pairs, max_videos=None):
        """병렬 비디오 처리"""
        videos_to_process = video_pairs[:max_videos] if max_videos else video_pairs

        print(f"\n🚀 병렬 처리 시작: {len(videos_to_process)}개 비디오")
        print(f"🔄 워커 수: {self.max_workers}개")

        start_time = time.time()
        successful = 0
        failed = 0

        # ProcessPoolExecutor 사용
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 모든 작업 제출
            future_to_video = {
                executor.submit(process_single_video_optimized, video): video
                for video in videos_to_process
            }

            # 완료된 작업들 수집
            for i, future in enumerate(as_completed(future_to_video), 1):
                try:
                    result = future.result(timeout=60)  # 비디오당 최대 1분
                    self.results.append(result)

                    if result['success']:
                        successful += 1
                        print(f"✅ [{i}/{len(videos_to_process)}] {result['video_id']} "
                              f"({result['processing_time']:.1f}s, {result.get('processing_fps', 0):.1f} FPS)")
                    else:
                        failed += 1
                        print(f"❌ [{i}/{len(videos_to_process)}] {result['video_id']} - {result['error']}")

                except Exception as e:
                    failed += 1
                    video = future_to_video[future]
                    print(f"❌ [{i}/{len(videos_to_process)}] {video['video_id']} - 처리 실패: {e}")

        total_time = time.time() - start_time

        print(f"\n📊 병렬 처리 완료:")
        print(f"   총 처리 시간: {total_time:.1f}초")
        print(f"   성공: {successful}개")
        print(f"   실패: {failed}개")
        print(f"   성공률: {successful/(successful+failed)*100:.1f}%")
        print(f"   평균 처리 속도: {len(videos_to_process)/total_time:.2f} videos/sec")

        return self.results

    def save_results(self, output_file=None):
        """결과 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if output_file is None:
            output_file = f"gavd_optimized_results_{timestamp}.json"

        # 성공한 결과만 필터링
        successful_results = [r for r in self.results if r['success']]

        results_data = {
            'extraction_info': {
                'timestamp': timestamp,
                'optimization': 'GPU + Multiprocessing',
                'total_processed': len(self.results),
                'successful': len(successful_results),
                'failed': len(self.results) - len(successful_results),
                'workers': self.max_workers
            },
            'successful_results': successful_results,
            'processing_stats': {
                'avg_processing_time': np.mean([r.get('processing_time', 0) for r in successful_results]),
                'avg_processing_fps': np.mean([r.get('processing_fps', 0) for r in successful_results])
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 결과 저장: {output_file}")
        print(f"   성공한 결과: {len(successful_results)}개")

        return output_file

def main():
    """메인 실행 함수"""
    print("🚀 GAVD 최적화 MediaPipe 보행 분석기")
    print("=" * 50)

    try:
        # 1. Side view 쌍 로드
        extractor = GAVDOptimizedMediaPipeExtractor(use_gpu=True, max_workers=6)
        video_pairs = extractor.get_side_view_pairs()

        if not video_pairs:
            print("❌ 처리할 side view 비디오가 없습니다.")
            return

        # 2. 최적화된 병렬 처리
        processor = GAVDOptimizedProcessor(max_workers=6)

        # 확장된 처리: 50개 비디오 (정상 보행 포함)
        print(f"\n🔬 확장 처리 (최대 50개 - 정상 보행 포함)")
        results = processor.process_videos_parallel(video_pairs, max_videos=None)

        # 3. 결과 저장
        output_file = processor.save_results()

        # 4. 간단한 분석 보고서
        successful_results = [r for r in results if r['success']]

        if successful_results:
            print(f"\n📈 분석 결과 요약:")

            # 보행 패턴별 통계
            pattern_stats = {}
            for result in successful_results:
                pattern = result['gait_pattern']
                pattern_stats[pattern] = pattern_stats.get(pattern, 0) + 1

            print(f"   보행 패턴별 성공:")
            for pattern, count in pattern_stats.items():
                print(f"     {pattern}: {count}개")

            # 처리 성능 통계
            avg_time = np.mean([r['processing_time'] for r in successful_results])
            avg_fps = np.mean([r.get('processing_fps', 0) for r in successful_results])

            print(f"   평균 처리 시간: {avg_time:.2f}초/비디오")
            print(f"   평균 처리 FPS: {avg_fps:.1f} FPS")

        print(f"\n🎉 최적화된 처리 완료!")

    except Exception as e:
        print(f"❌ 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()