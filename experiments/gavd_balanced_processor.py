#!/usr/bin/env python3
"""
GAVD Balanced Dataset Processor
Enhanced MediaPipe Gait Analysis System v3.0 - 균형잡힌 데이터 처리

정상/병적보행이 균형잡히게 포함된 GAVD 데이터셋 처리

Author: Research Team
Date: 2025-09-22
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict
import random
from gavd_optimized_mediapipe_extractor import GAVDOptimizedProcessor, process_single_video_optimized
from gavd_dataset_analyzer import GAVDDatasetAnalyzer
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

class GAVDBalancedProcessor:
    """균형잡힌 GAVD 데이터셋 처리기"""

    def __init__(self, max_workers=6):
        """
        균형잡힌 처리기 초기화

        Args:
            max_workers: 멀티프로세싱 워커 수
        """
        self.max_workers = max_workers
        self.analyzer = GAVDDatasetAnalyzer()
        self.side_view_pairs = []
        self.results = []

        print(f"⚖️  GAVD 균형잡힌 데이터셋 처리기 초기화")
        print(f"🔄 워커 수: {max_workers}개")

    def load_and_balance_dataset(self, target_samples_per_class=15):
        """균형잡힌 데이터셋 로드 및 샘플링"""
        print(f"\n📊 균형잡힌 데이터셋 구성...")

        # GAVD 데이터 로드
        self.analyzer.load_clinical_annotations()
        all_pairs = self.analyzer.match_videos_with_annotations(side_view_only=True)

        print(f"   전체 side view 쌍: {len(all_pairs)}개")

        # 패턴별 그룹화
        pattern_groups = defaultdict(list)
        for pair in all_pairs:
            pattern = pair['gait_pattern']
            pattern_groups[pattern].append(pair)

        print(f"\n📈 패턴별 분포:")
        for pattern, pairs in pattern_groups.items():
            print(f"   {pattern}: {len(pairs)}개")

        # 균형잡힌 샘플링
        balanced_pairs = []

        # 먼저 정상 보행 확보
        if 'normal' in pattern_groups:
            normal_pairs = pattern_groups['normal']
            normal_sample_count = min(target_samples_per_class, len(normal_pairs))
            selected_normal = random.sample(normal_pairs, normal_sample_count)
            balanced_pairs.extend(selected_normal)
            print(f"✅ normal: {len(selected_normal)}개 선택")

        # 병적 보행 패턴들 균형있게 샘플링
        pathological_patterns = [p for p in pattern_groups.keys() if p != 'normal']

        for pattern in pathological_patterns:
            pairs = pattern_groups[pattern]
            # 클래스별 최소 3개는 확보하되, 목표 수를 넘지 않도록
            sample_count = min(max(3, target_samples_per_class), len(pairs))
            if len(pairs) >= 3:  # 최소 3개 이상인 패턴만
                selected_pairs = random.sample(pairs, sample_count)
                balanced_pairs.extend(selected_pairs)
                print(f"✅ {pattern}: {len(selected_pairs)}개 선택")
            else:
                print(f"⚠️  {pattern}: {len(pairs)}개 (최소 3개 미만으로 제외)")

        # 랜덤 섞기
        random.shuffle(balanced_pairs)

        self.side_view_pairs = balanced_pairs

        print(f"\n⚖️  균형잡힌 데이터셋 구성 완료:")
        print(f"   총 선택된 샘플: {len(self.side_view_pairs)}개")

        # 최종 분포 확인
        final_distribution = defaultdict(int)
        for pair in self.side_view_pairs:
            final_distribution[pair['gait_pattern']] += 1

        print(f"\n📊 최종 균형잡힌 분포:")
        for pattern, count in final_distribution.items():
            print(f"   {pattern}: {count}개")

        return self.side_view_pairs

    def process_balanced_dataset(self):
        """균형잡힌 데이터셋 처리"""
        if not self.side_view_pairs:
            print(f"❌ 처리할 데이터가 없습니다. 먼저 load_and_balance_dataset()을 실행하세요.")
            return []

        print(f"\n🚀 균형잡힌 데이터셋 병렬 처리 시작...")
        print(f"   처리할 비디오: {len(self.side_view_pairs)}개")
        print(f"   워커 수: {self.max_workers}개")

        start_time = time.time()
        successful = 0
        failed = 0

        # ProcessPoolExecutor 사용
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 모든 작업 제출
            future_to_video = {
                executor.submit(process_single_video_optimized, video): video
                for video in self.side_view_pairs
            }

            # 완료된 작업들 수집
            for i, future in enumerate(as_completed(future_to_video), 1):
                try:
                    result = future.result(timeout=60)  # 비디오당 최대 1분
                    self.results.append(result)

                    if result['success']:
                        successful += 1
                        print(f"✅ [{i}/{len(self.side_view_pairs)}] {result['video_id']} "
                              f"({result['gait_pattern']}) - {result['processing_time']:.1f}s")
                    else:
                        failed += 1
                        print(f"❌ [{i}/{len(self.side_view_pairs)}] {result['video_id']} - {result['error']}")

                except Exception as e:
                    failed += 1
                    video = future_to_video[future]
                    print(f"❌ [{i}/{len(self.side_view_pairs)}] {video['video_id']} - 처리 실패: {e}")

        total_time = time.time() - start_time

        print(f"\n📊 균형잡힌 데이터셋 처리 완료:")
        print(f"   총 처리 시간: {total_time:.1f}초")
        print(f"   성공: {successful}개")
        print(f"   실패: {failed}개")
        print(f"   성공률: {successful/(successful+failed)*100:.1f}%")

        # 성공한 결과의 패턴 분포 확인
        successful_results = [r for r in self.results if r['success']]
        pattern_distribution = defaultdict(int)
        for result in successful_results:
            pattern_distribution[result['gait_pattern']] += 1

        print(f"\n🎯 성공적으로 처리된 패턴 분포:")
        for pattern, count in pattern_distribution.items():
            print(f"   {pattern}: {count}개")

        return self.results

    def save_balanced_results(self, output_file=None):
        """균형잡힌 처리 결과 저장"""
        if not self.results:
            print(f"❌ 저장할 결과가 없습니다.")
            return None

        timestamp = time.strftime('%Y%m%d_%H%M%S')

        if output_file is None:
            output_file = f"gavd_balanced_results_{timestamp}.json"

        # 성공한 결과만 필터링
        successful_results = [r for r in self.results if r['success']]

        results_data = {
            'extraction_info': {
                'timestamp': timestamp,
                'processing_type': 'Balanced Dataset',
                'total_processed': len(self.results),
                'successful': len(successful_results),
                'failed': len(self.results) - len(successful_results),
                'workers': self.max_workers,
                'sampling_strategy': 'balanced_per_class'
            },
            'successful_results': successful_results,
            'processing_stats': {
                'avg_processing_time': np.mean([r.get('processing_time', 0) for r in successful_results]),
                'avg_processing_fps': np.mean([r.get('processing_fps', 0) for r in successful_results])
            },
            'pattern_distribution': {
                pattern: len([r for r in successful_results if r['gait_pattern'] == pattern])
                for pattern in set(r['gait_pattern'] for r in successful_results)
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 균형잡힌 결과 저장: {output_file}")
        print(f"   성공한 결과: {len(successful_results)}개")
        print(f"   패턴 분포: {results_data['pattern_distribution']}")

        return output_file

def main():
    """메인 실행 함수"""
    print("⚖️  GAVD 균형잡힌 데이터셋 처리기")
    print("=" * 50)

    try:
        # 균형잡힌 처리기 초기화
        processor = GAVDBalancedProcessor(max_workers=6)

        # 1. 균형잡힌 데이터셋 로드 (패턴당 10개씩)
        print(f"\n📊 1단계: 균형잡힌 데이터셋 구성")
        balanced_pairs = processor.load_and_balance_dataset(target_samples_per_class=10)

        if not balanced_pairs:
            print(f"❌ 균형잡힌 데이터셋 구성 실패")
            return

        # 2. 균형잡힌 데이터셋 처리
        print(f"\n🚀 2단계: 균형잡힌 데이터셋 처리")
        results = processor.process_balanced_dataset()

        # 3. 결과 저장
        print(f"\n💾 3단계: 결과 저장")
        output_file = processor.save_balanced_results()

        print(f"\n🎉 균형잡힌 데이터셋 처리 완료!")
        print(f"💾 결과 파일: {output_file}")

        # 4. 간단한 통계
        successful_results = [r for r in results if r['success']]
        if successful_results:
            normal_count = len([r for r in successful_results if r['gait_pattern'] == 'normal'])
            pathological_count = len(successful_results) - normal_count

            print(f"\n📈 최종 통계:")
            print(f"   정상 보행: {normal_count}개")
            print(f"   병적 보행: {pathological_count}개")
            print(f"   총 성공: {len(successful_results)}개")

            if normal_count > 0 and pathological_count > 0:
                print(f"✅ 정상/병적 보행 데이터 균형 확보 - 분류 학습 가능!")
            else:
                print(f"⚠️  정상 또는 병적 보행 데이터 부족")

    except Exception as e:
        print(f"❌ 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 재현 가능한 결과를 위한 시드 설정
    random.seed(42)
    np.random.seed(42)

    main()