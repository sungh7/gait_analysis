#!/usr/bin/env python3
"""
GAVD Dataset Analyzer
Enhanced MediaPipe Gait Analysis System v2.0 - GAVD Integration

실제 임상 데이터를 활용한 병적보행 검출 시스템 개발을 위한 GAVD 데이터셋 분석기

Author: Research Team
Date: 2025-09-22
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import ast
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class GAVDDatasetAnalyzer:
    """GAVD 데이터셋 분석 및 임상 주석 추출기"""

    def __init__(self, gavd_path="/data/datasets/GAVD"):
        """
        GAVD 데이터셋 초기화

        Args:
            gavd_path: GAVD 데이터셋 경로
        """
        self.gavd_path = Path(gavd_path)
        self.data_path = self.gavd_path / "data"
        self.videos_path = self.gavd_path / "videos_cut_by_view"

        # 임상 주석 파일들
        self.annotation_files = list(self.data_path.glob("GAVD_Clinical_Annotations_*.csv"))

        # 통합 데이터프레임
        self.combined_annotations = None
        self.pathological_patterns = {}
        self.camera_views = {}
        self.gait_analysis_summary = {}

        print(f"🔍 GAVD Dataset Analyzer 초기화")
        print(f"📁 데이터셋 경로: {self.gavd_path}")
        print(f"📊 주석 파일 수: {len(self.annotation_files)}")
        print(f"🎬 비디오 파일 수: {len(list(self.videos_path.glob('*.mp4')))}")

    def load_clinical_annotations(self):
        """모든 임상 주석 파일 로드 및 통합"""
        print(f"\n📖 임상 주석 데이터 로드 중...")

        all_annotations = []

        for i, file_path in enumerate(self.annotation_files, 1):
            print(f"   {i}/5: {file_path.name}")

            try:
                df = pd.read_csv(file_path)
                all_annotations.append(df)
                print(f"      ✅ {len(df):,}개 행 로드 성공")
            except Exception as e:
                print(f"      ❌ 오류: {e}")

        # 모든 데이터 통합
        self.combined_annotations = pd.concat(all_annotations, ignore_index=True)

        print(f"\n📊 통합 완료:")
        print(f"   총 데이터 행 수: {len(self.combined_annotations):,}")
        print(f"   고유 시퀀스 ID: {self.combined_annotations['seq'].nunique()}")
        print(f"   고유 비디오 ID: {self.combined_annotations['id'].nunique()}")

        return self.combined_annotations

    def analyze_pathological_patterns(self):
        """병적보행 패턴 분석"""
        if self.combined_annotations is None:
            self.load_clinical_annotations()

        print(f"\n🦴 병적보행 패턴 분석...")

        # 기본 분포 분석
        gait_pattern_dist = self.combined_annotations['gait_pat'].value_counts()
        dataset_type_dist = self.combined_annotations['dataset'].value_counts()

        print(f"\n📈 보행 패턴 분포:")
        for pattern, count in gait_pattern_dist.items():
            percentage = (count / len(self.combined_annotations)) * 100
            print(f"   {pattern}: {count:,}개 ({percentage:.1f}%)")

        print(f"\n📋 데이터셋 유형 분포:")
        for dataset_type, count in dataset_type_dist.items():
            percentage = (count / len(self.combined_annotations)) * 100
            print(f"   {dataset_type}: {count:,}개 ({percentage:.1f}%)")

        # 고유 비디오별 병적보행 패턴
        unique_videos = self.combined_annotations.groupby('id').agg({
            'gait_pat': 'first',
            'dataset': 'first',
            'cam_view': lambda x: list(set(x)),
            'seq': 'nunique'
        }).reset_index()

        print(f"\n🎬 고유 비디오 분석:")
        print(f"   총 고유 비디오: {len(unique_videos)}")

        video_pattern_dist = unique_videos['gait_pat'].value_counts()
        print(f"\n📊 비디오별 보행 패턴:")
        for pattern, count in video_pattern_dist.items():
            percentage = (count / len(unique_videos)) * 100
            print(f"   {pattern}: {count}개 비디오 ({percentage:.1f}%)")

        # 카메라 뷰 분석
        all_views = []
        for views in unique_videos['cam_view']:
            all_views.extend(views)

        view_dist = Counter(all_views)
        print(f"\n📷 카메라 뷰 분포:")
        for view, count in view_dist.most_common():
            percentage = (count / len(unique_videos)) * 100
            print(f"   {view}: {count}개 비디오 ({percentage:.1f}%)")

        # 병적보행 패턴별 상세 분석
        self.pathological_patterns = {
            'pattern_distribution': gait_pattern_dist.to_dict(),
            'dataset_distribution': dataset_type_dist.to_dict(),
            'unique_videos': unique_videos.to_dict('records'),
            'video_pattern_distribution': video_pattern_dist.to_dict(),
            'camera_view_distribution': dict(view_dist)
        }

        return self.pathological_patterns

    def analyze_camera_views(self):
        """다중 카메라 뷰 분석"""
        if self.combined_annotations is None:
            self.load_clinical_annotations()

        print(f"\n📷 다중 카메라 뷰 분석...")

        # 카메라 뷰별 데이터 분포
        view_analysis = self.combined_annotations.groupby(['cam_view', 'gait_pat']).size().unstack(fill_value=0)

        print(f"\n📊 카메라 뷰별 보행 패턴 분포:")
        print(view_analysis)

        # 다중 뷰를 가진 비디오 찾기
        multi_view_videos = self.combined_annotations.groupby('id')['cam_view'].nunique()
        multi_view_videos = multi_view_videos[multi_view_videos > 1]

        print(f"\n🎥 다중 뷰 비디오 분석:")
        print(f"   다중 뷰 비디오 수: {len(multi_view_videos)}")
        print(f"   단일 뷰 비디오 수: {self.combined_annotations['id'].nunique() - len(multi_view_videos)}")

        # 다중 뷰 비디오 상세 정보
        if len(multi_view_videos) > 0:
            print(f"\n📋 다중 뷰 비디오 예시 (최대 10개):")
            for i, (video_id, view_count) in enumerate(multi_view_videos.head(10).items()):
                views = self.combined_annotations[self.combined_annotations['id'] == video_id]['cam_view'].unique()
                # NaN 값 처리
                views = [str(view) for view in views if pd.notna(view)]
                gait_pattern = self.combined_annotations[self.combined_annotations['id'] == video_id]['gait_pat'].iloc[0]
                print(f"   {i+1}. {video_id}: {view_count}개 뷰 ({', '.join(views)}) - {gait_pattern}")

        self.camera_views = {
            'view_pattern_matrix': view_analysis.to_dict(),
            'multi_view_videos': multi_view_videos.to_dict(),
            'multi_view_count': len(multi_view_videos),
            'single_view_count': self.combined_annotations['id'].nunique() - len(multi_view_videos)
        }

        return self.camera_views

    def match_videos_with_annotations(self, side_view_only=True):
        """비디오 파일과 임상 주석 매칭 (side view 전용 옵션)"""
        print(f"\n🎬 비디오 파일과 임상 주석 매칭...")
        if side_view_only:
            print(f"   📐 Side view 전용 모드 활성화")

        # 실제 비디오 파일 목록
        video_files = list(self.videos_path.glob("*.mp4"))

        # Side view 필터링
        if side_view_only:
            side_view_files = []
            for video_file in video_files:
                filename = video_file.stem
                parts = filename.split('_')
                if len(parts) >= 3:
                    view = '_'.join(parts[1:-1])
                    if view in ['left_side', 'right_side']:
                        side_view_files.append(video_file)
            video_files = side_view_files
            print(f"   📐 Side view 필터링 후: {len(video_files)}개")

        video_file_names = [f.stem for f in video_files]
        print(f"   실제 비디오 파일: {len(video_files)}개")

        # 주석에서 비디오 ID 추출
        annotated_video_ids = set(self.combined_annotations['id'].unique())
        print(f"   주석 데이터 비디오 ID: {len(annotated_video_ids)}개")

        # 파일명에서 비디오 ID 추출 (파일명 패턴: {video_id}_{view}_{frame_range}.mp4)
        extracted_ids = set()
        for filename in video_file_names:
            parts = filename.split('_')
            if len(parts) >= 3:
                video_id = parts[0]
                extracted_ids.add(video_id)

        print(f"   파일명에서 추출된 비디오 ID: {len(extracted_ids)}개")

        # 매칭 분석
        matched_ids = annotated_video_ids.intersection(extracted_ids)
        unmatched_annotations = annotated_video_ids - extracted_ids
        unmatched_files = extracted_ids - annotated_video_ids

        print(f"\n📊 매칭 결과:")
        print(f"   매칭된 비디오 ID: {len(matched_ids)}개")
        print(f"   주석만 있는 ID: {len(unmatched_annotations)}개")
        print(f"   파일만 있는 ID: {len(unmatched_files)}개")

        # 매칭된 비디오의 병적보행 패턴 분석
        matched_annotations = self.combined_annotations[
            self.combined_annotations['id'].isin(matched_ids)
        ]

        matched_pattern_dist = matched_annotations.groupby('id')['gait_pat'].first().value_counts()
        print(f"\n🦴 매칭된 비디오의 병적보행 패턴:")
        for pattern, count in matched_pattern_dist.items():
            percentage = (count / len(matched_ids)) * 100
            print(f"   {pattern}: {count}개 ({percentage:.1f}%)")

        # 사용 가능한 비디오-주석 쌍 생성
        available_pairs = []

        for video_file in video_files:
            filename = video_file.stem
            parts = filename.split('_')

            if len(parts) >= 3:
                video_id = parts[0]
                view = '_'.join(parts[1:-1])  # view가 'left_side' 같이 언더스코어 포함할 수 있음
                frame_range = parts[-1]

                if video_id in annotated_video_ids:
                    # 카메라 뷰 표기 통일 (파일명: left_side <-> 주석: left side)
                    normalized_view = view.replace('_', ' ')

                    # 해당 비디오의 주석 정보 가져오기
                    video_annotations = self.combined_annotations[
                        (self.combined_annotations['id'] == video_id) &
                        (self.combined_annotations['cam_view'] == normalized_view)
                    ]

                    if not video_annotations.empty:
                        gait_pattern = video_annotations['gait_pat'].iloc[0]
                        dataset_type = video_annotations['dataset'].iloc[0]

                        available_pairs.append({
                            'video_file': str(video_file),
                            'video_id': video_id,
                            'camera_view': view,
                            'frame_range': frame_range,
                            'gait_pattern': gait_pattern,
                            'dataset_type': dataset_type,
                            'annotation_count': len(video_annotations)
                        })

        print(f"\n✅ 사용 가능한 비디오-주석 쌍: {len(available_pairs)}개")

        # 패턴별 사용 가능한 데이터 수
        pattern_counts = Counter([pair['gait_pattern'] for pair in available_pairs])
        print(f"\n📈 패턴별 사용 가능한 비디오 수:")
        for pattern, count in pattern_counts.most_common():
            percentage = (count / len(available_pairs)) * 100
            print(f"   {pattern}: {count}개 ({percentage:.1f}%)")

        # 카메라 뷰 분포 (side view only일 때)
        if side_view_only and available_pairs:
            view_counts = Counter([pair['camera_view'] for pair in available_pairs])
            print(f"\n📷 Side view 분포:")
            for view, count in view_counts.most_common():
                print(f"   {view}: {count}개")

        return available_pairs

    def generate_dataset_summary(self):
        """GAVD 데이터셋 종합 요약 생성"""
        print(f"\n📋 GAVD 데이터셋 종합 요약 생성...")

        # 기본 정보 수집
        if self.combined_annotations is None:
            self.load_clinical_annotations()

        if not self.pathological_patterns:
            self.analyze_pathological_patterns()

        if not self.camera_views:
            self.analyze_camera_views()

        available_pairs = self.match_videos_with_annotations()

        # 종합 요약 생성
        summary = {
            'dataset_info': {
                'total_annotation_rows': len(self.combined_annotations),
                'unique_video_ids': self.combined_annotations['id'].nunique(),
                'unique_sequences': self.combined_annotations['seq'].nunique(),
                'total_video_files': len(list(self.videos_path.glob("*.mp4"))),
                'available_video_annotation_pairs': len(available_pairs)
            },
            'pathological_patterns': self.pathological_patterns,
            'camera_views': self.camera_views,
            'clinical_applications': {
                'parkinsons_videos': len([p for p in available_pairs if p['gait_pattern'] == 'parkinsons']),
                'normal_videos': len([p for p in available_pairs if p['dataset_type'] == 'Normal Gait']),
                'abnormal_videos': len([p for p in available_pairs if p['dataset_type'] == 'Abnormal Gait']),
                'multi_view_potential': self.camera_views['multi_view_count']
            },
            'technical_specs': {
                'video_format': 'MP4',
                'annotation_format': 'CSV',
                'bounding_box_available': True,
                'frame_level_annotations': True,
                'clinical_labels': True
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'available_pairs_sample': available_pairs[:10]  # 샘플 10개
        }

        self.gait_analysis_summary = summary

        # 요약 출력
        print(f"\n🎯 GAVD 데이터셋 분석 완료!")
        print(f"=" * 60)
        print(f"📊 기본 통계:")
        print(f"   총 주석 데이터: {summary['dataset_info']['total_annotation_rows']:,}행")
        print(f"   고유 비디오 ID: {summary['dataset_info']['unique_video_ids']}개")
        print(f"   실제 비디오 파일: {summary['dataset_info']['total_video_files']}개")
        print(f"   사용 가능한 비디오-주석 쌍: {summary['dataset_info']['available_video_annotation_pairs']}개")

        print(f"\n🦴 임상 활용 가능성:")
        print(f"   파킨슨병 비디오: {summary['clinical_applications']['parkinsons_videos']}개")
        print(f"   정상 보행 비디오: {summary['clinical_applications']['normal_videos']}개")
        print(f"   비정상 보행 비디오: {summary['clinical_applications']['abnormal_videos']}개")
        print(f"   다중 뷰 비디오: {summary['clinical_applications']['multi_view_potential']}개")

        return summary

    def save_analysis_results(self, output_file="gavd_dataset_analysis.json"):
        """분석 결과 저장"""
        if not self.gait_analysis_summary:
            self.generate_dataset_summary()

        output_path = Path(output_file)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.gait_analysis_summary, f, indent=2, ensure_ascii=False)

        print(f"\n💾 분석 결과 저장: {output_path}")
        print(f"   파일 크기: {output_path.stat().st_size / 1024:.1f} KB")

        return output_path

    def create_visualization(self, save_path="gavd_dataset_visualization.png"):
        """데이터셋 분석 시각화"""
        if not self.pathological_patterns:
            self.analyze_pathological_patterns()

        # 시각화 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('GAVD Dataset Analysis Visualization', fontsize=16, fontweight='bold')

        # 1. 보행 패턴 분포
        pattern_data = self.pathological_patterns['video_pattern_distribution']
        axes[0, 0].pie(pattern_data.values(), labels=pattern_data.keys(), autopct='%1.1f%%')
        axes[0, 0].set_title('Gait Pattern Distribution')

        # 2. 데이터셋 유형 분포
        dataset_data = self.pathological_patterns['dataset_distribution']
        axes[0, 1].bar(dataset_data.keys(), dataset_data.values())
        axes[0, 1].set_title('Dataset Type Distribution')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. 카메라 뷰 분포
        view_data = self.pathological_patterns.get('camera_view_distribution', {})
        if view_data:
            # NaN 값 제거
            clean_view_data = {k: v for k, v in view_data.items() if pd.notna(k) and k != 'nan'}
            if clean_view_data:
                axes[1, 0].bar(clean_view_data.keys(), clean_view_data.values())
                axes[1, 0].set_title('Camera View Distribution')
                axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. 시간별 프레임 분포 (샘플)
        if self.combined_annotations is not None:
            frame_counts = self.combined_annotations.groupby('id')['frame_num'].count()
            axes[1, 1].hist(frame_counts, bins=20, alpha=0.7)
            axes[1, 1].set_title('Frames per Video Distribution')
            axes[1, 1].set_xlabel('Number of Frames')
            axes[1, 1].set_ylabel('Number of Videos')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 시각화 저장: {save_path}")

        return save_path

def main():
    """메인 실행 함수"""
    print("🔍 GAVD 데이터셋 분석 시작")
    print("=" * 60)

    # 분석기 초기화
    analyzer = GAVDDatasetAnalyzer()

    try:
        # 1. 임상 주석 로드
        analyzer.load_clinical_annotations()

        # 2. 병적보행 패턴 분석
        analyzer.analyze_pathological_patterns()

        # 3. 카메라 뷰 분석
        analyzer.analyze_camera_views()

        # 4. 종합 요약 생성
        summary = analyzer.generate_dataset_summary()

        # 5. 결과 저장
        output_file = f"gavd_dataset_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        analyzer.save_analysis_results(output_file)

        # 6. 시각화 생성
        viz_file = f"gavd_dataset_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        analyzer.create_visualization(viz_file)

        print(f"\n🎉 GAVD 데이터셋 분석 완료!")
        print(f"📋 요약: {len(summary['dataset_info']['available_video_annotation_pairs'])}개 비디오-주석 쌍 사용 가능")
        print(f"🦴 임상 데이터: 파킨슨병 {summary['clinical_applications']['parkinsons_videos']}개, "
              f"정상 {summary['clinical_applications']['normal_videos']}개, "
              f"비정상 {summary['clinical_applications']['abnormal_videos']}개")

    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()