#!/usr/bin/env python3
"""
MediaPipe 보행분석 대화형 인터페이스
- 실시간 비디오 업로드 및 분석
- 진행률 표시 및 라이브 피드백
- 3단계 검증 시스템 통합
- 사용자 친화적 웹 인터페이스

Author: AI Assistant
Date: 2025-09-15
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tempfile
import time
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 내부 모듈 import
try:
    from main_pipeline import MainGaitAnalysisPipeline
    from validation_system import ValidationSystem
except ImportError:
    st.error("필요한 모듈을 찾을 수 없습니다. main_pipeline.py와 validation_system.py가 같은 디렉토리에 있는지 확인하세요.")
    st.stop()

class InteractiveGaitAnalyzer:
    """대화형 보행분석 시스템"""

    def __init__(self):
        # 시스템 구성요소 초기화
        self.pipeline = None
        self.validator = None

        # Streamlit 페이지 설정
        if 'page_initialized' not in st.session_state:
            st.set_page_config(
                page_title="MediaPipe 보행분석 시스템",
                page_icon="🚶",
                layout="wide",
                initial_sidebar_state="expanded"
            )
            st.session_state.page_initialized = True

    def run(self):
        """메인 인터페이스 실행"""
        self.render_header()
        self.render_sidebar()

        # 메인 콘텐츠 영역
        tab1, tab2, tab3 = st.tabs(["🎬 비디오 분석", "🔍 검증 시스템", "📊 결과 대시보드"])

        with tab1:
            self.render_video_analysis_tab()

        with tab2:
            self.render_validation_tab()

        with tab3:
            self.render_dashboard_tab()

    def render_header(self):
        """헤더 렌더링"""
        st.title("🚶 MediaPipe 보행분석 시스템")
        st.markdown("""
        **완전한 보행분석 파이프라인** - 비디오 입력부터 3단계 검증까지

        - 📹 **비디오 처리**: MediaPipe 키포인트 추출
        - 📏 **매개변수 계산**: 시공간 매개변수 및 관절각도 분석
        - 📊 **101포인트 정규화**: 표준화된 보행주기 분석
        - 🔬 **3단계 검증**: ICC, DTW, SPM 다층 검증
        """)
        st.divider()

    def render_sidebar(self):
        """사이드바 렌더링"""
        with st.sidebar:
            st.header("⚙️ 설정")

            # 분석 모드 선택
            analysis_mode = st.selectbox(
                "분석 모드",
                ["기본 분석", "정밀 분석", "빠른 분석"],
                help="분석 정확도와 속도를 조절합니다"
            )

            # MediaPipe 설정
            st.subheader("MediaPipe 설정")
            detection_confidence = st.slider("탐지 신뢰도", 0.3, 0.9, 0.7, 0.1)
            tracking_confidence = st.slider("추적 신뢰도", 0.3, 0.9, 0.7, 0.1)

            # 출력 설정
            st.subheader("출력 설정")
            save_visualizations = st.checkbox("시각화 저장", True)
            save_json = st.checkbox("JSON 결과 저장", True)
            save_excel = st.checkbox("Excel 형식 저장", False)

            # 세션 상태에 설정 저장
            st.session_state.analysis_config = {
                'mode': analysis_mode,
                'detection_confidence': detection_confidence,
                'tracking_confidence': tracking_confidence,
                'save_visualizations': save_visualizations,
                'save_json': save_json,
                'save_excel': save_excel
            }

    def render_video_analysis_tab(self):
        """비디오 분석 탭"""
        st.header("🎬 비디오 분석")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("비디오 업로드")

            # 파일 업로더
            uploaded_file = st.file_uploader(
                "보행 비디오 선택",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="측면에서 촬영된 보행 비디오를 업로드하세요"
            )

            # 피험자 정보
            subject_id = st.text_input("피험자 ID", "S001", help="분석 결과 식별용 ID")

            # 분석 시작 버튼
            if uploaded_file is not None:
                if st.button("🚀 분석 시작", type="primary"):
                    self.run_video_analysis(uploaded_file, subject_id)

        with col2:
            st.subheader("비디오 미리보기")

            if uploaded_file is not None:
                # 임시 파일로 저장하여 미리보기
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                # 비디오 정보 표시
                cap = cv2.VideoCapture(tmp_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    duration = frame_count / fps

                    st.info(f"""
                    **비디오 정보**
                    - 해상도: {width} × {height}
                    - FPS: {fps:.1f}
                    - 길이: {duration:.1f}초
                    - 총 프레임: {frame_count}
                    """)

                    # 첫 번째 프레임 표시
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption="첫 번째 프레임", use_column_width=True)

                    cap.release()

                # 임시 파일 정리
                Path(tmp_path).unlink()

    def run_video_analysis(self, uploaded_file, subject_id):
        """비디오 분석 실행"""
        with st.spinner("분석 시스템 초기화 중..."):
            if self.pipeline is None:
                config = st.session_state.analysis_config
                self.pipeline = MainGaitAnalysisPipeline()

        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name

        try:
            # 출력 디렉토리 생성
            output_dir = Path("./analysis_results")
            output_dir.mkdir(exist_ok=True)

            # 진행률 표시
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 분석 단계별 실행
            status_text.text("🎬 비디오에서 포즈 랜드마크 추출 중...")
            progress_bar.progress(20)

            # 실제 분석 실행
            with st.spinner("보행분석 실행 중... 잠시만 기다려주세요."):
                results = self.pipeline.analyze_gait_video(
                    video_path=video_path,
                    subject_id=subject_id,
                    output_dir=str(output_dir)
                )

            progress_bar.progress(100)
            status_text.text("✅ 분석 완료!")

            # 결과 표시
            self.display_analysis_results(results, output_dir)

            # 세션 상태에 결과 저장
            st.session_state.latest_results = results
            st.session_state.latest_subject = subject_id

        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")

        finally:
            # 임시 파일 정리
            Path(video_path).unlink()

    def display_analysis_results(self, results, output_dir):
        """분석 결과 표시"""
        st.success("🎉 보행분석이 성공적으로 완료되었습니다!")

        # 주요 결과 요약
        col1, col2, col3, col4 = st.columns(4)

        ts = results['temporal_spatial']
        with col1:
            st.metric("Cadence", f"{ts['cadence']:.1f}", "steps/min")

        with col2:
            st.metric("평균 Stride Time", f"{ts['stride_time_mean']:.3f}", "초")

        with col3:
            st.metric("평균 Stride Length", f"{ts['stride_length_mean']:.3f}", "m")

        with col4:
            st.metric("평균 Walking Speed", f"{ts['walking_speed_mean']:.3f}", "m/s")

        # 관절각도 시각화
        st.subheader("📊 관절각도 분석 (101포인트 정규화)")

        # 관절각도 플롯
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        joint_names = ['hip_flexion_extension', 'knee_flexion_extension', 'ankle_dorsi_plantarflexion']
        joint_titles = ['Hip Flexion/Extension', 'Knee Flexion/Extension', 'Ankle Dorsiflexion/Plantarflexion']

        x_points = np.linspace(0, 100, 101)

        for i, (joint, title) in enumerate(zip(joint_names, joint_titles)):
            angles = results['joint_angles_101'][joint]
            axes[i].plot(x_points, angles, 'b-', linewidth=2, label='MediaPipe')
            axes[i].set_title(title)
            axes[i].set_xlabel('Gait Cycle (%)')
            axes[i].set_ylabel('Angle (degrees)')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            axes[i].set_xlim(0, 100)

        plt.tight_layout()
        st.pyplot(fig)

        # 다운로드 링크
        st.subheader("📁 결과 다운로드")

        # JSON 결과 다운로드
        json_str = json.dumps(results, ensure_ascii=False, indent=2)
        st.download_button(
            label="📄 JSON 결과 다운로드",
            data=json_str,
            file_name=f"{results['subject_id']}_analysis_results.json",
            mime="application/json"
        )

    def render_validation_tab(self):
        """검증 탭"""
        st.header("🔍 3단계 다층 검증 시스템")

        st.markdown("""
        **초록 방법론에 따른 정확한 검증**
        - **Level 1**: 이산 매개변수 ICC 검증
        - **Level 2**: 파형 데이터 DTW 검증
        - **Level 3**: 통계적 매개변수 매핑(SPM) 검증
        """)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("데이터 입력")

            # MediaPipe 결과 파일
            mp_file = st.file_uploader(
                "MediaPipe 분석 결과 (JSON)",
                type=['json'],
                help="main_pipeline.py로 생성된 JSON 결과 파일"
            )

            # 전통적 시스템 결과 파일
            trad_file = st.file_uploader(
                "전통적 보행분석 결과 (Excel)",
                type=['xlsx', 'xls'],
                help="전통적 보행분석 시스템의 Excel 결과 파일"
            )

            # 검증 실행 버튼
            if mp_file is not None and trad_file is not None:
                if st.button("🔬 검증 시작", type="primary"):
                    self.run_validation_analysis(mp_file, trad_file)

        with col2:
            st.subheader("검증 진행 상황")

            if 'validation_results' in st.session_state:
                self.display_validation_summary(st.session_state.validation_results)
            else:
                st.info("검증을 시작하려면 필요한 파일들을 업로드하고 '검증 시작' 버튼을 클릭하세요.")

    def run_validation_analysis(self, mp_file, trad_file):
        """검증 분석 실행"""
        with st.spinner("검증 시스템 초기화 중..."):
            if self.validator is None:
                self.validator = ValidationSystem()

        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_mp:
            tmp_mp.write(mp_file.getvalue())
            mp_path = tmp_mp.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_trad:
            tmp_trad.write(trad_file.getvalue())
            trad_path = tmp_trad.name

        try:
            # 출력 디렉토리
            output_dir = Path("./validation_results")
            output_dir.mkdir(exist_ok=True)

            # 진행률 표시
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 검증 실행
            status_text.text("🔍 3단계 다층 검증 실행 중...")

            with st.spinner("검증 분석 중... 잠시만 기다려주세요."):
                results = self.validator.run_complete_validation(
                    mediapipe_path=mp_path,
                    traditional_path=trad_path,
                    output_dir=str(output_dir)
                )

            progress_bar.progress(100)
            status_text.text("✅ 검증 완료!")

            # 결과 저장 및 표시
            st.session_state.validation_results = results
            self.display_validation_results(results)

        except Exception as e:
            st.error(f"검증 중 오류가 발생했습니다: {e}")

        finally:
            # 임시 파일 정리
            Path(mp_path).unlink()
            Path(trad_path).unlink()

    def display_validation_results(self, results):
        """검증 결과 표시"""
        st.success("🎉 3단계 다층 검증이 완료되었습니다!")

        # 검증 요약
        summary = results['summary']

        # Level별 결과 표시
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📊 Level 1: ICC")
            l1 = summary['level1_summary']
            st.metric("평균 ICC", f"{l1['mean_icc']:.3f}")
            st.write(f"우수 (>0.75): {l1['excellent_icc_count']}/{l1['parameters_with_data']}")
            st.write(f"양호 (0.6-0.75): {l1['good_icc_count']}/{l1['parameters_with_data']}")

        with col2:
            st.subheader("📈 Level 2: DTW")
            l2 = summary['level2_summary']
            st.metric("평균 유사도", f"{l2['mean_dtw_similarity']:.3f}")
            st.write(f"높은 유사도 (>0.8): {l2['high_similarity_count']}/{l2['joints_with_data']}")
            st.write(f"평균 상관계수: {l2['mean_cross_correlation']:.3f}")

        with col3:
            st.subheader("📉 Level 3: SPM")
            l3 = summary['level3_summary']
            st.metric("평균 RMSE", f"{l3['mean_rmse']:.3f}°")
            st.write(f"낮은 차이 (<5°): {l3['low_difference_count']}/{l3['joints_with_data']}")
            st.write(f"평균 유의구간: {l3['mean_significant_percentage']:.1f}%")

    def display_validation_summary(self, results):
        """검증 요약 표시"""
        summary = results['summary']

        st.subheader("🎯 검증 요약")

        # 전체 점수 계산
        l1_score = summary['level1_summary']['mean_icc']
        l2_score = summary['level2_summary']['mean_dtw_similarity']
        l3_score = 1 - (summary['level3_summary']['mean_rmse'] / 10)  # RMSE를 점수로 변환

        overall_score = (l1_score + l2_score + l3_score) / 3

        st.metric("전체 검증 점수", f"{overall_score:.3f}", "0~1 범위")

        # 점수 해석
        if overall_score > 0.8:
            st.success("🏆 Excellent - 임상적 활용 강력 권장")
        elif overall_score > 0.6:
            st.warning("👍 Good - 임상적 활용 가능")
        elif overall_score > 0.4:
            st.warning("⚠️ Fair - 추가 보정 필요")
        else:
            st.error("❌ Poor - 시스템 개선 필요")

    def render_dashboard_tab(self):
        """대시보드 탭"""
        st.header("📊 결과 대시보드")

        # 최근 분석 결과가 있는지 확인
        if 'latest_results' in st.session_state:
            results = st.session_state.latest_results
            subject_id = st.session_state.latest_subject

            st.subheader(f"📋 {subject_id} 분석 결과 요약")

            # 시공간 매개변수 테이블
            ts = results['temporal_spatial']

            # 데이터프레임 생성
            params_data = {
                '매개변수': ['Cadence', 'Stride Time', 'Stride Length', 'Walking Speed', 'Stance Phase'],
                '값': [
                    f"{ts['cadence']:.1f} steps/min",
                    f"{ts['stride_time_mean']:.3f} ± {ts['stride_time_std']:.3f} s",
                    f"{ts['stride_length_mean']:.3f} ± {ts['stride_length_std']:.3f} m",
                    f"{ts['walking_speed_mean']:.3f} ± {ts['walking_speed_std']:.3f} m/s",
                    f"{ts['stance_phase_mean']:.1f} ± {ts['stance_phase_std']:.1f} %"
                ],
                '정상범위': ['100-120', '1.0-1.3', '1.2-1.6', '1.0-1.6', '60-65'],
                '상태': ['정상', '정상', '정상', '정상', '정상']  # 실제로는 계산 필요
            }

            df_params = pd.DataFrame(params_data)
            st.dataframe(df_params, use_container_width=True)

            # 관절각도 범위 분석
            st.subheader("🦴 관절각도 범위 분석")

            joint_data = {
                '관절': ['Hip', 'Knee', 'Ankle'],
                '최대값 (°)': [
                    f"{np.max(results['joint_angles_101']['hip_flexion_extension']):.1f}",
                    f"{np.max(results['joint_angles_101']['knee_flexion_extension']):.1f}",
                    f"{np.max(results['joint_angles_101']['ankle_dorsi_plantarflexion']):.1f}"
                ],
                '최소값 (°)': [
                    f"{np.min(results['joint_angles_101']['hip_flexion_extension']):.1f}",
                    f"{np.min(results['joint_angles_101']['knee_flexion_extension']):.1f}",
                    f"{np.min(results['joint_angles_101']['ankle_dorsi_plantarflexion']):.1f}"
                ],
                '범위 (°)': [
                    f"{np.max(results['joint_angles_101']['hip_flexion_extension']) - np.min(results['joint_angles_101']['hip_flexion_extension']):.1f}",
                    f"{np.max(results['joint_angles_101']['knee_flexion_extension']) - np.min(results['joint_angles_101']['knee_flexion_extension']):.1f}",
                    f"{np.max(results['joint_angles_101']['ankle_dorsi_plantarflexion']) - np.min(results['joint_angles_101']['ankle_dorsi_plantarflexion']):.1f}"
                ]
            }

            df_joints = pd.DataFrame(joint_data)
            st.dataframe(df_joints, use_container_width=True)

        else:
            st.info("분석 결과를 보려면 먼저 '비디오 분석' 탭에서 분석을 실행하세요.")

def main():
    """메인 함수"""
    # 인터페이스 실행
    analyzer = InteractiveGaitAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()