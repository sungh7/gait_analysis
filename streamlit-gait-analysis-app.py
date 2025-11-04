import streamlit as st
import cv2
import numpy as np
import tempfile
import mediapipe as mp
from gait_analysis_tool import GaitAnalysisTool

# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 보행 분석 도구 초기화
gait_tool = GaitAnalysisTool()

def process_frame(frame):
    # BGR을 RGB로 변환
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # MediaPipe Pose로 키포인트 추출
    results = pose.process(frame_rgb)
    
    if results.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        return landmarks
    return None

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks = process_frame(frame)
        if landmarks is not None:
            frame_data.append(landmarks)
    
    cap.release()
    return np.array(frame_data)

def main():
    st.title("보행 분석 웹 앱")

    mode = st.radio("분석 모드 선택", ("비디오 업로드", "실시간 스트리밍"))

    if mode == "비디오 업로드":
        uploaded_file = st.file_uploader("보행 비디오 업로드", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            video_data = analyze_video(tfile.name)
            results = gait_tool.analyze_video(video_data)

            st.subheader("분석 결과")
            for i, result in enumerate(results):
                st.write(f"세그먼트 {i+1}")
                st.write(f"케이던스: {result['cadence']:.2f} steps/min")
                st.write(f"보폭: {result['stride_length']:.2f} m")
                st.write(f"보행 속도: {result['walking_speed']:.2f} m/s")
                st.write(f"평균 힙 각도: {np.mean(result['hip_angle']):.2f} rad")
                st.write(f"평균 무릎 각도: {np.mean(result['knee_angle']):.2f} rad")
                st.write(f"평균 발목 각도: {np.mean(result['ankle_angle']):.2f} rad")
                st.write("---")

    elif mode == "실시간 스트리밍":
        st.write("카메라를 준비해주세요.")
        run = st.checkbox('실시간 분석 시작')
        FRAME_WINDOW = st.image([])
        
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.write("카메라를 찾을 수 없습니다.")
                break

            landmarks = process_frame(frame)
            if landmarks is not None:
                result = gait_tool.add_frame(landmarks)
                if result:
                    st.subheader("실시간 분석 결과")
                    st.write(f"케이던스: {result['cadence']:.2f} steps/min")
                    st.write(f"보폭: {result['stride_length']:.2f} m")
                    st.write(f"보행 속도: {result['walking_speed']:.2f} m/s")
                    st.write(f"평균 힙 각도: {result['hip_angle']:.2f} rad")
                    st.write(f"평균 무릎 각도: {result['knee_angle']:.2f} rad")
                    st.write(f"평균 발목 각도: {result['ankle_angle']:.2f} rad")

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

if __name__ == "__main__":
    main()
