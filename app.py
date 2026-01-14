import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import joblib
import tempfile
import os
import sys
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Add current directory to path to import local modules
sys.path.append('/data/gait')
from mediapipe_csv_processor import MediaPipeCSVProcessor

# Page Config
st.set_page_config(
    page_title="Pathological Gait Classifier",
    page_icon="🚶",
    layout="wide"
)

# Load Model
@st.cache_resource
def load_model():
    model_path = '/data/gait/gait_classifier.joblib'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

artifacts = load_model()

def process_video_to_csv(video_path):
    """Run MediaPipe Pose on video and save to CSV"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    landmarks_data = []
    
    # Header
    header = ['frame', 'time']
    for i in range(33):
        header.extend([f'x{i}', f'y{i}', f'z{i}', f'vis{i}'])
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        if results.pose_landmarks:
            row = [frame_count, frame_count / fps]
            for lm in results.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])
            landmarks_data.append(row)
            
        frame_count += 1
        
    cap.release()
    
    # Save to temp CSV
    df = pd.DataFrame(landmarks_data, columns=header)
    temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name
    df.to_csv(temp_csv, index=False)
    return temp_csv

def extract_features(csv_path):
    """Use existing processor to extract features"""
    processor = MediaPipeCSVProcessor(csv_path)
    processor.process()
    
    # Check if processing was successful
    if not processor.gait_cycles:
        return None
        
    # Get features using the same logic as gavd_feature_extractor
    # We need to aggregate cycle data
    
    # 1. Spatio-temporal
    params = processor.spatio_temporal_params
    if not params:
        return None
        
    # 2. Kinematics (Mean of cycles)
    # We need to calculate summary stats from processor.gait_cycles
    # This part requires accessing the internal data structure of processor
    # Let's reconstruct the feature dictionary
    
    features = {}
    
    # Spatio-temporal (Mean)
    features['cadence_spm'] = np.mean(params['cadence'])
    features['stride_time_s'] = np.mean(params['stride_time'])
    features['velocity_mps'] = np.mean(params['velocity'])
    features['step_length_m'] = np.mean(params['step_length'])
    features['stride_length_m'] = np.mean(params['stride_length'])
    
    # Kinematics
    joints = ['hip', 'knee', 'ankle']
    for joint in joints:
        roms = []
        means = []
        stds = []
        
        for cycle in processor.gait_cycles:
            if joint in cycle:
                data = cycle[joint].values
                roms.append(data.max() - data.min())
                means.append(data.mean())
                stds.append(data.std())
        
        features[f'{joint}_rom'] = np.mean(roms) if roms else 0
        features[f'{joint}_mean'] = np.mean(means) if means else 0
        features[f'{joint}_std'] = np.mean(stds) if stds else 0
        
    return features

# UI Layout
st.title("🚶 Pathological Gait Analysis System")
st.markdown("Upload a walking video to detect pathological gait patterns using AI.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Video (MP4, AVI)", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Save uploaded file
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        st.video(video_path)
        
        if st.button("Analyze Gait", type="primary"):
            if artifacts is None:
                st.error("Model not found! Please run training script first.")
            else:
                with st.spinner("Processing video... (This may take a moment)"):
                    try:
                        # 1. Video -> CSV
                        csv_path = process_video_to_csv(video_path)
                        
                        # 2. CSV -> Features
                        features = extract_features(csv_path)
                        
                        if features:
                            # 3. Predict
                            pipeline = artifacts['pipeline']
                            feature_cols = artifacts['features']
                            le = artifacts['label_encoder']
                            
                            # Create DataFrame for prediction
                            input_df = pd.DataFrame([features])
                            # Ensure columns order
                            input_df = input_df[feature_cols]
                            
                            # Predict
                            pred_idx = pipeline.predict(input_df)[0]
                            pred_prob = pipeline.predict_proba(input_df)[0]
                            pred_label = le.inverse_transform([pred_idx])[0]
                            
                            # Display Results in Col2
                            with col2:
                                st.success("Analysis Complete!")
                                
                                # Diagnosis Card
                                st.subheader("Diagnosis Result")
                                if pred_label == 'Normal':
                                    st.info(f"### {pred_label}")
                                else:
                                    st.error(f"### {pred_label}")
                                
                                # Probability Bar
                                st.write("Confidence Scores:")
                                prob_df = pd.DataFrame({
                                    'Class': le.classes_,
                                    'Probability': pred_prob
                                })
                                st.bar_chart(prob_df.set_index('Class'))
                                
                                # Feature Radar Chart
                                st.subheader("Gait Profile")
                                
                                # Normalize features for radar chart (simple scaling)
                                categories = ['Velocity', 'Stride Length', 'Cadence', 'Hip ROM', 'Knee ROM', 'Ankle ROM']
                                values = [
                                    features['velocity_mps'], features['stride_length_m'], features['cadence_spm']/100,
                                    features['hip_rom']/50, features['knee_rom']/70, features['ankle_rom']/40
                                ]
                                
                                fig = go.Figure(data=go.Scatterpolar(
                                    r=values,
                                    theta=categories,
                                    fill='toself',
                                    name='Patient'
                                ))
                                
                                fig.update_layout(
                                    polar=dict(
                                        radialaxis=dict(
                                            visible=True,
                                            range=[0, 2]
                                        )),
                                    showlegend=False
                                )
                                st.plotly_chart(fig)
                                
                                # Detailed Metrics
                                st.subheader("Detailed Metrics")
                                st.json(features)
                                
                        else:
                            st.error("Could not extract gait cycles. Please ensure the full body is visible and the subject is walking sideways.")
                            
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())
