import streamlit as st
import cv2
import tempfile
import torch
import os
import pathlib
import re
from imutils import paths
import numpy as np
from PIL import Image



from super_gradients.training import models
from super_gradients.common.object_names import Models


# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

st.title("Pose Estimation with Streamlit and MediaPipe")

uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

yolo_nas_pose = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")

def make_prediction(input_file, action, confidence=0.55):
    """
    Make a prediction using the fixed model and device, and either show or save the result.

    Args:
    - input_file (str): Path to the input file.
    - action (str): Either 'show' or 'save'.
    - confidence (float, optional): Confidence threshold. Defaults to 0.75.

    Returns:
    - None

    Raises:
    - ValueError: If the action is not 'show' or 'save'.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(input_file.read())

    
    if action == "show":
        return yolo_nas_pose.to(device).predict(input_file, conf=confidence)
    elif action == "save":
        output_file = pathlib.Path(input_file).stem + "-detections" + pathlib.Path(input_file).suffix
        yolo_nas_pose.to(device).predict(input_file, conf=confidence).save(output_file)
        print(f"Prediction saved to {output_file}")
        return output_file
    else:
        raise ValueError("Action must be either 'show' or 'save'.")
    
def process_video(video_file):
    # Temp file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
        tmpfile.write(video_file.read())
        cap = cv2.VideoCapture(tmpfile.name)

    # Width and height of the Streamlit column for displaying videos
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Process video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = []
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect pose
        results = pose.process(frame)

        # Draw the pose annotations on the frame
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        processed_frames.append(frame)

    cap.release()
    return width, height, processed_frames


if uploaded_file is not None:
    # Convert the uploaded file into a video stream
    # width, height, processed_frames = process_video(uploaded_file)

    # # Convert frames to video
    # out_file = 'output.mp4'

    # out = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(
    #     *'mp4v'), 20, (width, height))
    # for frame in processed_frames:
    #     out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # out.release()
    out_file = make_prediction(uploaded_file, 'save')
    # Display the processed video
    st.video(out_file)
