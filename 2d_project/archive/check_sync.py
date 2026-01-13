import cv2

v1 = "/data/gait/data/1/1-1.mp4"
v2 = "/data/gait/data/1/1-2.mp4"

def check(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frames / fps if fps > 0 else 0
    print(f"{path}: FPS={fps:.2f}, Frames={frames}, Duration={duration:.2f}s")

check(v1)
check(v2)
