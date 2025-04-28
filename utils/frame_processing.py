import cv2
import numpy as np
import os

def extract_video_frames(video_path, time_steps=15, w=224, h=224):
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    while success and len(frames) < time_steps:
        frame = cv2.resize(frame, (w, h))
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
        success, frame = cap.read()

    cap.release()

    if len(frames) < time_steps:
        padding = [np.zeros((h, w, 3), dtype=np.float32)] * (time_steps - len(frames))
        frames.extend(padding)

    return np.array(frames[:time_steps])
