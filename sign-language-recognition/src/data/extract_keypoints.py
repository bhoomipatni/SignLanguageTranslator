import cv2
import mediapipe as mp
import os
import json
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

def extract_keypoints_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    keypoints = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_keypoints = []
            for landmark in hand_landmarks.landmark:
                hand_keypoints.append((landmark.x, landmark.y, landmark.z))
            keypoints.append(hand_keypoints)

    return keypoints

def process_video(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    keypoints_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = extract_keypoints_from_frame(frame)
        keypoints_data.append(keypoints)
        frame_count += 1

    cap.release()

    output_file = os.path.join(output_dir, os.path.basename(video_path).replace('.mp4', '.json'))
    with open(output_file, 'w') as f:
        json.dump(keypoints_data, f)

def extract_keypoints_from_directory(video_dir, output_dir):
    for filename in os.listdir(video_dir):
        if filename.endswith('.mp4'):
            video_path = os.path.join(video_dir, filename)
            process_video(video_path, output_dir)