import os
import numpy as np
import cv2
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels = json.load(f)
    return labels

def normalize_keypoints(keypoints):
    keypoints = np.array(keypoints)
    keypoints = (keypoints - np.mean(keypoints, axis=0)) / np.std(keypoints, axis=0)
    return keypoints

def augment_data(keypoints):
    # Example augmentation: random rotation
    angle = np.random.uniform(-15, 15)
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
    augmented_keypoints = cv2.warpAffine(keypoints, rotation_matrix, (keypoints.shape[1], keypoints.shape[0]))
    return augmented_keypoints

def preprocess_data(raw_data_dir, label_file, test_size=0.2):
    labels = load_labels(label_file)
    all_keypoints = []
    all_labels = []

    for video_file in os.listdir(raw_data_dir):
        # Assume each video file corresponds to a label in the labels dictionary
        label = labels.get(video_file)
        if label:
            keypoints = extract_keypoints_from_video(os.path.join(raw_data_dir, video_file))
            normalized_keypoints = normalize_keypoints(keypoints)
            augmented_keypoints = augment_data(normalized_keypoints)

            all_keypoints.append(augmented_keypoints)
            all_labels.append(label)

    X_train, X_test, y_train, y_test = train_test_split(all_keypoints, all_labels, test_size=test_size, random_state=42)
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

def extract_keypoints_from_video(video_path):
    # Placeholder function to extract keypoints from a video
    # This should call the keypoint extraction logic from extract_keypoints.py
    return []  # Replace with actual keypoint extraction logic