import cv2
import numpy as np
from data.data_loader import DataLoader
from models.train import train_model
from inference.webcam_detector import start_webcam_detection
from config import Config

def main():
    # Load the data
    data_loader = DataLoader(Config.DATA_PATH, Config.LABELS_PATH)
    keypoints, labels = data_loader.load_data()

    # Train the model
    model = train_model(keypoints, labels)

    # Start webcam detection for real-time sign language recognition
    start_webcam_detection(model)

if __name__ == "__main__":
    main()