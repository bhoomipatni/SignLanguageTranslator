import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from src.utils.mediapipe_utils import HandTracking
from src.inference.predictor import Predictor

class WebcamDetector:
    def __init__(self, model_path):
        self.cap = cv2.VideoCapture(0)
        self.hand_tracker = HandTracking()
        self.predictor = Predictor(model_path)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            keypoints = self.hand_tracker.get_hand_keypoints(frame)
            if keypoints is not None:
                gesture = self.predictor.predict(keypoints)
                self.display_gesture(frame, gesture)

            cv2.imshow('Webcam Feed', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def display_gesture(self, frame, gesture):
        cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

if __name__ == "__main__":
    model_path = 'models/saved_models/your_model.h5'  # Update with your model path
    detector = WebcamDetector(model_path)
    detector.run()