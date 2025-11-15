from tensorflow.keras.models import load_model
import cv2
import numpy as np
import json
from src.utils.mediapipe_utils import HandTracking
from src.utils.visualization import visualize_gesture

class Predictor:
    def __init__(self, model_path, labels_path):
        self.model = load_model(model_path)
        with open(labels_path, 'r') as f:
            self.labels = json.load(f)
        self.hand_tracker = HandTracking()

    def predict(self, frame):
        keypoints = self.hand_tracker.get_hand_keypoints(frame)
        if keypoints is not None:
            keypoints = np.array(keypoints).flatten()
            keypoints = np.expand_dims(keypoints, axis=0)
            prediction = self.model.predict(keypoints)
            predicted_class = np.argmax(prediction, axis=1)[0]
            return self.labels[str(predicted_class)]
        return None

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gesture = self.predict(frame)
            visualize_gesture(frame, gesture)

            cv2.imshow('Sign Language Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()