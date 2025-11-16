import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import sys
import json
from collections import deque
from tensorflow import keras

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
import config
Config = config.Config

class SignLanguageDetector:
    def __init__(self, model_path, labels_path):
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = keras.models.load_model(model_path)
        
        # Load labels
        print(f"Loading labels from: {labels_path}")
        with open(labels_path, 'r') as f:
            self.labels_map = json.load(f)
        
        # Reverse mapping for predictions
        self.idx_to_word = {v: k for k, v in self.labels_map.items()}
        
        print(f"Loaded {len(self.labels_map)} signs: {list(self.labels_map.keys())}")
        
        # MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Rolling sequence buffer (continuously updated)
        self.sequence = deque(maxlen=Config.SEQUENCE_LENGTH)
        
        # For smoothing predictions
        self.prediction_history = deque(maxlen=10)
        
    def extract_keypoints(self, results):
        """Extract keypoints from MediaPipe results"""
        keypoints = np.zeros(126)
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx < 2:
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        keypoints[hand_idx * 63 + idx * 3] = landmark.x
                        keypoints[hand_idx * 63 + idx * 3 + 1] = landmark.y
                        keypoints[hand_idx * 63 + idx * 3 + 2] = landmark.z
        
        return keypoints
    
    def predict_sign(self):
        """Predict sign from current sequence"""
        if len(self.sequence) < Config.SEQUENCE_LENGTH:
            return None, 0.0
        
        # Convert sequence to numpy array and add batch dimension
        sequence_array = np.array(list(self.sequence))
        sequence_array = np.expand_dims(sequence_array, axis=0)
        
        # Predict
        prediction = self.model.predict(sequence_array, verbose=0)[0]
        predicted_idx = np.argmax(prediction)
        confidence = prediction[predicted_idx]
        
        predicted_word = self.idx_to_word.get(predicted_idx, "Unknown")
        
        return predicted_word, confidence
    
    def get_smoothed_prediction(self):
        """Get most common prediction from recent history"""
        if len(self.prediction_history) == 0:
            return None, 0.0
        
        # Count predictions
        pred_counts = {}
        total_confidence = {}
        
        for word, conf in self.prediction_history:
            if word:
                pred_counts[word] = pred_counts.get(word, 0) + 1
                total_confidence[word] = total_confidence.get(word, 0) + conf
        
        if not pred_counts:
            return None, 0.0
        
        # Get most common prediction
        best_word = max(pred_counts, key=pred_counts.get)
        avg_confidence = total_confidence[best_word] / pred_counts[best_word]
        
        # Only return if it appears in at least 30% of recent frames
        if pred_counts[best_word] / len(self.prediction_history) >= 0.3:
            return best_word, avg_confidence
        
        return None, 0.0
    
    def run(self):
    #Real-time continuous webcam sign detector (much simpler + faster).

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("\n==============================")
        print(" REAL-TIME SIGN DETECTOR")
        print("==============================")
        print("Press Q to quit.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )

                # Add keypoints to sliding window
                keypoints = self.extract_keypoints(results)
                self.sequence.append(keypoints)

                # Predict ONLY when window is full
                if len(self.sequence) == Config.SEQUENCE_LENGTH:
                    seq = np.array(self.sequence)[None, ...]  # shape (1, seq, kp)

                    preds = self.model.predict(seq, verbose=0)[0]
                    idx = np.argmax(preds)
                    conf = preds[idx]
                    word = self.idx_to_word.get(idx, "Unknown")

                    # Display prediction
                    cv2.putText(frame,
                                f"{word} ({conf:.0%})",
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (0, 255, 0),
                                3)

            else:
                # Clear the sequence if hands disappear
                self.sequence.clear()

            cv2.putText(frame, "Real-time mode", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Sign Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def start_webcam_detection(model_path=None, labels_path=None):
    """Start webcam detection with trained model"""
    if model_path is None:
        model_path = Path(Config.MODELS_PATH) / "best_model.h5"
    
    if labels_path is None:
        labels_path = Config.LABELS_PATH
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first: python src/models/train.py")
        return
    
    if not Path(labels_path).exists():
        print(f"Error: Labels not found at {labels_path}")
        print("Please extract keypoints first: python src/data/extract_keypoints.py")
        return
    
    detector = SignLanguageDetector(model_path, labels_path)
    detector.run()

if __name__ == "__main__":
    start_webcam_detection()