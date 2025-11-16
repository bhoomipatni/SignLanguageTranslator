"""
ASL Gesture Classifier
Classifies full ASL signs (words/phrases) — currently supports "Hello" and "Thank you"
"""

import numpy as np
import pickle
import os
from typing import Optional, Dict, Any
from utils.config import MODEL_PATHS

class ASLClassifier:
    def __init__(self):
        """Initialize ASL gesture classifier"""
        self.trained_phrases = {}
        self.model_type = "mock"
        
        # Your trained gesture classes
        self.classes = ["HELLO", "THANK YOU"]
        
        self.load_model()

    def load_model(self):
        """Load your trained phrase data"""
        try:
            if os.path.exists(MODEL_PATHS['asl_pickle']):
                with open(MODEL_PATHS['asl_pickle'], "rb") as f:
                    self.trained_phrases = pickle.load(f)
                    self.model_type = "phrase_data"
                    print(f"✅ Loaded phrase data: {list(self.trained_phrases.keys())}")
                    return True
            
            print("⚠️ No trained phrase data found — using mock classifier.")
            return False
        
        except Exception as e:
            print(f"⚠️ Failed to load phrase data: {e}")
            return False

    def predict_single_frame(self, landmarks) -> Optional[Dict[str, Any]]:
        """Predict gesture from single frame landmarks (for web interface)"""
        if landmarks is None or len(landmarks) == 0:
            return None
        
        try:
            if self.model_type == "phrase_data":
                # For web interface, we'll use a simple approach:
                # Compare single frame with first frame of each trained gesture
                return self._predict_single_frame_with_data(landmarks)
            else:
                return self._mock_prediction()
        except Exception as e:
            print(f"Single frame prediction error: {e}")
            return None
    
    def _predict_single_frame_with_data(self, landmarks):
        """Predict using single frame comparison with trained data"""
        if not self.trained_phrases:
            print("Debug: No trained phrases available")
            return None
        
        print(f"Debug: Starting classification with landmarks length: {len(landmarks)}")
        print(f"Debug: Available phrases: {list(self.trained_phrases.keys())}")
        
        best_phrase = None
        best_score = float('inf')
        confidence_threshold = 5000.0  # Much higher threshold based on observed scores
        
        # Compare with each trained phrase's first frame
        for phrase_name, samples_list in self.trained_phrases.items():
            print(f"Debug: Checking phrase '{phrase_name}' with {len(samples_list)} samples")
            for i, sample in enumerate(samples_list):
                if 'gesture_sequence' in sample and len(sample['gesture_sequence']) > 0:
                    # Get first frame from trained data
                    first_frame = sample['gesture_sequence'][0]
                    
                    if isinstance(first_frame, dict) and 'landmarks' in first_frame:
                        trained_landmarks = first_frame['landmarks']
                        print(f"Debug: Sample {i} trained landmarks length: {len(trained_landmarks)}")
                        
                        # Convert our MediaPipe landmarks to match trained format
                        # MediaPipe gives us 21 landmarks * 2 coordinates (x,y) = 42 values
                        # Trained data has 126 values = 2 hands * 21 landmarks * 3 coordinates
                        if len(landmarks) == 42:  # 21 landmarks × 2 coords
                            # Convert [x1,y1,x2,y2,...] to [x1,y1,0,x2,y2,0,...] for first hand
                            first_hand_landmarks = []
                            for j in range(0, len(landmarks), 2):
                                first_hand_landmarks.extend([landmarks[j], landmarks[j+1], 0.0])
                            
                            # Add zeros for second hand (63 values)
                            second_hand_landmarks = [0.0] * 63
                            
                            # Combine both hands: first hand (63) + second hand (63) = 126
                            extended_landmarks = first_hand_landmarks + second_hand_landmarks
                            
                            if len(extended_landmarks) == len(trained_landmarks):
                                score = np.linalg.norm(np.array(extended_landmarks) - np.array(trained_landmarks))
                                print(f"Debug: Phrase '{phrase_name}' sample {i} score: {score}")
                                if score < best_score:
                                    best_score = score
                                    best_phrase = phrase_name
                            else:
                                print(f"Debug: Extended landmark length mismatch: {len(extended_landmarks)} vs {len(trained_landmarks)}")
                        else:
                            print(f"Debug: Unexpected landmarks length: {len(landmarks)}, expected 42")
                    else:
                        print(f"Debug: Sample {i} frame format unexpected: {type(first_frame)}")
                else:
                    print(f"Debug: Sample {i} has no gesture_sequence or empty sequence")
        
        print(f"Debug: Best phrase: {best_phrase}, best score: {best_score}, threshold: {confidence_threshold}")
        
        # Return result if confidence is good enough
        if best_score < confidence_threshold:
            confidence = max(0.0, min(1.0, 1.0 - (best_score / confidence_threshold)))
            result = {
                "gesture": best_phrase,
                "confidence": confidence,
                "success": True
            }
            print(f"Debug: Returning successful prediction: {result}")
            return result
        
        print(f"Debug: Score {best_score} above threshold {confidence_threshold}, returning None")
        return None

    def predict(self, landmarks_sequence) -> Optional[Dict[str, Any]]:
        """Predict a phrase from a sequence of hand landmarks"""
        if landmarks_sequence is None or len(landmarks_sequence) == 0:
            return None
        
        try:
            if self.model_type == "phrase_data":
                return self._predict_with_phrase_data(landmarks_sequence)
            else:
                return self._mock_prediction()
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    def _predict_with_phrase_data(self, landmarks_sequence):
        """Prediction using your trained phrase data with similarity matching"""
        if not self.trained_phrases:
            return None
        
        best_phrase = None
        best_score = float('inf')
        confidence_threshold = 3.0  # Based on your training results
        
        # Compare with each trained phrase
        for phrase_name, samples_list in self.trained_phrases.items():
            # Your data format: phrase_name -> list of sample dictionaries
            for sample in samples_list:
                if 'gesture_sequence' in sample:
                    score = self._calculate_similarity(landmarks_sequence, sample['gesture_sequence'])
                    if score < best_score:
                        best_score = score
                        best_phrase = phrase_name
        
        # Return result if confidence is good enough
        if best_score < confidence_threshold:
            return {
                "gesture": best_phrase,
                "confidence": float(best_score),
                "success": True
            }
        
        return None
    
    def _calculate_similarity(self, current_sequence, trained_sequence):
        """Calculate similarity between gesture sequences (simple DTW-like)"""
        if len(current_sequence) == 0 or len(trained_sequence) == 0:
            return float('inf')
        
        min_length = min(len(current_sequence), len(trained_sequence))
        total_distance = 0
        
        for i in range(min_length):
            if len(current_sequence[i]) == len(trained_sequence[i]):
                frame_distance = np.linalg.norm(
                    np.array(current_sequence[i]) - np.array(trained_sequence[i])
                )
                total_distance += frame_distance
        
        return total_distance / min_length if min_length > 0 else float('inf')

    def _mock_prediction(self):
        """Mock classifier for testing without a trained model"""
        idx = np.random.choice(len(self.classes))
        return {
            "gesture": self.classes[idx],
            "confidence": float(np.random.uniform(0.7, 0.95)),
            "mock": True
        }
