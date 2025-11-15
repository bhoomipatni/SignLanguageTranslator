"""
ASL Classifier Module
Loads and manages ASL classification models
"""

import numpy as np
import pickle
import os
from typing import Optional, Dict, Any
from utils.config import MODEL_PATHS
# import tensorflow as tf  # For CNN model

class ASLClassifier:
    def __init__(self, model_path: Optional[str] = None):
        """Initialize ASL classifier"""
        self.model = None
        self.model_path = model_path
        self.classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.model_type = "mock"  # Will be 'pickle', 'tensorflow', or 'mock'
        
        # Load model if available
        self.load_model()
        
    def load_model(self, model_type: str = "pickle") -> bool:
        """Load pre-trained ASL model"""
        try:
            # Try to load pickle model first
            if os.path.exists(MODEL_PATHS['asl_pickle']):
                with open(MODEL_PATHS['asl_pickle'], 'rb') as f:
                    self.model = pickle.load(f)
                    self.model_type = "pickle"
                    print("Loaded pickle ASL model")
                    return True
                    
            # Try to load TensorFlow model
            elif os.path.exists(MODEL_PATHS['asl_cnn']):
                # Uncomment when you have tensorflow model
                # import tensorflow as tf
                # self.model = tf.keras.models.load_model(MODEL_PATHS['asl_cnn'])
                # self.model_type = "tensorflow"
                # print("Loaded TensorFlow ASL model")
                # return True
                pass
                
            else:
                print("No pre-trained model found, using mock classifier")
                self.model_type = "mock"
                return True
                
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_type = "mock"
            return False
    
    def predict(self, landmarks: np.ndarray) -> Optional[Dict[str, Any]]:
        """Predict ASL letter from hand landmarks"""
        if landmarks is None or len(landmarks) != 42:
            return None
            
        try:
            if self.model_type == "pickle" and self.model is not None:
                return self._predict_with_pickle(landmarks)
            elif self.model_type == "tensorflow" and self.model is not None:
                return self._predict_with_tensorflow(landmarks)
            else:
                return self._mock_prediction(landmarks)
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return self._mock_prediction(landmarks)
    
    def _predict_with_pickle(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """Make prediction using scikit-learn model"""
        # Reshape for sklearn (expects 2D input)
        input_data = landmarks.reshape(1, -1)
        
        # Get prediction and probability
        prediction = self.model.predict(input_data)[0]
        probabilities = self.model.predict_proba(input_data)[0]
        
        # Get confidence score
        confidence = np.max(probabilities)
        
        return {
            "letter": self.classes[prediction],
            "confidence": float(confidence),
            "probabilities": {self.classes[i]: float(prob) for i, prob in enumerate(probabilities)}
        }
    
    def _predict_with_tensorflow(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """Make prediction using TensorFlow model"""
        # Reshape for TensorFlow
        input_data = landmarks.reshape(1, 42, 1)  # Adjust shape as needed
        
        # Make prediction
        predictions = self.model.predict(input_data, verbose=0)[0]
        
        # Get best prediction
        predicted_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_idx])
        
        return {
            "letter": self.classes[predicted_idx],
            "confidence": confidence,
            "probabilities": {self.classes[i]: float(prob) for i, prob in enumerate(predictions)}
        }
    
    def _mock_prediction(self, landmarks: np.ndarray) -> Dict[str, Any]:
        """Mock prediction for testing without real model"""
        # Simple mock: use hand position to determine letter
        landmarks_2d = landmarks.reshape(21, 2)
        
        # Use fingertip positions for mock classification
        thumb_tip = landmarks_2d[4]
        index_tip = landmarks_2d[8]
        middle_tip = landmarks_2d[12]
        ring_tip = landmarks_2d[16]
        pinky_tip = landmarks_2d[20]
        
        # Simple heuristics for common letters
        # This is just for demonstration - real model would be much more sophisticated
        
        # Calculate finger extensions (relative to palm center)
        palm_center = np.mean(landmarks_2d[[0, 5, 9, 13, 17]], axis=0)
        
        finger_distances = {\n            'thumb': np.linalg.norm(thumb_tip - palm_center),\n            'index': np.linalg.norm(index_tip - palm_center),\n            'middle': np.linalg.norm(middle_tip - palm_center),\n            'ring': np.linalg.norm(ring_tip - palm_center),\n            'pinky': np.linalg.norm(pinky_tip - palm_center)\n        }\n        \n        # Mock classification based on simple rules\n        extended_fingers = [finger for finger, dist in finger_distances.items() if dist > 50]\n        \n        if len(extended_fingers) == 1:\n            if 'index' in extended_fingers:\n                letter = 'I'\n            elif 'thumb' in extended_fingers:\n                letter = 'A'\n            else:\n                letter = 'L'\n        elif len(extended_fingers) == 2:\n            if 'index' in extended_fingers and 'middle' in extended_fingers:\n                letter = 'V'\n            else:\n                letter = 'U'\n        elif len(extended_fingers) >= 4:\n            letter = 'B'\n        else:\n            # Random letter for other cases\n            letter = np.random.choice(['H', 'E', 'L', 'O', 'W', 'R', 'D'])\n        \n        # Add some randomness to confidence\n        confidence = np.random.uniform(0.7, 0.95)\n        \n        # Create mock probabilities\n        probabilities = np.random.uniform(0.01, 0.1, 26)\n        letter_idx = self.classes.index(letter)\n        probabilities[letter_idx] = confidence\n        probabilities = probabilities / np.sum(probabilities)  # Normalize\n        \n        return {\n            \"letter\": letter,\n            \"confidence\": float(confidence),\n            \"probabilities\": {self.classes[i]: float(prob) for i, prob in enumerate(probabilities)},\n            \"mock\": True\n        }\n    \n    def predict_batch(self, landmark_batch: np.ndarray) -> List[Dict[str, Any]]:\n        \"\"\"Predict multiple ASL letters at once\"\"\"\n        results = []\n        for landmarks in landmark_batch:\n            prediction = self.predict(landmarks)\n            if prediction:\n                results.append(prediction)\n        return results\n    \n    def get_confidence_threshold(self) -> float:\n        \"\"\"Return minimum confidence threshold for predictions\"\"\"\n        return 0.7  # Default threshold\n    \n    def get_model_info(self) -> Dict[str, Any]:\n        \"\"\"Get information about the loaded model\"\"\"\n        return {\n            \"model_type\": self.model_type,\n            \"model_loaded\": self.model is not None,\n            \"classes\": len(self.classes),\n            \"confidence_threshold\": self.get_confidence_threshold()\n        }