"""
Hand Tracking Module using MediaPipe and OpenCV
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple

class HandTracker:
    def __init__(self):
        """Initialize MediaPipe hand tracking pipeline"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def setup_camera(self, device_id: int = 0) -> cv2.VideoCapture:
        """Setup camera for hand tracking"""
        cap = cv2.VideoCapture(device_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap
    
    def detect_hands(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect hands in the given frame"""
        if frame is None:
            return None
            
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Extract landmarks if hands are detected
        if results.multi_hand_landmarks:
            # Use the first hand detected
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = self.extract_landmarks(hand_landmarks, frame.shape)
            return landmarks
        
        return None
    
    def extract_landmarks(self, hand_landmarks, frame_shape) -> np.ndarray:
        """Extract hand landmarks from detection results"""
        height, width, _ = frame_shape
        landmarks = []
        
        # Extract x, y coordinates for all 21 landmarks
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks.extend([x, y])
        
        # Convert to numpy array (42 features: 21 landmarks * 2 coordinates)
        return np.array(landmarks)
    
    def preprocess_for_model(self, landmarks: np.ndarray) -> np.ndarray:
        """Preprocess landmarks for ASL classification"""
        if landmarks is None or len(landmarks) != 42:
            return None
            
        # Reshape to (21, 2) for easier processing
        landmarks_2d = landmarks.reshape(21, 2)
        
        # Normalize landmarks relative to wrist position (landmark 0)
        wrist = landmarks_2d[0]
        normalized = landmarks_2d - wrist
        
        # Scale by hand size (distance from wrist to middle finger tip)
        middle_finger_tip = landmarks_2d[12]
        hand_size = np.linalg.norm(middle_finger_tip - wrist)
        
        if hand_size > 0:
            normalized = normalized / hand_size
        
        # Flatten back to 1D array
        return normalized.flatten()
    
    def draw_landmarks_on_image(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Draw hand landmarks on image for visualization"""
        if landmarks is None:
            return image
            
        # Convert landmarks back to MediaPipe format for drawing
        landmarks_2d = landmarks.reshape(21, 2)
        
        # Draw landmarks
        for i, (x, y) in enumerate(landmarks_2d):
            # Draw landmark points
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
            
            # Draw landmark numbers for debugging
            cv2.putText(image, str(i), (int(x)+5, int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        # Draw hand connections
        self._draw_hand_connections(image, landmarks_2d)
        
        return image
    
    def _draw_hand_connections(self, image: np.ndarray, landmarks_2d: np.ndarray):
        """Draw connections between hand landmarks"""
        # Define hand connections (MediaPipe hand model)
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger  
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks_2d) and end_idx < len(landmarks_2d):
                start_point = (int(landmarks_2d[start_idx][0]), int(landmarks_2d[start_idx][1]))
                end_point = (int(landmarks_2d[end_idx][0]), int(landmarks_2d[end_idx][1]))
                cv2.line(image, start_point, end_point, (255, 0, 0), 2)
    
    def get_hand_bbox(self, landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        """Calculate bounding box around hand landmarks"""
        if landmarks is None or len(landmarks) < 42:
            return (0, 0, 0, 0)
            
        landmarks_2d = landmarks.reshape(21, 2)
        x_coords = landmarks_2d[:, 0]
        y_coords = landmarks_2d[:, 1]
        
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
        
        width = x_max - x_min
        height = y_max - y_min
        
        return (x_min, y_min, width, height)