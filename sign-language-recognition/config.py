import os

class Config:
    def __init__(self):
        # Data paths
        self.RAW_DATA_PATH = os.path.join('data', 'raw', 'wlasl')
        self.PROCESSED_DATA_PATH = os.path.join('data', 'processed', 'keypoints')
        self.LABELS_PATH = os.path.join('data', 'labels.json')
        
        # Model paths
        self.MODEL_SAVE_PATH = os.path.join('models', 'saved_models')
        
        # Training parameters
        self.BATCH_SIZE = 32
        self.EPOCHS = 50
        self.LEARNING_RATE = 0.001
        
        # Model parameters
        self.INPUT_SHAPE = (30, 21, 3)  # Example shape for keypoints
        self.NUM_CLASSES = 100  # Adjust based on the number of gestures in your dataset
        
        # MediaPipe settings
        self.HAND_CONNECTIONS = True  # Set to True to visualize hand connections
        self.MIN_DETECTION_CONFIDENCE = 0.5
        self.MIN_TRACKING_CONFIDENCE = 0.5

config = Config()