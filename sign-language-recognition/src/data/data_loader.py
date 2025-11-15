class DataLoader:
    def __init__(self, keypoints_path, labels_path):
        self.keypoints_path = keypoints_path
        self.labels_path = labels_path
        self.keypoints_data = self.load_keypoints()
        self.labels_data = self.load_labels()

    def load_keypoints(self):
        import os
        import numpy as np
        keypoints_files = [f for f in os.listdir(self.keypoints_path) if f.endswith('.npy')]
        keypoints = []
        for file in keypoints_files:
            data = np.load(os.path.join(self.keypoints_path, file), allow_pickle=True)
            keypoints.append(data)
        return keypoints

    def load_labels(self):
        import json
        with open(self.labels_path, 'r') as f:
            labels = json.load(f)
        return labels

    def get_data(self):
        return self.keypoints_data, self.labels_data

    def __len__(self):
        return len(self.keypoints_data)