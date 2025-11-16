import numpy as np
import json
from pathlib import Path
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DataLoader:
    def __init__(self, processed_data_path, labels_path):
        self.processed_data_path = Path(processed_data_path)
        self.labels_path = Path(labels_path)
        
    def load_data(self):
        """Load processed keypoints and labels"""
        # Load index
        index_path = self.processed_data_path / "index.json"
        
        if not index_path.exists():
            print(f"Error: index.json not found at {index_path}")
            print("Please run extract_keypoints.py first")
            return None, None, None
        
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        if len(index_data) == 0:
            print("Error: index.json is empty")
            return None, None, None
        
        # Build labels from index data
        print(f"Building labels from {len(index_data)} entries...")
        
        # Extract unique words and create labels map
        words_to_labels = {}
        current_label = 0
        
        for entry in index_data:
            word = entry.get('word', 'unknown')
            if word not in words_to_labels:
                words_to_labels[word] = current_label
                current_label += 1
        
        # Save updated labels
        self.labels_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.labels_path, 'w') as f:
            json.dump(words_to_labels, f, indent=2)
        
        print(f"Found {len(words_to_labels)} unique words: {list(words_to_labels.keys())}")
        
        sequences = []
        labels = []
        
        print(f"Loading {len(index_data)} sequences...")
        
        for entry in index_data:
            keypoints_file = entry['keypoints_file']
            word = entry.get('word', 'unknown')
            label = words_to_labels[word]
            
            # Check if file exists
            if not Path(keypoints_file).exists():
                print(f"Warning: Keypoints file not found: {keypoints_file}")
                continue
            
            # Load keypoints
            keypoints = np.load(keypoints_file)
            
            if len(keypoints) > 0:
                sequences.append(keypoints)
                labels.append(label)
        
        if len(sequences) == 0:
            print("Error: No valid sequences loaded")
            return None, None, None
        
        # Pad sequences to same length
        max_length = max(len(seq) for seq in sequences)
        print(f"Padding sequences to max length: {max_length}")
        X = pad_sequences(sequences, maxlen=max_length, dtype='float32', padding='post')
        
        # Convert labels to categorical
        y = to_categorical(labels, num_classes=len(words_to_labels))
        
        print(f"Loaded {len(X)} sequences")
        print(f"Sequence shape: {X.shape}")
        print(f"Number of classes: {len(words_to_labels)}")
        
        return X, y, words_to_labels