#!/usr/bin/env python3

import pickle
import os

# Load the trained model data
model_path = "models/asl_model.pkl"

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    
    print("=== TRAINED MODEL DATA ANALYSIS ===")
    print(f"Data type: {type(data)}")
    print(f"Keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
    
    for gesture, samples in data.items():
        print(f"\n--- GESTURE: {gesture} ---")
        print(f"Number of samples: {len(samples)}")
        
        for i, sample in enumerate(samples):
            print(f"\nSample {i}:")
            print(f"  Keys: {list(sample.keys()) if isinstance(sample, dict) else 'Not a dict'}")
            
            if 'gesture_sequence' in sample:
                seq = sample['gesture_sequence']
                print(f"  Sequence length: {len(seq)}")
                if len(seq) > 0:
                    first_frame = seq[0]
                    print(f"  First frame type: {type(first_frame)}")
                    print(f"  First frame shape: {len(first_frame) if hasattr(first_frame, '__len__') else 'Not a sequence'}")
                    print(f"  First frame data: {first_frame}")
                    
                    if len(seq) > 1:
                        print(f"  Second frame: {seq[1]}")
else:
    print(f"Model file not found: {model_path}")