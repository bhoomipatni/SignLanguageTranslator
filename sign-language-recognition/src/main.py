import sys
from pathlib import Path

parent_dir = Path(__file__).parent
sys.path.insert(0, str(parent_dir))

import config
Config = config.Config

def main():
    print("="*60)
    print("SIGN LANGUAGE TRANSLATOR")
    print("="*60)
    print("\nChoose an option:")
    print("1. Extract keypoints from videos")
    print("2. Train model")
    print("3. Run webcam detection")
    print("4. Full pipeline (extract + train + detect)")
    print("="*60)
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        from data.extract_keypoints import process_wlasl_dataset
        process_wlasl_dataset()
    
    elif choice == "2":
        from models.train import train_model
        train_model()
    
    elif choice == "3":
        from inference.webcam_detector import start_webcam_detection
        start_webcam_detection()
    
    elif choice == "4":
        print("\nRunning full pipeline...\n")
        
        # Step 1: Extract keypoints
        print("Step 1: Extracting keypoints...")
        from data.extract_keypoints import process_wlasl_dataset
        process_wlasl_dataset()
        
        # Step 2: Train model
        print("\nStep 2: Training model...")
        from models.train import train_model
        model = train_model()
        
        if model is not None:
            # Step 3: Run detection
            print("\nStep 3: Starting webcam detection...")
            from inference.webcam_detector import start_webcam_detection
            start_webcam_detection()
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    main()