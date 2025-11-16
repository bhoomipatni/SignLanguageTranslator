"""
Test hand tracking functionality to debug hand detection issues
"""
import cv2
import sys
sys.path.append('.')

from backend.hand_tracking import HandTracker

def test_hand_tracking():
    print('Testing hand tracking...')
    hand_tracker = HandTracker()

    # Test camera setup
    cap = hand_tracker.setup_camera(0)
    if cap.isOpened():
        print('✅ Camera opened successfully')
        
        print('Press any key while showing your hand to test detection...')
        frame_count = 0
        
        while frame_count < 10:  # Test for 10 frames
            ret, frame = cap.read()
            if ret:
                print(f'Frame {frame_count + 1}: {frame.shape}')
                
                # Test hand detection on the frame
                landmarks = hand_tracker.detect_hands(frame)
                if landmarks is not None:
                    print(f'✅ Hands detected: {len(landmarks)} hands')
                    for i, hand_landmarks in enumerate(landmarks):
                        print(f'  Hand {i+1}: {len(hand_landmarks.landmark)} landmarks')
                else:
                    print('⚠️ No hands detected in frame')
                
                # Show the frame with any detected landmarks
                annotated_frame = frame.copy()
                if landmarks is not None:
                    annotated_frame = hand_tracker.draw_landmarks_on_image(annotated_frame, landmarks)
                
                cv2.imshow('Hand Detection Test', annotated_frame)
                
                # Wait for key press or short delay
                key = cv2.waitKey(1000) & 0xFF  # 1 second delay
                if key != 255:  # If any key pressed
                    break
                    
                frame_count += 1
            else:
                print('❌ Failed to capture frame')
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print('Test complete!')
    else:
        print('❌ Failed to open camera')

if __name__ == "__main__":
    test_hand_tracking()