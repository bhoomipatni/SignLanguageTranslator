"""
Complete ASL Translation Pipeline
MediaPipe → Gemini → ElevenLabs Agent Integration
"""


import os
import cv2
import time
from typing import Dict, Any, Optional
from backend.hand_tracking import HandTracker
from backend.classifier import ASLClassifier
from backend.word_builder import WordBuilder
from backend.translator import GeminiTranslator
from backend.speech import SpeechSynthesizer, ElevenLabsAgent
from utils.config import PREDICTION_CONFIG

class ASLTranslationPipeline:
    """Complete pipeline for ASL translation with ElevenLabs agent"""
    
    def __init__(self, agent_id: Optional[str] = None):
        """Initialize the complete translation pipeline"""
        # Initialize all components
        self.hand_tracker = HandTracker()
        self.classifier = ASLClassifier()
        self.word_builder = WordBuilder()
        self.translator = GeminiTranslator()
        self.speech_synthesizer = SpeechSynthesizer(preferred_service="elevenlabs_agent")
        
        # Setup ElevenLabs agent if provided
        if agent_id:
            self.speech_synthesizer.setup_agent(agent_id)
        
        # Pipeline state
        self.is_running = False
        self.current_sentence = ""
        self.last_translation = ""
        
    def start_pipeline(self):
        """Start the complete ASL translation pipeline"""
        self.is_running = True
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("🤟 ASL Translation Pipeline Started!")
        print("Pipeline: MediaPipe → Classifier → Word Builder → Gemini → ElevenLabs Agent")
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Step 1: MediaPipe hand tracking
                landmarks = self.hand_tracker.detect_hands(frame)
                
                if landmarks is not None:
                    # Step 2: ASL classification
                    prediction = self.classifier.predict(landmarks)
                    
                    if prediction and prediction['confidence'] > PREDICTION_CONFIG['confidence_threshold']:
                        # Step 3: Word building
                        self.word_builder.add_letter_prediction(
                            prediction['letter'], 
                            prediction['confidence'],
                            time.time()
                        )
                        
                        # Check if sentence is updated
                        current_sentence = self.word_builder.get_current_sentence()
                        
                        if current_sentence != self.current_sentence and len(current_sentence.strip()) > 0:
                            self.current_sentence = current_sentence
                            
                            # Step 4: Translation with Gemini
                            translation = self.translator.improve_sentence(self.current_sentence)
                            
                            if translation and translation != self.last_translation:
                                self.last_translation = translation
                                
                                # Step 5: ElevenLabs Agent Response
                                response = self.speech_synthesizer.process_asl_translation(translation)
                                
                                print(f"ASL: {self.current_sentence}")
                                print(f"Translation: {translation}")
                                print(f"Speech: {response['service']} - {response.get('success', False)}")
                                print("-" * 50)
                
                # Display frame with landmarks
                if landmarks is not None:
                    frame = self.hand_tracker.draw_landmarks_on_image(frame, landmarks)
                
                # Show current sentence on frame
                cv2.putText(frame, f"Sentence: {self.current_sentence}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Translation: {self.last_translation}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                cv2.imshow('ASL Translation Pipeline', frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\\nPipeline stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.is_running = False
    
    def process_single_translation(self, asl_text: str) -> Dict[str, Any]:
        """Process a single ASL text through the pipeline"""
        try:
            # Step 1: Improve sentence with Gemini
            improved_text = self.translator.improve_sentence(asl_text)
            
            # Step 2: Generate speech with ElevenLabs agent
            speech_response = self.speech_synthesizer.process_asl_translation(improved_text)
            
            return {
                "original": asl_text,
                "improved": improved_text,
                "speech": speech_response,
                "success": True
            }
            
        except Exception as e:
            return {
                "original": asl_text,
                "error": str(e),
                "success": False
            }
    
    def reset_conversation(self):
        """Reset the conversation and sentence building"""
        self.word_builder.reset_sentence()
        self.current_sentence = ""
        self.last_translation = ""
        print("Conversation reset!")
    
    def stop_pipeline(self):
        """Stop the translation pipeline"""
        self.is_running = False

# Example usage and testing
if __name__ == "__main__":
    # Initialize pipeline with your ElevenLabs agent ID
    pipeline = ASLTranslationPipeline(agent_id=os.getenv('ELEVENLABS_AGENT_ID'))
    
    # Test single translation
    test_result = pipeline.process_single_translation("HELLO HOW ARE YOU")
    print("Test Result:", test_result)
    
    # Start live pipeline (uncomment to run)
    # pipeline.start_pipeline()