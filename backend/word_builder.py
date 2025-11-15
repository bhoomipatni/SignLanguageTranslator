"""
Word Builder Module
Converts individual letters to words and sentences
"""

import time
from collections import deque

class WordBuilder:
    def __init__(self, letter_hold_time=1.0, word_gap_time=2.0):
        """Initialize word building system"""
        self.letter_buffer = deque(maxlen=50)
        self.current_word = ""
        self.sentence = []
        self.letter_hold_time = letter_hold_time
        self.word_gap_time = word_gap_time
        self.last_letter_time = 0
        self.last_letter = None
        
    def add_letter_prediction(self, letter, confidence, timestamp=None):
        """Add a letter prediction to the buffer"""
        if timestamp is None:
            timestamp = time.time()
        
        # Add to buffer with metadata
        prediction = {
            'letter': letter.upper(),
            'confidence': confidence,
            'timestamp': timestamp
        }
        self.letter_buffer.append(prediction)
        
        # Check if this is the same letter as before
        if letter.upper() == self.last_letter:
            # Check if letter has been held long enough
            hold_duration = timestamp - self.last_letter_time
            if hold_duration >= self.letter_hold_time:
                # Letter confirmed, add to current word
                self.current_word += letter.upper()
                self.last_letter = None  # Reset to avoid duplicates
                print(f"Letter confirmed: {letter.upper()}")
        else:
            # New letter detected
            self.last_letter = letter.upper()
            self.last_letter_time = timestamp
        
        # Check for word gap (no predictions for word_gap_time)
        if timestamp - self.last_letter_time > self.word_gap_time and self.current_word:
            self.finalize_word()
    
    def build_word(self):
        """Convert letter sequence to word"""
        if not self.current_word:
            return ""
            
        # Apply basic spell correction for common ASL words
        word = self.current_word.upper()
        
        # Common ASL word corrections
        corrections = {
            "HLEP": "HELP",
            "THNK": "THINK", 
            "WATR": "WATER",
            "FUD": "FOOD",
            "PLZ": "PLEASE",
            "THX": "THANKS",
            "YS": "YES",
            "WTR": "WATER"
        }
        
        corrected = corrections.get(word, word)
        return corrected
    
    def finalize_word(self):
        """Finalize current word and add to sentence"""
        if self.current_word:
            word = self.build_word()
            self.sentence.append(word)
            print(f"Word finalized: {word}")
            self.current_word = ""
            self.last_letter = None
    
    def get_current_sentence(self):
        """Get the current sentence being built"""
        return " ".join(self.sentence + [self.current_word]).strip()
    
    def reset_sentence(self):
        """Reset the entire sentence"""
        self.sentence = []
        self.current_word = ""
        self.letter_buffer.clear()