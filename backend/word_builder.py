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
        
        # TODO: Implement letter buffering and filtering logic
        pass
    
    def build_word(self):
        """Convert letter sequence to word"""
        # TODO: Apply letter sequence rules and spell correction
        pass
    
    def finalize_word(self):
        """Finalize current word and add to sentence"""
        # TODO: Add word to sentence and reset current word
        pass
    
    def get_current_sentence(self):
        """Get the current sentence being built"""
        return " ".join(self.sentence + [self.current_word]).strip()
    
    def reset_sentence(self):
        """Reset the entire sentence"""
        self.sentence = []
        self.current_word = ""
        self.letter_buffer.clear()