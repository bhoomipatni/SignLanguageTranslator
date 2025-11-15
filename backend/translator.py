"""
Translation Module using Gemini API with Service Account Support
"""

import requests
import json
import os
from typing import Optional, Dict, Any
try:
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False

class GeminiTranslator:
    def __init__(self):
        """Initialize Gemini API translator with service account support"""
        self.service_account_path = "service_account.json"
        # Use the working Gemini 2.5 Flash model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        
        # Try to get access token from service account
        self.access_token = None
        if GOOGLE_AUTH_AVAILABLE and os.path.exists(self.service_account_path):
            self.access_token = self._get_access_token_from_service_account()
        
        print(f"🤖 Gemini initialized: {'✅ Service Account' if self.access_token else '❌ No Auth'}")
        
        # ASL grammar patterns for improvement
        self.asl_patterns = {
            "ME": "I",
            "YOU": "you",
            "HE": "he", 
            "SHE": "she",
            "THEY": "they",
            "WE": "we",
            "WANT": "want",
            "NEED": "need",
            "LIKE": "like", 
            "LOVE": "love",
            "FOOD": "food",
            "WATER": "water",
            "HELP": "help",
            "PLEASE": "please",
            "THANK": "thank you",
            "SORRY": "sorry",
            "YES": "yes",
            "NO": "no",
            "HUNGRY": "hungry",
            "THIRSTY": "thirsty",
            "TIRED": "tired",
            "HOW": "how",
            "WHAT": "what",
            "WHERE": "where",
            "GOOD": "good",
            "BAD": "bad",
            "NICE": "nice",
            "MEET": "meet",
            "BATHROOM": "bathroom"
        }
        
    def _get_access_token_from_service_account(self) -> Optional[str]:
        """Get access token using service account credentials"""
        try:
            import google.auth
            from google.auth.transport.requests import Request
            
            # Load credentials from service account file
            credentials, project = google.auth.load_credentials_from_file(
                self.service_account_path,
                scopes=['https://www.googleapis.com/auth/generative-language']
            )
            
            # Refresh credentials to get access token
            credentials.refresh(Request())
            return credentials.token
            
        except Exception as e:
            print(f"Service account authentication failed: {e}")
            return None
        
    def translate_text(self, text: str, target_language: str = "es") -> Optional[str]:
        """Translate text to target language using Gemini"""
        if not self.access_token and not self.api_key:
            print("Warning: No Gemini authentication available")
            return self._fallback_translation(text, target_language)
            
        try:
            prompt = f"Translate this English text to {target_language}: '{text}'. Return only the translation."
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }
            
            response = self._make_api_request(payload)
            if response and "candidates" in response:
                translation = response["candidates"][0]["content"]["parts"][0]["text"].strip()
                return translation
            else:
                return self._fallback_translation(text, target_language)
                
        except Exception as e:
            print(f"Translation error: {e}")
            return self._fallback_translation(text, target_language)
    
    def improve_sentence(self, asl_sentence: str) -> Optional[str]:
        """Use Gemini to improve ASL-to-English sentence structure"""
        if not asl_sentence or not asl_sentence.strip():
            return ""
            
        # First, apply basic ASL-to-English pattern replacements
        improved = self._apply_asl_patterns(asl_sentence)
        
        if not self.api_key:
            print("Warning: No Gemini API key found, using pattern matching only")
            return improved
            
        try:
            prompt = f"""
Convert this ASL (American Sign Language) sentence to natural English grammar. 
ASL often uses different word order and lacks articles/auxiliary verbs.

ASL sentence: "{asl_sentence}"
Basic conversion: "{improved}"

Return only the improved English sentence with proper grammar, articles, and natural word order. Keep it conversational and natural.
"""
            
            payload = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }
            
            response = self._make_api_request(payload)
            if response and "candidates" in response:
                result = response["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Remove quotes if present
                result = result.strip('"\'')
                return result
            else:
                return improved
                
        except Exception as e:
            print(f"Sentence improvement error: {e}")
            return improved
    
    def _apply_asl_patterns(self, asl_sentence: str) -> str:
        """Apply basic ASL to English pattern matching"""
        words = asl_sentence.upper().split()
        improved_words = []
        
        for word in words:
            if word in self.asl_patterns:
                improved_words.append(self.asl_patterns[word])
            else:
                improved_words.append(word.lower())
                
        # Basic sentence structure improvements
        sentence = " ".join(improved_words)
        
        # Capitalize first letter
        if sentence:
            sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            
        # Add period if missing
        if sentence and not sentence.endswith(('.', '!', '?')):
            sentence += "."
            
        return sentence
    
    def _fallback_translation(self, text: str, target_language: str) -> str:
        """Simple fallback translations without API"""
        fallback_translations = {
            "es": {  # Spanish
                "hello": "hola",
                "thank you": "gracias",
                "please": "por favor",
                "yes": "sí",
                "no": "no",
                "help": "ayuda",
                "water": "agua",
                "food": "comida",
                "I am hungry": "Tengo hambre",
                "I need help": "Necesito ayuda"
            },
            "fr": {  # French
                "hello": "bonjour",
                "thank you": "merci",
                "please": "s'il vous plaît",
                "yes": "oui",
                "no": "non",
                "help": "aide",
                "water": "eau",
                "food": "nourriture"
            }
        }
        
        text_lower = text.lower()
        if target_language in fallback_translations:
            return fallback_translations[target_language].get(text_lower, f"[{target_language.upper()}] {text}")
        else:
            return f"[{target_language.upper()}] {text}"
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Return list of supported translation languages"""
        return {
            "es": "Spanish",
            "fr": "French", 
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ru": "Russian",
            "ar": "Arabic"
        }
    
    def _make_api_request(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make API request to Gemini using service account or API key"""
        try:
            # Use access token if available (service account)
            if self.access_token:
                url = self.base_url.replace('?key=', '')  # Remove key parameter
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.access_token}"
                }
            elif self.api_key:
                # Fallback to API key
                url = f"{self.base_url}?key={self.api_key}"
                headers = {
                    "Content-Type": "application/json"
                }
            else:
                print("No authentication method available (no service account or API key)")
                return None
            
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Gemini API error {response.status_code}: {response.text}")
                # If service account token expired, try to refresh
                if response.status_code == 401 and self.access_token:
                    print("Attempting to refresh access token...")
                    self.access_token = self._get_access_token_from_service_account()
                    if self.access_token:
                        # Retry with new token
                        headers["Authorization"] = f"Bearer {self.access_token}"
                        response = requests.post(url, json=payload, headers=headers, timeout=10)
                        if response.status_code == 200:
                            return response.json()
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None