"""
ElevenLabs Agent Speech Synthesis Module
Integrated pipeline: MediaPipe → Gemini → ElevenLabs Agent
"""

import io
import base64
import requests
import pygame
import time
from typing import Optional, Dict, Any
from utils.config import get_elevenlabs_api_key, get_google_tts_config

class ElevenLabsAgent:
    """ElevenLabs Agent for intelligent voice responses"""
    
    def __init__(self, api_key: str, agent_id: Optional[str] = None):
        self.api_key = api_key
        self.agent_id = agent_id
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "Accept": "application/json",
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
        
    def create_conversation(self, signature_id: str) -> str:
        """Create a new conversation with the agent"""
        url = f"{self.base_url}/convai/conversations"
        payload = {"agent_id": signature_id}
        
        response = requests.post(url, json=payload, headers=self.headers)
        if response.status_code == 200:
            return response.json()["conversation_id"]
        else:
            raise Exception(f"Failed to create conversation: {response.text}")
    
    def send_message_to_agent(self, conversation_id: str, message: str) -> Dict[str, Any]:
        """Send translated ASL text to ElevenLabs agent"""
        url = f"{self.base_url}/convai/conversations/{conversation_id}/add_message"
        payload = {"message": message}
        
        response = requests.post(url, json=payload, headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to send message: {response.text}")
    
    def get_agent_response_audio(self, conversation_id: str) -> bytes:
        """Get audio response from agent"""
        url = f"{self.base_url}/convai/conversations/{conversation_id}/audio"
        
        response = requests.get(url, headers=self.headers)
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Failed to get audio: {response.text}")

class SpeechSynthesizer:
    """Enhanced Speech Synthesizer with ElevenLabs Agent Integration"""
    
    def __init__(self, preferred_service="elevenlabs_agent"):
        """Initialize speech synthesis with agent support"""
        self.preferred_service = preferred_service
        self.elevenlabs_api_key = get_elevenlabs_api_key()
        self.agent = None
        self.conversation_id = None
        self.voice_settings = {
            "stability": 0.5,
            "similarity_boost": 0.5,
            "style": 0.0,
            "use_speaker_boost": True
        }
        
        # Initialize pygame mixer for audio playback
        try:
            pygame.mixer.init()
        except:
            print("Warning: Could not initialize audio playback")
    
    def setup_agent(self, agent_id: str) -> None:
        """Setup ElevenLabs agent for conversation"""
        if not self.elevenlabs_api_key:
            raise ValueError("ElevenLabs API key not found")
        
        self.agent = ElevenLabsAgent(self.elevenlabs_api_key, agent_id)
        self.conversation_id = self.agent.create_conversation(agent_id)
        print(f"Agent setup complete. Conversation ID: {self.conversation_id}")
    
    def process_asl_translation(self, translated_text: str, language: str = "en") -> Dict[str, Any]:
        """Process ASL translation through ElevenLabs agent pipeline"""
        if self.preferred_service == "elevenlabs_agent" and self.agent:
            return self._agent_response(translated_text)
        elif self.preferred_service == "elevenlabs_tts":
            return self._elevenlabs_tts(translated_text)
        else:
            return self._google_tts(translated_text, language)
    
    def _agent_response(self, text: str) -> Dict[str, Any]:
        """Generate intelligent response using ElevenLabs agent"""
        try:
            # Send translated ASL text to agent
            message_response = self.agent.send_message_to_agent(self.conversation_id, text)
            
            # Get audio response
            audio_data = self.agent.get_agent_response_audio(self.conversation_id)
            
            # Play audio
            self._play_audio_bytes(audio_data)
            
            return {
                "success": True,
                "message_response": message_response,
                "audio_length": len(audio_data),
                "service": "elevenlabs_agent"
            }
        except Exception as e:
            print(f"Agent response error: {e}")
            # Fallback to regular TTS
            return self._elevenlabs_tts(text)
    
    def _elevenlabs_tts(self, text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> Dict[str, Any]:
        """Generate speech using ElevenLabs TTS API"""
        try:
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            
            payload = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": self.voice_settings
            }
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.elevenlabs_api_key
            }
            
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                self._play_audio_bytes(response.content)
                return {
                    "success": True,
                    "audio_length": len(response.content),
                    "service": "elevenlabs_tts"
                }
            else:
                raise Exception(f"TTS API error: {response.text}")
                
        except Exception as e:
            print(f"ElevenLabs TTS error: {e}")
            return self._google_tts(text)
    
    def _google_tts(self, text: str, language: str = "en") -> Dict[str, Any]:
        """Fallback Google TTS implementation"""
        try:
            from google.cloud import texttospeech
            
            client = texttospeech.TextToSpeechClient()
            
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(
                language_code=language,
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            
            response = client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            
            self._play_audio_bytes(response.audio_content)
            return {
                "success": True,
                "audio_length": len(response.audio_content),
                "service": "google_tts"
            }
            
        except Exception as e:
            print(f"Google TTS error: {e}")
            return {"success": False, "error": str(e)}
    
    def _play_audio_bytes(self, audio_data: bytes) -> None:
        """Play audio from bytes using pygame"""
        try:
            # Save to temporary file and play
            temp_file = "temp_audio.mp3"
            with open(temp_file, "wb") as f:
                f.write(audio_data)
            
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Audio playback error: {e}")
    
    def get_available_voices(self) -> Dict[str, Any]:
        """Get available voices from ElevenLabs"""
        try:
            url = "https://api.elevenlabs.io/v1/voices"
            headers = {
                "Accept": "application/json",
                "xi-api-key": self.elevenlabs_api_key
            }
            
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": "Failed to fetch voices"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def set_voice_settings(self, stability: float = 0.5, similarity: float = 0.5, style: float = 0.0):
        """Configure voice settings for ElevenLabs"""
        self.voice_settings = {
            "stability": stability,
            "similarity_boost": similarity,
            "style": style,
            "use_speaker_boost": True
        }