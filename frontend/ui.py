"""
Streamlit UI for SignSpeak AI with ElevenLabs Agent Integration
"""

import streamlit as st
import cv2
import numpy as np
import os
from backend.pipeline import ASLTranslationPipeline
from backend.hand_tracking import HandTracker
from backend.classifier import ASLClassifier
from backend.word_builder import WordBuilder
from backend.translator import GeminiTranslator
from backend.speech import SpeechSynthesizer

def main_interface():
    """Main Streamlit interface for SignSpeak AI"""
    st.title("🤟 SignSpeak AI")
    st.subheader("Real-time ASL Translation with ElevenLabs Agent")
    
    # Initialize session state
    setup_session_state()
    
    # Sidebar controls
    st.sidebar.header("🎛️ Pipeline Settings")
    
    # ElevenLabs Agent Configuration
    st.sidebar.subheader("ElevenLabs Agent")
    agent_id = st.sidebar.text_input("Agent ID", 
                                    value=os.getenv('ELEVENLABS_AGENT_ID', ''),
                                    help="Your ElevenLabs agent ID")
    
    if st.sidebar.button("🔗 Setup Agent"):
        if agent_id:
            try:
                st.session_state.pipeline = ASLTranslationPipeline(agent_id=agent_id)
                st.sidebar.success("Agent connected!")
            except Exception as e:
                st.sidebar.error(f"Agent setup failed: {e}")
        else:
            st.sidebar.error("Please enter an Agent ID")
    
    # Translation settings
    translation_language = st.sidebar.selectbox(
        "Translation Language",
        ["Spanish", "French", "German", "Italian", "Portuguese", "Japanese"]
    )
    
    voice_enabled = st.sidebar.checkbox("Enable Voice Output", value=True)
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 1.0, 0.7)
    
    # Voice settings
    st.sidebar.subheader("Voice Settings")
    stability = st.sidebar.slider("Voice Stability", 0.0, 1.0, 0.5)
    similarity = st.sidebar.slider("Voice Similarity", 0.0, 1.0, 0.5)
    style = st.sidebar.slider("Voice Style", 0.0, 1.0, 0.0)
    
    # Main interface columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📹 Live ASL Detection")
        camera_placeholder = st.empty()
        
        # Pipeline controls
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            if st.button("🎥 Start Pipeline"):
                if hasattr(st.session_state, 'pipeline'):
                    st.info("Pipeline started! Check your camera window.")
                    # Note: In production, you'd run this in a separate thread
                else:
                    st.error("Please setup the agent first!")
        
        with col1b:
            if st.button("⏹️ Stop Pipeline"):
                if hasattr(st.session_state, 'pipeline'):
                    st.session_state.pipeline.stop_pipeline()
                    st.success("Pipeline stopped!")
        
        with col1c:
            if st.button("🔄 Reset Conversation"):
                if hasattr(st.session_state, 'pipeline'):
                    st.session_state.pipeline.reset_conversation()
                    st.success("Conversation reset!")
        
    with col2:
        st.header("📝 Translation Pipeline")
        
        # Current sentence display
        st.subheader("Current ASL Sentence")
        sentence_display = st.empty()
        sentence_display.text("ASL sentence will appear here...")
        
        # Translation display
        st.subheader("Improved Translation")
        translation_display = st.empty()
        translation_display.text("Translation will appear here...")
        
        # Test translation
        st.subheader("🧪 Test Translation")
        test_input = st.text_input("Enter ASL text to test:", "HELLO HOW ARE YOU")
        
        if st.button("🎤 Test Agent Response"):
            if hasattr(st.session_state, 'pipeline') and test_input:
                with st.spinner("Processing through pipeline..."):
                    result = st.session_state.pipeline.process_single_translation(test_input)
                    
                    if result['success']:
                        st.success("Translation successful!")
                        st.write(f"**Original:** {result['original']}")
                        st.write(f"**Improved:** {result['improved']}")
                        st.write(f"**Speech Service:** {result['speech']['service']}")
                    else:
                        st.error(f"Translation failed: {result['error']}")
            else:
                st.error("Please setup agent and enter text!")
        
        # Pipeline status
        st.subheader("🔍 Pipeline Status")
        status_container = st.container()
        
        with status_container:
            st.write("**Components:**")
            st.write("✅ MediaPipe Hand Tracking")
            st.write("✅ ASL Classifier")
            st.write("✅ Word Builder")
            st.write("✅ Gemini Translation")
            
            agent_status = "✅ ElevenLabs Agent" if hasattr(st.session_state, 'pipeline') else "⏸️ ElevenLabs Agent (Setup Required)"
            st.write(agent_status)
    
    # Pipeline flow diagram
    st.header("🔄 Translation Pipeline Flow")
    col_flow1, col_flow2, col_flow3, col_flow4, col_flow5 = st.columns(5)
    
    with col_flow1:
        st.markdown("### 📷\\nMediaPipe\\n*Hand Detection*")
    
    with col_flow2:
        st.markdown("### 🧠\\nClassifier\\n*Letter Recognition*")
    
    with col_flow3:
        st.markdown("### 📝\\nWord Builder\\n*Sentence Formation*")
        
    with col_flow4:
        st.markdown("### 🌐\\nGemini API\\n*Translation*")
        
    with col_flow5:
        st.markdown("### 🎙️\\nElevenLabs Agent\\n*Voice Response*")
    
    # API Keys status
    st.header("🔑 API Configuration")
    col_api1, col_api2 = st.columns(2)
    
    with col_api1:
        elevenlabs_key = "✅ Configured" if os.getenv('ELEVENLABS_API_KEY') else "❌ Missing"
        st.write(f"**ElevenLabs API:** {elevenlabs_key}")
        
    with col_api2:
        gemini_key = "✅ Configured" if os.getenv('GEMINI_API_KEY') else "❌ Missing"
        st.write(f"**Gemini API:** {gemini_key}")
    
    if not os.getenv('ELEVENLABS_API_KEY') or not os.getenv('GEMINI_API_KEY'):
        st.warning("⚠️ Please configure your API keys in the .env file!")

def setup_session_state():
    """Initialize Streamlit session state"""
    if 'hand_tracker' not in st.session_state:
        st.session_state.hand_tracker = HandTracker()
    
    if 'classifier' not in st.session_state:
        st.session_state.classifier = ASLClassifier()
    
    if 'word_builder' not in st.session_state:
        st.session_state.word_builder = WordBuilder()
    
    if 'translator' not in st.session_state:
        st.session_state.translator = GeminiTranslator()
    
    if 'speech' not in st.session_state:
        st.session_state.speech = SpeechSynthesizer()