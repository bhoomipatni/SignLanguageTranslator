"""
SignSpeak AI - Main Entry Point
Real-time ASL translation system for hackathon
"""

import streamlit as st
from frontend.ui import main_interface

def main():
    """Main entry point for SignSpeak AI application"""
    st.set_page_config(
        page_title="SignSpeak AI",
        page_icon="🤟",
        layout="wide"
    )
    
    main_interface()

if __name__ == "__main__":
    main()