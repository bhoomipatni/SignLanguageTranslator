// SignSpeak AI JavaScript - unified camera, UI and simple speak/reset handlers

document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const startStop = document.getElementById('startStop');
    const speakBtn = document.getElementById('speak-btn');
    const resetBtn = document.getElementById('reset-btn');
    const textOutput = document.getElementById('textOutput');
    
    let stream = null;
    let running = false;

    async function startStream() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
            video.srcObject = stream;
            running = true;
            startStop.textContent = 'Stop';
        } catch (err) {
            alert('Camera access denied or not available: ' + err.message);
        }
    }

    function stopStream() {
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
            video.srcObject = null;
        }
        running = false;
        startStop.textContent = 'Start';
    }

    startStop.addEventListener('click', () => {
        if (running) stopStream(); else startStream();
    });

    // Initialize text output placeholder
    textOutput.innerHTML = '<p style="margin:0; color:var(--text-dark); opacity:0.6;">Output will appear here...</p>';

    resetBtn.addEventListener('click', function() {
        textOutput.innerHTML = '<p style="margin:0; color:var(--text-dark); opacity:0.6;">Output will appear here...</p>';
        console.log('Reset button clicked');
    });

    speakBtn.addEventListener('click', function() {
        const text = textOutput.innerText || textOutput.textContent || 'Hello';
        if ('speechSynthesis' in window) {
            const utter = new SpeechSynthesisUtterance(text);
            window.speechSynthesis.speak(utter);
        } else {
            console.warn('Speech Synthesis not supported in this browser.');
        }
    });

    // TODO: Add WebSocket connection for real-time predictions and pipeline integration
    // For example: open a ws to the server, capture frames to canvas, and send base64 frames.
});