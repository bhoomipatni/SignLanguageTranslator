// SignSpeak AI JavaScript - unified camera, UI and ASL processing

document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const startStop = document.getElementById('startStop');
    const speakBtn = document.getElementById('speak-btn');
    const resetBtn = document.getElementById('reset-btn');
    const textOutput = document.getElementById('textOutput');
    
    let stream = null;
    let running = false;
    let processingInterval = null;

    async function startStream() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
            video.srcObject = stream;
            running = true;
            startStop.textContent = 'Stop';
            
            // Start ASL processing
            processingInterval = setInterval(processCurrentFrame, 1000); // Process every second
            updateTextOutput('ASL Recognition Active - Show your gestures!');
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
        
        // Stop ASL processing
        if (processingInterval) {
            clearInterval(processingInterval);
            processingInterval = null;
        }
        updateTextOutput('Camera stopped');
    }

    async function processCurrentFrame() {
        if (!video.videoWidth || !video.videoHeight || !running) {
            return;
        }
        
        try {
            // Create a canvas to capture the current video frame
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            // Draw the current video frame to the canvas
            ctx.drawImage(video, 0, 0);
            
            // Convert canvas to base64 image data
            const imageData = canvas.toDataURL('image/jpeg', 0.8);
            
            // Send frame to backend for processing
            const response = await fetch('/process_frame', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });
            
            const result = await response.json();
            
            if (result.success && result.gesture && result.gesture !== 'Unknown') {
                const confidence = (result.confidence * 100).toFixed(1);
                updateTextOutput(`Detected: ${result.gesture} (${confidence}% confident)`);
            }
        } catch (error) {
            console.error('Error processing frame:', error);
        }
    }

    function updateTextOutput(message) {
        textOutput.innerHTML = `<p style="margin:0; color:var(--text-dark);">${message}</p>`;
    }

    startStop.addEventListener('click', () => {
        if (running) stopStream(); else startStream();
    });

    // Initialize text output placeholder
    textOutput.innerHTML = '<p style="margin:0; color:var(--text-dark); opacity:0.6;">Start camera to begin ASL recognition...</p>';

    resetBtn.addEventListener('click', function() {
        if (running) {
            updateTextOutput('ASL Recognition Active - Show your gestures!');
        } else {
            textOutput.innerHTML = '<p style="margin:0; color:var(--text-dark); opacity:0.6;">Start camera to begin ASL recognition...</p>';
        }
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
});