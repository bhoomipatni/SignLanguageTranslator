from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from backend.pipeline import ASLPipeline

# Configure Flask to use the frontend folders for static files and templates
app = Flask(__name__, static_folder='frontend/static', template_folder='frontend/templates')

# Initialize the ASL pipeline
pipeline = ASLPipeline()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Get the image data from the request
        data = request.json
        image_data = data['image']
        
        # Decode the base64 image
        header, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process the frame through the ASL pipeline
        result = pipeline.process_frame(frame)
        
        return jsonify({
            'success': True,
            'gesture': result['gesture'],
            'confidence': result['confidence'],
            'translation': result['translation']
        })
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)