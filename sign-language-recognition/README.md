# Sign Language Recognition Project

This project aims to recognize sign language gestures using a combination of TensorFlow, Keras, and MediaPipe. The model is trained on the WLASL (Word-Level American Sign Language) video database, which provides a rich dataset for recognizing various sign language gestures.

## Project Structure

```
sign-language-recognition
├── src
│   ├── data
│   │   ├── preprocess.py          # Functions for preprocessing raw data
│   │   ├── extract_keypoints.py    # Functions to extract hand keypoints using MediaPipe
│   │   └── data_loader.py          # Data loader for processed keypoints and labels
│   ├── models
│   │   ├── lstm_model.py           # LSTM model architecture for gesture recognition
│   │   ├── train.py                 # Training loop for the LSTM model
│   │   └── evaluate.py              # Model evaluation functions
│   ├── inference
│   │   ├── webcam_detector.py       # Live webcam feed setup for real-time recognition
│   │   └── predictor.py             # Prediction logic for classifying gestures
│   ├── utils
│   │   ├── mediapipe_utils.py       # Utility functions for MediaPipe
│   │   └── visualization.py          # Functions for visualizing keypoints and gestures
│   └── main.py                      # Entry point for the application
├── data
│   ├── raw
│   │   └── wlasl                    # Raw video files from the WLASL dataset
│   ├── processed
│   │   └── keypoints                # Processed keypoints extracted from videos
│   └── labels.json                  # Mapping of gestures to labels
├── models
│   └── saved_models                  # Directory for saving trained models
├── requirements.txt                  # Required Python packages
├── config.py                         # Configuration settings
└── README.md                         # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd sign-language-recognition
   ```

2. **Install the required packages**:
   Create a virtual environment and install the dependencies listed in `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```

3. **Download the WLASL dataset**:
   Place the raw video files in the `data/raw/wlasl` directory.

4. **Preprocess the data**:
   Run the preprocessing script to extract keypoints and prepare the dataset for training:
   ```
   python src/data/preprocess.py
   ```

5. **Train the model**:
   Execute the training script to train the LSTM model:
   ```
   python src/models/train.py
   ```

6. **Run inference**:
   To start recognizing gestures using your webcam, run:
   ```
   python src/inference/webcam_detector.py
   ```

## Usage

- The model will process the live webcam feed and display the recognized gestures in real-time.
- You can modify the configuration settings in `config.py` to adjust paths and model parameters as needed.

## Acknowledgments

This project utilizes the WLASL dataset and the MediaPipe library for hand tracking and keypoint extraction. Special thanks to the contributors of these resources.