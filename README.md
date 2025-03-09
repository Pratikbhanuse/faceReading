# Face Reading Web App

This web application performs real-time facial attribute detection using computer vision and deep learning models. The app can detect various attributes such as age, gender, and emotion from video or webcam feed. Users can either upload a video or start real-time analysis directly from their webcam.

The app uses:
- **OpenCV** for face detection.
- **DeepFace** for emotion analysis.
- **Pre-trained DNN models** for age and gender prediction.

## Features

- **Upload Video**: Upload a video file to analyze the faces in it for age, gender, and emotion.
- **Real-Time Webcam Analysis**: Stream your webcam feed for real-time facial analysis.
- **Age Prediction**: Predict the approximate age group of detected faces.
- **Gender Prediction**: Predict the gender of detected faces.
- **Emotion Analysis**: Detect the dominant emotion of the detected faces.

## Requirements

- Python 3.x
- Flask
- OpenCV
- DeepFace
- Werkzeug

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/pratikbhanuse/faceReading.git
   cd faceReading
