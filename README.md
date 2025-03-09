# Face Attribute Detection with Streamlit

This project uses **Streamlit** to create a real-time face attribute detection application. The app detects the following attributes of faces in a video:

- **Age**
- **Gender**
- **Emotion**

It uses pre-trained models for age and gender detection, and the **DeepFace** library to analyze emotions.

## Features

- **Real-time face detection** using OpenCV.
- **Age and gender prediction** using pre-trained models.
- **Emotion analysis** using DeepFace.

## Requirements

To run the app, you need the following Python packages:

- Streamlit
- OpenCV
- DeepFace
- NumPy

You can install the required dependencies using the following command:

```bash
pip install -r requirements.txt
