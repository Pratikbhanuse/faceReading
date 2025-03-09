import cv2
import numpy as np
import streamlit as st
from deepface import DeepFace
import tempfile

# Radio button option for real-time vs upload video
option = st.sidebar.radio("Choose Input", ("Upload Video", "Use Real-Time Webcam"))

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Function for face attribute detection
def detect_attributes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender = gender_list[gender_net.forward().argmax()]

        # Predict age
        age_net.setInput(blob)
        age = age_list[age_net.forward().argmax()]

        # Predict emotion using DeepFace
        result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        label = f'{gender}, {age}, {emotion}'
        label_position = (x, y-10) if y-10 > 10 else (x, y + h + 10)  # Ensure text is visible
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the label
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return frame

# Analyze real-time footage from webcam
if option == "Use Real-Time Webcam":
    cap = cv2.VideoCapture(0)  # 0 is the default camera index
    stframe = st.empty()  # Empty frame for updating the video in Streamlit

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform face attribute detection
        frame = detect_attributes(frame)

        # Convert the frame to RGB (Streamlit expects RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in the Streamlit app
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

# Upload a video file for processing (default)
else:
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

    if video_file is not None:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(video_file.read())
            cap = cv2.VideoCapture(tmp_file.name)

        stframe = st.empty()  # Empty frame for updating the video in Streamlit

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform face attribute detection
            frame = detect_attributes(frame)

            # Convert the frame to RGB (Streamlit expects RGB images)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame in the Streamlit app
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        cap.release()
    else:
        st.write("Please upload a video file.")
