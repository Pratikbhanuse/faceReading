import cv2
import numpy as np
from keras.models import load_model
from deepface import DeepFace

# Load pre-trained models for age and gender detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

def detect_attributes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender = gender_list[gender_net.forward().argmax()]

        # Predict age
        age_net.setInput(blob)
        age = age_list[age_net.forward().argmax()]

        # Using DeepFace to predict emotion
        try:
            # Analyze the face with DeepFace for emotion detection
            emotion_analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            emotion = emotion_analysis[0]['dominant_emotion']
        except Exception as e:
            print(f"Error during emotion detection: {e}")
            emotion = "Unknown"

        # Create label for the face (Gender, Age, Emotion)
        label = f'{gender}, {age}, {emotion}'
        label_position = (x, y - 10) if y - 10 > 10 else (x, y + h + 10)  # Ensure text is visible
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the label
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return frame


# Open webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = detect_attributes(frame)
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
