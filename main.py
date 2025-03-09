from flask import Flask, render_template, request, Response
import cv2
import numpy as np
from deepface import DeepFace
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')

age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

        # Predict emotion using DeepFace
        try:
            emotion_analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            emotion = emotion_analysis[0]['dominant_emotion']
        except:
            emotion = "Unknown"

        # Draw label
        label = f"{gender}, {age}, {emotion}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return frame

@app.route('/')
def home():
    return render_template('upload.html')  # This page gives options to upload or start webcam analysis

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        return Response(process_video(file_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    return "Invalid file format", 400

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_attributes(frame)  # Process the frame with face detection, age, gender, and emotion

        # Convert frame to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Yield the image as part of a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed', methods=['GET'])
def video_feed():
    return Response(generate_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_video_feed():
    cap = cv2.VideoCapture(0)  # Using webcam for live video feed
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_attributes(frame)  # Process the frame with face detection, age, gender, and emotion

        # Convert frame to JPEG
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Yield the image as part of a multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
