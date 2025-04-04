import cv2
import numpy as np
import os
from flask import Flask, render_template, request, Response, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from datetime import datetime
import threading
import time
import platform
import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the deepfake detection model
model = None
try:
    model_path = 'deepfake_detector.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        print("Model file not found. Simulation mode enabled.")
except Exception as e:
    print(f"Error loading model: {str(e)}")

# Face detection models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables
frame_buffer = []
BUFFER_SIZE = 15
history = []
MAX_HISTORY = 20
recording = False
record_frames = []
last_prediction = None
last_confidence = None

# System info
def get_system_info():
    info = {
        "System": platform.system(),
        "Processor": platform.processor() or "Unknown",
        "RAM": f"{round(psutil.virtual_memory().total / (1024**3), 1)} GB",
        "CPU Cores": psutil.cpu_count(logical=False),
        "Threads": psutil.cpu_count(logical=True),
        "GPU": "None" if not GPUtil else GPUtil.getGPUs()[0].name if GPUtil.getGPUs() else "None"
    }
    return info

system_info = get_system_info()

@app.route('/')
def index():
    return render_template('index.html', system_info=system_info)

@app.route('/upload', methods=['POST'])
def upload_video():
    global frame_buffer, recording, record_frames, last_prediction, last_confidence
    if 'video' not in request.files:
        return jsonify({'status': 'No file uploaded'})
    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'No file selected'})
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    frame_buffer = []
    record_frames = []
    recording = False
    result, confidence = process_video(filepath)
    last_prediction = result
    last_confidence = confidence
    add_to_history(result, confidence, file.filename)
    return render_template('result.html', result=result, confidence=confidence, filename=file.filename, history=history)

def process_video(video_path):
    if not model:
        return "FAKE (Simulated)", 0.85
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Error: Could not open video", 0.0
    
    frames = []
    frame_count = 0
    while frame_count < BUFFER_SIZE and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.copy())
        frame_count += 1
    cap.release()
    
    if not frames:
        return "Error: No frames extracted", 0.0
    
    return predict_sequence(frames)

def predict_sequence(frames):
    try:
        processed_frames = [preprocess_input(cv2.resize(f, (128, 128))) for f in frames]
        sequence = np.array(processed_frames)[np.newaxis, ...]
        prediction = model.predict(sequence, verbose=0)[0][0]
        threshold = 0.7  # Default sensitivity
        if prediction > threshold:
            return "FAKE", prediction
        else:
            return "AUTHENTIC", 1 - prediction
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return "Error", 0.0

def add_to_history(result, confidence, source):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history.append((result, confidence, source, timestamp))
    if len(history) > MAX_HISTORY:
        history.pop(0)

@app.route('/system_status')
def system_status():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    gpu = "--" if not GPUtil else f"{GPUtil.getGPUs()[0].load*100:.1f}" if GPUtil.getGPUs() else "--"
    return jsonify({'cpu': cpu, 'gpu': gpu, 'mem': mem})

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if __name__ == '__main__':
    app.run(debug=True, threaded=True)