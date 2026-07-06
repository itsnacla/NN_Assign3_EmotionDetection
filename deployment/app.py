import os
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify

# Prevent tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# Define Flask application
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Set upload limits (max 10MB)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Set paths
DEPLOYMENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DEPLOYMENT_DIR)
MODEL_PATH = os.path.join(REPO_ROOT, 'src', 'models', 'model.h5')

# Emotion dictionary corresponding to the model output classes
EMOTION_DICT = {
    0: "Angry", 
    1: "Disgusted", 
    2: "Fearful", 
    3: "Happy", 
    4: "Neutral", 
    5: "Sad", 
    6: "Surprised"
}

# Global references for model and face cascade
model = None
face_cascade = None

def get_face_cascade():
    global face_cascade
    if face_cascade is None:
        # Load OpenCV Haar cascade from cv2's internal data directory
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar Cascade xml not found at {cascade_path}")
        face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade

def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Keras model.h5 not found at {MODEL_PATH}. Make sure to train the model first.")
        # Load the pre-trained Keras model
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

def predict_emotions_in_image(image_cv):
    """
    Process OpenCV image, detect faces, predict emotions, and return structured result.
    """
    try:
        cascade = get_face_cascade()
        nn_model = get_model()
    except Exception as e:
        print(f"Error loading dependencies: {e}")
        return {"faces": [], "dominant": None, "scores": None, "error": str(e)}

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces_detected = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    faces_list = []
    dominant_face = None
    max_area = 0

    for (x, y, w, h) in faces_detected:
        # Extract face Region of Interest (ROI) and resize to 48x48
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        # Preprocess: scale 1/255.0 and reshape to (1, 48, 48, 1)
        roi_gray = roi_gray.astype('float32') / 255.0
        cropped_img = np.expand_dims(np.expand_dims(roi_gray, -1), 0)
        
        # Predict emotion
        preds = nn_model(cropped_img, training=False).numpy()[0]
        max_idx = int(np.argmax(preds))
        dominant_emotion = EMOTION_DICT[max_idx]
        confidence = float(preds[max_idx])
        
        face_data = {
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "emotion": dominant_emotion,
            "confidence": confidence,
            "all_scores": {EMOTION_DICT[i]: float(preds[i]) for i in range(len(EMOTION_DICT))}
        }
        faces_list = append_face(faces_list, face_data)
        
        # Determine dominant face based on size (largest bounding box area)
        area = w * h
        if area > max_area:
            max_area = area
            dominant_face = face_data

    # Remove full scores from individual faces to keep output clean, keep only dominant scores
    faces_response = []
    for f in faces_list:
        faces_response.append({
            "x": f["x"],
            "y": f["y"],
            "w": f["w"],
            "h": f["h"],
            "emotion": f["emotion"],
            "confidence": f["confidence"]
        })

    if dominant_face:
        return {
            "faces": faces_response,
            "dominant": dominant_face["emotion"],
            "scores": dominant_face["all_scores"]
        }
    else:
        return {
            "faces": [],
            "dominant": None,
            "scores": None
        }

def append_face(faces_list, face_data):
    faces_list.append(face_data)
    return faces_list

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_upload():
    """
    Handle static image uploads from file selector or drag & drop.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    try:
        # Read file stream as OpenCV BGR image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image_cv is None:
            return jsonify({"error": "Failed to decode image file"}), 400
            
        results = predict_emotions_in_image(image_cv)
        return jsonify(results)
    except Exception as e:
        print(f"Error processing upload: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    """
    Handle real-time webcam frames sent as base64 string.
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data in payload"}), 400
        
    try:
        # Decode base64 image data
        image_data = base64.b64decode(data['image'])
        file_bytes = np.frombuffer(image_data, np.uint8)
        image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image_cv is None:
            return jsonify({"error": "Failed to decode frame data"}), 400
            
        results = predict_emotions_in_image(image_cv)
        return jsonify(results)
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize model at startup to verify and preheat TF session
    try:
        print("Pre-heating emotion detection model...")
        get_model()
        print("Pre-heating face cascade classifier...")
        get_face_cascade()
        print("Initialization successful. Starting server on http://127.0.0.1:5000")
    except Exception as ex:
        print(f"WARNING: Startup loading failed: {ex}")
        print("Server will start but might fail during predictions if dependencies are missing.")
        
    app.run(host='127.0.0.1', port=5000, debug=True)
