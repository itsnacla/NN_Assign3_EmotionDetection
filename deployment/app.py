import os
import base64
import numpy as np
import cv2
import csv
import time
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Prevent tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

# Define Flask application
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Flask session config
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super-secret-key-zen-study")

# Initialize Supabase client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"WARNING: Failed to initialize Supabase client: {e}")
else:
    print("WARNING: SUPABASE_URL and SUPABASE_KEY environment variables are missing.")


# Set upload limits (max 10MB)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Set paths
DEPLOYMENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(DEPLOYMENT_DIR)
MODEL_PATH = os.path.join(REPO_ROOT, 'src', 'models', 'model.h5')
LOGS_DIR = os.path.join(REPO_ROOT, 'output', 'logs')
CSV_PATH = os.path.join(LOGS_DIR, 'user_emotions.csv')

# Telemetry log settings
last_log_time = 0
LOG_COOLDOWN_SEC = 2.0  # Log once every 2 seconds for smooth webcam updates

def log_emotion_event(emotion, confidence):
    global last_log_time
    current_time = time.time()
    if current_time - last_log_time < LOG_COOLDOWN_SEC:
        return
        
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        file_exists = os.path.exists(CSV_PATH)
        
        with open(CSV_PATH, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'emotion', 'confidence'])
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([timestamp, emotion, f"{confidence:.4f}"])
            last_log_time = current_time
            print(f"Logged user emotion: {emotion} at {timestamp} with confidence {confidence:.4f}")
    except Exception as e:
        print(f"Error logging user emotion: {e}")


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
        # Log all dominant user emotions continuously
        log_emotion_event(dominant_face["emotion"], dominant_face["confidence"])

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
def landing():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', user_email=session['user'].get('email'))

@app.route('/login')
def login():
    if not supabase:
        return "Supabase client is not configured. Please check your environment variables or .env file.", 500
    
    # Construct the redirect URL (callback) pointing back to this Flask server
    redirect_url = url_for('auth_callback', _external=True)
    try:
        response = supabase.auth.sign_in_with_oauth(
            {
                "provider": "google",
                "options": {
                    "redirect_to": redirect_url,
                }
            }
        )
        return redirect(response.url)
    except Exception as e:
        print(f"Error during Google OAuth redirect: {e}")
        return f"OAuth Redirect Failed: {str(e)}", 500

@app.route('/auth/callback')
def auth_callback():
    code = request.args.get('code')
    if not code:
        return "No authorization code provided by OAuth flow.", 400
        
    try:
        # Exchange the code for a session
        res = supabase.auth.exchange_code_for_session({"auth_code": code})
        
        # Save user and session info into Flask session
        session['user'] = {
            'id': res.user.id,
            'email': res.user.email
        }
        session['access_token'] = res.session.access_token
        
        return redirect(url_for('dashboard'))
    except Exception as e:
        print(f"Error exchanging authorization code: {e}")
        return f"Authentication Session Exchange Failed: {str(e)}", 500

@app.route('/logout')
def logout():
    session.clear()
    try:
        if supabase:
            supabase.auth.sign_out()
    except Exception as e:
        print(f"Error signing out from Supabase: {e}")
    return redirect(url_for('landing'))


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

@app.route('/logs', methods=['GET'])
def get_logs():
    """
    Retrieve the last 7 logged events from the CSV file.
    """
    logs = []
    if os.path.exists(CSV_PATH):
        try:
            with open(CSV_PATH, mode='r') as f:
                reader = csv.reader(f)
                header = next(reader, None) # skip header
                rows = list(reader)
                # Take the last 7 logs and reverse to show newest first
                for r in reversed(rows[-7:]):
                    if len(r) == 3:
                        logs.append({
                            "timestamp": r[0],
                            "emotion": r[1],
                            "confidence": float(r[2])
                        })
        except Exception as e:
            print(f"Error reading CSV logs: {e}")
    return jsonify(logs)

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
