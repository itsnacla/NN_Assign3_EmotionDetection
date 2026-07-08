import os
import base64
import numpy as np
import cv2
import csv
import time
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, has_request_context
from dotenv import load_dotenv
from supabase import create_client, Client, ClientOptions
from supabase_auth import SyncSupportedStorage

# Load environment variables
load_dotenv()

# Prevent tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf

from werkzeug.middleware.proxy_fix import ProxyFix

# Define Flask application
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Trust headers sent by reverse proxies (like Railway) to generate correct HTTPS redirect URLs
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Flask session config
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super-secret-key-zen-study")

class FlaskSessionStorage(SyncSupportedStorage):
    def get_item(self, key: str) -> str or None:
        if has_request_context():
            return session.get(key)
        return None

    def set_item(self, key: str, value: str) -> None:
        if has_request_context():
            session[key] = value
            session.modified = True

    def remove_item(self, key: str) -> None:
        if has_request_context():
            session.pop(key, None)
            session.modified = True

# Initialize Supabase client
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(
            SUPABASE_URL, 
            SUPABASE_KEY,
            options=ClientOptions(storage=FlaskSessionStorage())
        )
    except Exception as e:
        print(f"WARNING: Failed to initialize Supabase client: {e}")
else:
    print("WARNING: SUPABASE_URL and SUPABASE_KEY environment variables are missing.")

def get_request_supabase():
    """
    Creates a request-scoped Supabase client initialized with the current user's session token.
    This ensures all database queries respect Postgres Row-Level Security (RLS) policies.
    """
    if not (SUPABASE_URL and SUPABASE_KEY):
        return None
    try:
        # Create a new client instance for the current request
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        if 'access_token' in session:
            refresh_token = session.get('refresh_token', '')
            client.auth.set_session(session['access_token'], refresh_token)
        return client
    except Exception as e:
        print(f"Error creating request-scoped Supabase client: {e}")
        return None



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
        
    # Mark cooldown immediately to prevent concurrent triggers in subsequent frame requests
    last_log_time = current_time

    # Retrieve required authentication and session details from Flask's request context
    user_data = session.get('user')
    access_token = session.get('access_token')
    refresh_token = session.get('refresh_token')

    # Log status for debugging
    session_status = f"user_in_session={'user' in session}, access_token_present={access_token is not None}"
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(os.path.join(LOGS_DIR, 'auth_debug.log'), 'a') as debug_f:
            debug_f.write(f"{datetime.now()}: Status check: {session_status}\n")
    except Exception:
        pass

    def background_logging():
        logged_to_supabase = False
        if SUPABASE_URL and SUPABASE_KEY and user_data and access_token:
            try:
                # Create a request-scoped Supabase client for this background thread
                client = create_client(SUPABASE_URL, SUPABASE_KEY)
                client.auth.set_session(access_token, refresh_token or '')
                client.table("emotion_logs").insert({
                    "user_id": user_data['id'],
                    "email": user_data['email'],
                    "emotion": emotion,
                    "confidence": float(confidence)
                }).execute()
                print(f"Logged user emotion to Supabase for {user_data['email']}: {emotion} with confidence {confidence:.4f}")
                logged_to_supabase = True
            except Exception as e:
                error_msg = f"Failed to log to Supabase in background: {e}\n"
                print(error_msg, end='')
                try:
                    os.makedirs(LOGS_DIR, exist_ok=True)
                    with open(os.path.join(LOGS_DIR, 'auth_debug.log'), 'a') as debug_f:
                        debug_f.write(f"{datetime.now()}: {error_msg}")
                except Exception:
                    pass
                    
        # Fallback to local CSV logging if not logged to Supabase
        if not logged_to_supabase:
            try:
                os.makedirs(LOGS_DIR, exist_ok=True)
                file_exists = os.path.exists(CSV_PATH)
                
                with open(CSV_PATH, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(['timestamp', 'emotion', 'confidence'])
                    
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    writer.writerow([timestamp, emotion, f"{confidence:.4f}"])
                    print(f"Logged user emotion to CSV: {emotion} at {timestamp} with confidence {confidence:.4f}")
            except Exception as e:
                print(f"Error logging user emotion to CSV: {e}")

    # Launch daemon background thread
    threading.Thread(target=background_logging, daemon=True).start()


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
            # Fallback to local repository assets directory
            deployment_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(deployment_dir)
            cascade_path = os.path.join(repo_root, 'assets', 'haarcascade_frontalface_default.xml')
            
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
    faces_detected = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
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
        return "Supabase client is not configured.", 500

    try:
        response = supabase.auth.sign_in_with_oauth(
            {
                "provider": "google",
                "options": {
                    "redirect_to": url_for('auth_callback', _external=True)
                }
            }
        )

        return redirect(response.url)

    except Exception as e:
        print(f"OAuth error: {e}")
        return str(e), 500

@app.route("/auth/callback")
def auth_callback():
    code = request.args.get("code")

    print("CALLBACK CODE:", code)

    if not code:
        return "No authorization code provided by OAuth flow.", 400

    try:
        # Exchange the code for a session (retrieves code_verifier from Flask session storage)
        res = supabase.auth.exchange_code_for_session({"auth_code": code})
        
        # Save user and session info into Flask session
        session['user'] = {
            'id': res.user.id,
            'email': res.user.email
        }
        session['access_token'] = res.session.access_token
        session['refresh_token'] = res.session.refresh_token
        session.modified = True
        
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
    Retrieve the last 7 logged events from Supabase (or CSV fallback).
    """
    logs = []
    req_supabase = get_request_supabase()
    
    # Try fetching from Supabase first if a user session exists
    if req_supabase and 'user' in session:
        try:
            user_data = session['user']
            response = req_supabase.table("emotion_logs") \
                .select("timestamp, emotion, confidence") \
                .eq("user_id", user_data['id']) \
                .order("timestamp", desc=True) \
                .limit(7) \
                .execute()
                
            # Process and format timestamps for display on the front-end chart
            for item in response.data:
                # Timestamptz format: "2026-07-08T01:36:25+00:00"
                # Convert to "YYYY-MM-DD HH:MM:SS" local format
                try:
                    dt = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                    formatted_ts = dt.strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    formatted_ts = item['timestamp']
                    
                logs.append({
                    "timestamp": formatted_ts,
                    "emotion": item['emotion'],
                    "confidence": float(item['confidence'])
                })
            return jsonify(logs)
        except Exception as e:
            print(f"Failed to fetch logs from Supabase: {e}. Falling back to CSV...")
            
    # Fallback to local CSV logging
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
        print("Initialization successful.")
    except Exception as ex:
        print(f"WARNING: Startup loading failed: {ex}")
        print("Server will start but might fail during predictions if dependencies are missing.")
        
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
