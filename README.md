# Zen Study: Emotion & Stress Detection System

[![Deployment Status](https://img.shields.io/badge/Status-Live-success?style=for-the-badge&logo=railway)](https://zenstudy.up.railway.app/)
[![Live Demo](https://img.shields.io/badge/Demo-zenstudy.up.railway.app-blue?style=for-the-badge)](https://zenstudy.up.railway.app/)

This project is an AI-powered emotion detection and stress level classification system using a custom **Convolutional Neural Network (CNN)** built with TensorFlow/Keras. The system classifies facial expressions into 7 basic emotions, maps them to a binary category (**Stress** vs. **Non-Stress**), and integrates them into an interactive Flask web application connected with Supabase for user authentication and historical log tracking.

> [!IMPORTANT]
> **Live Application**: This web application is deployed live and can be accessed at:
> 👉 **[https://zenstudy.up.railway.app/](https://zenstudy.up.railway.app/)**

---

## 🚀 Key Features

1. **Automated Data Preprocessing**: Extracts raw pixel data from `fer2013.csv` and converts them into structured `.png` image sets categorized by emotion label.
2. **Custom CNN Model**: A Deep Learning model designed using a layered CNN architecture (Conv2D, MaxPooling2D, Dropout, Flatten, and Dense) for 7-class facial emotion classification.
3. **Real-Time Detection**: Real-time facial emotion detection via local webcam feeds using OpenCV and Haar Cascade Classifier.
4. **Comprehensive Metrics & Evaluation**:
   - Evaluation of 7-class emotion classification (Confusion Matrix).
   - Binary classification metrics for Stress vs. Non-Stress (Stress: *Angry, Fearful, Sad*; Non-Stress: *Disgusted, Happy, Neutral, Surprised*).
   - ROC (Receiver Operating Characteristic) curve visualization with AUC (Area Under Curve) scores.
5. **Web Application Deployment**:
   - Interactive user dashboard powered by Flask, HTML, CSS, and JavaScript.
   - Google OAuth integration via Supabase Auth.
   - Emotion telemetry logging synchronized to Supabase Database, with an automatic fallback to a local CSV file (`user_emotions.csv`) if the database is unreachable or the user session is inactive.
   - Real-time user emotion history chart rendered dynamically on the dashboard.

---

## 📂 Project Structure

Below is the folder and file structure of the project:

```text
Emotion-detection/
├── .env                  # Environment variables configuration file (Supabase & Flask)
├── .env.example          # Environment variables configuration template
├── .gitignore            # Git ignore list
├── requirements.txt      # Required Python dependencies list
│
├── assets/               # Static assets storage directory
│   └── haarcascade_frontalface_default.xml  # Haar Cascade model for face detection
│
├── data/                 # Dataset directory (ignored by Git)
│   ├── raw/              # Raw dataset directory (for 'fer2013.csv')
│   ├── train/            # Extracted training image set (.png) categorized by emotion
│   └── test/             # Extracted test image set (.png) categorized by emotion
│
├── deployment/           # Flask web application code
│   ├── app.py            # Main Flask server backend
│   ├── auth_google.py    # Google OAuth Supabase authentication module
│   ├── static/           # Web static assets (CSS & JS)
│   │   ├── css/
│   │   │   └── style.css # Dashboard CSS stylesheet
│   │   └── js/
│   │       └── main.js   # Camera logic, API requests, and chart rendering
│   └── templates/        # HTML page templates
│       ├── index.html    # Main dashboard page
│       └── landing.html  # Authentication/landing page
│
├── output/               # Training & evaluation output directory (ignored by Git)
│   ├── logs/             # Local log files (user_emotions.csv & auth_debug.log)
│   └── plots/            # Accuracy charts, confusion matrices, and ROC curves
│
└── src/                  # Python source code for data processing and modeling
    ├── evaluate.py       # Evaluation script for the trained model on test data
    ├── train.py          # Main script to train the model or display live webcam classifications
    ├── models/           # Trained model storage directory
    │   └── model.h5      # Trained CNN model weights file (.h5)
    └── preprocessing/
        └── dataset_prepare.py  # Script to extract CSV entries to PNG images
```

---

## 🛠️ Installation & Setup Guide

Follow these steps to set up this project on your local machine:

### 1. Prerequisites
* Python 3.8 to 3.11.
* An active webcam (for live detection and web app camera telemetry).
* An internet connection for Supabase database integration.

### 2. Dependency Installation
Open a terminal/command prompt (PowerShell on Windows or Bash on macOS/Linux is recommended) and run the following commands:

```powershell
# 1. Navigate to the project directory
cd "e:\8th Sem UUM\Neural Network\Assignment03\Emotion-detection"

# 2. Create a new Virtual Environment
python -m venv .venv

# 3. Activate the Virtual Environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Variables Configuration (`.env`)
Copy the `.env.example` file to `.env`:

```powershell
copy .env.example .env
```

Open `.env` and fill in your Supabase credentials:
```ini
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-supabase-anon-public-key
FLASK_SECRET_KEY=generate-a-secure-secret-key-here
REDIRECT_URL=http://127.0.0.1:5000/auth/callback
```

### 4. Supabase Database Setup
To enable telemetry synchronization to the database, create a new table named `emotion_logs` in your Supabase project using the following SQL query in the Supabase SQL Editor:

```sql
create table public.emotion_logs (
  id bigint generated by default as identity primary key,
  timestamp timestamp with time zone default timezone('utc'::text, now()) not null,
  user_id uuid references auth.users(id) on delete cascade not null,
  email text not null,
  emotion text not null,
  confidence double precision not null
);

-- Enable Row Level Security (RLS)
alter table public.emotion_logs enable row level security;

-- Create policies to restrict users to their own logs
create policy "Allow user insert own logs" 
on public.emotion_logs 
for insert 
with check (auth.uid() = user_id);

create policy "Allow user select own logs" 
on public.emotion_logs 
for select 
using (auth.uid() = user_id);
```

Make sure to enable the Google OAuth provider under **Authentication -> Providers -> Google** in your Supabase dashboard and fill in the Client ID and Client Secret from your Google Cloud Console. Add the following Redirect URLs in your Supabase authentication configuration:
* Local environment: `http://127.0.0.1:5000/auth/callback`
* Production environment: `https://zenstudy.up.railway.app/auth/callback`

---

## 🏃 Running the Application

### Step 1: Dataset Extraction
Ensure that your raw `fer2013.csv` dataset is placed inside the `data/raw/` directory. Run the following command to extract the CSV records into PNG images:

```bash
python src/preprocessing/dataset_prepare.py
```
*This will generate folders containing categorized 48x48 pixel images under `data/train/` and `data/test/`.*

### Step 2: Model Training
To train the CNN model from scratch using the extracted images:

```bash
python src/train.py --mode train
```
*Once completed, the trained model weights will be saved to `src/models/model.h5` and accuracy/loss curves will be exported to `output/plots/`.*

### Step 3: Live Webcam Detection (Desktop App)
You can run a local desktop session to test the model's classification capability in real-time using your computer's webcam:

```bash
python src/train.py --mode display
```
*Press the `q` key on your keyboard to close the webcam window.*

### Step 4: Model Evaluation
Run the evaluation pipeline to compute metric summaries for both 7-class emotion detection and binary stress classification:

```bash
python src/evaluate.py
```
*Evaluation outputs (confusion matrix, ROC curves) will be saved to `output/plots/`, and a textual summary of metrics will be written to `output/plots/evaluation_metrics_summary.txt`.*

### Step 5: Start the Web Application Server
To start the interactive local Flask web application:

```bash
python deployment/app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`. You can log in using your Google account, analyze your facial expression in real-time via camera, upload a photo, and view your emotion logs on a dynamic telemetry chart.

---

## ☁️ Production Deployment (Railway)

This Flask web application is configured for deployment on **Railway** and is live at:
👉 **[https://zenstudy.up.railway.app/](https://zenstudy.up.railway.app/)**

### 1. Deployment Configuration on Railway
The repository includes configuration assets designed for instant Railway integration:
* **[Procfile](file:///e:/8th%20Sem%20UUM/Neural%20Network/Assignment03/Emotion-detection/Procfile)**: Specifies the production WSGI command using Gunicorn: `web: gunicorn wsgi:app`.
* **[wsgi.py](file:///e:/8th%20Sem%20UUM/Neural%20Network/Assignment03/Emotion-detection/wsgi.py)**: Serves as the production WSGI entry point importing the Flask app object.
* **Proxy Fix**: Utilizes Werkzeug's `ProxyFix` middleware inside [app.py](file:///e:/8th%20Sem%20UUM/Neural%20Network/Assignment03/Emotion-detection/deployment/app.py) to securely digest forwarding headers (`X-Forwarded-Proto`, `X-Forwarded-For`, `X-Forwarded-Host`) so Flask can reliably construct OAuth HTTPS redirect URLs.

### 2. Environment Variables in Railway (Variables)
Add the following variables to your Railway project control panel:
| Variable Name | Description / Example Value |
| :--- | :--- |
| `SUPABASE_URL` | Supabase project URL endpoint (`https://xxx.supabase.co`) |
| `SUPABASE_KEY` | Supabase anon public API key |
| `FLASK_SECRET_KEY` | A secure random key to encrypt Flask session cookies |
| `REDIRECT_URL` | Points to the production callback URL: `https://zenstudy.up.railway.app/auth/callback` |
| `PORT` | *Optional* (Railway sets this dynamically, defaults to `5000`) |

---

## 📊 Evaluation Metrics & Stress Mapping

Stress mapping is derived from the facial emotion probabilities predicted by the model:
* **Stress**: *Angry*, *Fearful*, *Sad*
* **Non-Stress**: *Disgusted*, *Happy*, *Neutral*, *Surprised*

All training performance metrics, plots, and figures are exported and can be reviewed in detail under the [output/plots/](file:///e:/8th%20Sem%20UUM/Neural%20Network/Assignment03/Emotion-detection/output/plots) directory.
