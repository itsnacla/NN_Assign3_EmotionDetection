// State Variables
let currentMode = 'webcam'; // 'webcam' or 'upload'
let streamActive = false;
let webcamStream = null;
let animationFrameId = null;
let lastFrameTime = 0;
const frameThrottleMs = 180; // Send frames roughly 5-6 times per second to prevent network lag

// Audio beep alert settings
let lastBeepTime = 0;
const beepCooldownMs = 3000; // 3 seconds cooldown between alerts

// Web Audio API Beep Synthesizer
function playBeep(frequency = 600, duration = 0.25) {
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        
        oscillator.type = 'sine';
        oscillator.frequency.value = frequency;
        
        gainNode.gain.setValueAtTime(0.15, audioCtx.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration);
        
        oscillator.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        
        oscillator.start();
        oscillator.stop(audioCtx.currentTime + duration);
    } catch (err) {
        console.error("Audio playback error:", err);
    }
}

// DOM Elements
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const processingInfo = document.getElementById('processing-info');
const dominantEmotionBadge = document.getElementById('dominant-emotion-badge');

// Tab Buttons
const btnWebcamMode = document.getElementById('btn-webcam-mode');
const btnUploadMode = document.getElementById('btn-upload-mode');

// Webcam Workspace Elements
const webcamWorkspace = document.getElementById('webcam-workspace');
const videoElement = document.getElementById('webcam-stream');
const webcamOverlay = document.getElementById('webcam-overlay');
const cameraPrompt = document.getElementById('camera-prompt');
const btnStartCamera = document.getElementById('btn-start-camera');
const btnToggleStream = document.getElementById('btn-toggle-stream');

// Upload Workspace Elements
const uploadWorkspace = document.getElementById('upload-workspace');
const dragDropZone = document.getElementById('drag-drop-zone');
const imageInput = document.getElementById('image-input');
const uploadPreviewContainer = document.getElementById('upload-preview-container');
const uploadPreview = document.getElementById('upload-preview');
const uploadOverlay = document.getElementById('upload-overlay');
const btnResetUpload = document.getElementById('btn-reset-upload');

// Initialize Web App
document.addEventListener('DOMContentLoaded', () => {
    btnStartCamera.addEventListener('click', startWebcam);
    btnToggleStream.addEventListener('click', toggleStream);
    btnResetUpload.addEventListener('click', resetUpload);
    
    // Setup Image Input Handler
    imageInput.addEventListener('change', handleFileSelect);
    
    // Drag and Drop events
    ['dragenter', 'dragover'].forEach(eventName => {
        dragDropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dragDropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dragDropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dragDropZone.classList.remove('dragover');
        }, false);
    });

    dragDropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            imageInput.files = files;
            processUpload(files[0]);
        }
    });
});

// Mode Switching (Webcam vs. Upload)
function switchMode(mode) {
    if (currentMode === mode) return;
    
    currentMode = mode;
    
    // Update Button styling
    if (mode === 'webcam') {
        btnWebcamMode.classList.add('active');
        btnUploadMode.classList.remove('active');
        webcamWorkspace.classList.add('active');
        uploadWorkspace.classList.remove('active');
        
        // Resume webcam if it was active
        if (streamActive && videoElement.srcObject) {
            startFrameLoop();
        }
    } else {
        btnWebcamMode.classList.remove('active');
        btnUploadMode.classList.add('active');
        webcamWorkspace.classList.remove('active');
        uploadWorkspace.classList.add('active');
        
        // Pause webcam frame sending
        stopFrameLoop();
    }
    
    // Clear display overlays and metrics on switch
    clearOverlay(webcamOverlay);
    clearOverlay(uploadOverlay);
    resetMetrics();
}

// Start Webcam Feed
async function startWebcam() {
    updateStatus('connecting', 'Requesting camera access...');
    try {
        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        };
        
        webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
        videoElement.srcObject = webcamStream;
        
        videoElement.onloadedmetadata = () => {
            videoElement.play();
            
            // Adjust overlay canvas bounds to match internal video resolution
            webcamOverlay.width = videoElement.videoWidth;
            webcamOverlay.height = videoElement.videoHeight;
            
            cameraPrompt.classList.add('hidden');
            btnToggleStream.classList.remove('hidden');
            btnToggleStream.textContent = 'Stop Stream';
            
            streamActive = true;
            updateStatus('active', 'Webcam stream active');
            startFrameLoop();
        };
    } catch (err) {
        console.error('Camera Access Error:', err);
        updateStatus('error', 'Camera access denied or unavailable');
    }
}

// Stop Webcam Feed
function stopWebcam() {
    stopFrameLoop();
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    videoElement.srcObject = null;
    cameraPrompt.classList.remove('hidden');
    btnToggleStream.classList.add('hidden');
    streamActive = false;
    updateStatus('idle', 'Webcam stream terminated');
    clearOverlay(webcamOverlay);
    resetMetrics();
}

function toggleStream() {
    if (streamActive) {
        stopWebcam();
    } else {
        startWebcam();
    }
}

// Webcam Frame Capture Loop (throttled)
function startFrameLoop() {
    if (animationFrameId) cancelAnimationFrame(animationFrameId);
    
    const sendFrame = async (timestamp) => {
        if (!streamActive || currentMode !== 'webcam') return;
        
        if (timestamp - lastFrameTime >= frameThrottleMs) {
            lastFrameTime = timestamp;
            
            // Draw video frame to an offscreen canvas
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = videoElement.videoWidth;
            tempCanvas.height = videoElement.videoHeight;
            const ctx = tempCanvas.getContext('2d');
            
            // Draw video to canvas (normal orientation, no mirror logic needed as overlay mirror matches)
            ctx.drawImage(videoElement, 0, 0, tempCanvas.width, tempCanvas.height);
            
            // Compress and convert to base64 jpeg
            const dataUrl = tempCanvas.toDataURL('image/jpeg', 0.7);
            const base64Data = dataUrl.split(',')[1];
            
            showProcessing(true);
            
            try {
                const response = await fetch('/predict_frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64Data })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    drawFaces(webcamOverlay, data.faces, true);
                    updateMetrics(data.dominant, data.scores);
                }
            } catch (err) {
                console.error('API Frame processing error:', err);
            } finally {
                showProcessing(false);
            }
        }
        
        animationFrameId = requestAnimationFrame(sendFrame);
    };
    
    animationFrameId = requestAnimationFrame(sendFrame);
}

function stopFrameLoop() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
}

// Upload Mode Handling
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processUpload(files[0]);
    }
}

function processUpload(file) {
    updateStatus('connecting', 'Uploading image for telemetry analysis...');
    showProcessing(true);
    
    const reader = new FileReader();
    reader.onload = (event) => {
        uploadPreview.src = event.target.result;
        
        uploadPreview.onload = () => {
            // Setup canvas size
            uploadOverlay.width = uploadPreview.naturalWidth;
            uploadOverlay.height = uploadPreview.naturalHeight;
            
            dragDropZone.classList.add('hidden');
            uploadPreviewContainer.classList.remove('hidden');
            btnResetUpload.classList.remove('hidden');
            
            // Send to Server
            const formData = new FormData();
            formData.append('image', file);
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(res => {
                if (!res.ok) throw new Error('Prediction API failed');
                return res.json();
            })
            .then(data => {
                drawFaces(uploadOverlay, data.faces, false);
                updateMetrics(data.dominant, data.scores);
                updateStatus('idle', 'Analysis completed');
            })
            .catch(err => {
                console.error('File Upload Prediction Error:', err);
                updateStatus('error', 'Failed to analyze uploaded image');
            })
            .finally(() => {
                showProcessing(false);
            });
        };
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    imageInput.value = '';
    dragDropZone.classList.remove('hidden');
    uploadPreviewContainer.classList.add('hidden');
    btnResetUpload.classList.add('hidden');
    uploadPreview.src = '';
    clearOverlay(uploadOverlay);
    resetMetrics();
    updateStatus('idle', 'Waiting for image upload');
}

// Render bounding boxes and predictions onto canvas overlays
function drawFaces(canvas, faces, isMirrored = false) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!faces || faces.length === 0) return;
    
    faces.forEach(face => {
        const { x, y, w, h, emotion, confidence } = face;
        
        // Compute horizontal placement depending on mirrored camera view
        const drawX = isMirrored ? (canvas.width - x - w) : x;
        
        const stressEmotions = ['Angry', 'Sad', 'Disgusted'];
        const isStress = stressEmotions.includes(emotion);
        const accentColor = isStress ? '#800000' : '#38bdf8';
        const shadowColor = isStress ? 'rgba(128, 0, 0, 0.5)' : 'rgba(56, 189, 248, 0.5)';
        const labelBgColor = isStress ? 'rgba(128, 0, 0, 0.85)' : 'rgba(13, 13, 27, 0.85)';
        
        // Set glow and line styles
        ctx.strokeStyle = accentColor;
        ctx.lineWidth = Math.max(3, canvas.width / 200);
        ctx.shadowColor = shadowColor;
        ctx.shadowBlur = 10;
        
        // Draw primary facial bounding box
        ctx.strokeRect(drawX, y, w, h);
        
        // Draw corners decorations (hud overlay style - inherits box color)
        const offset = Math.min(w, h) * 0.15;
        ctx.strokeStyle = accentColor;
        ctx.shadowColor = shadowColor;
        ctx.lineWidth = ctx.lineWidth * 1.5;
        
        // Top Left corner
        ctx.beginPath();
        ctx.moveTo(drawX, y + offset);
        ctx.lineTo(drawX, y);
        ctx.lineTo(drawX + offset, y);
        ctx.stroke();
        
        // Top Right corner
        ctx.beginPath();
        ctx.moveTo(drawX + w - offset, y);
        ctx.lineTo(drawX + w, y);
        ctx.lineTo(drawX + w, y + offset);
        ctx.stroke();
        
        // Bottom Left corner
        ctx.beginPath();
        ctx.moveTo(drawX, y + h - offset);
        ctx.lineTo(drawX, y + h);
        ctx.lineTo(drawX + offset, y + h);
        ctx.stroke();
        
        // Bottom Right corner
        ctx.beginPath();
        ctx.moveTo(drawX + w - offset, y + h);
        ctx.lineTo(drawX + w, y + h);
        ctx.lineTo(drawX + w, y + h - offset);
        ctx.stroke();
        
        // Draw Prediction Label
        ctx.shadowBlur = 0;
        ctx.fillStyle = labelBgColor;
        
        const fontSize = Math.max(14, canvas.width / 40);
        ctx.font = `600 ${fontSize}px 'Outfit', sans-serif`;
        const labelText = `${emotion} (${(confidence * 100).toFixed(0)}%)`;
        const textWidth = ctx.measureText(labelText).width;
        const textHeight = fontSize;
        
        // Label background container
        ctx.fillRect(drawX, y - textHeight - 12, textWidth + 16, textHeight + 8);
        
        // Label Text
        ctx.fillStyle = '#f3f3fb';
        ctx.fillText(labelText, drawX + 8, y - 8);
    });
}

function clearOverlay(canvas) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// Update the dominant badge & confidence bars
function updateMetrics(dominant, scores) {
    if (!dominant || !scores) {
        resetMetrics();
        return;
    }
    
    // Update badge
    dominantEmotionBadge.textContent = dominant;
    dominantEmotionBadge.className = 'badge detected';
    
    // Sort scores and update bars
    for (const [emotion, score] of Object.entries(scores)) {
        const pctText = document.getElementById(`pct-${emotion}`);
        const barFill = document.getElementById(`bar-${emotion}`);
        const row = document.querySelector(`.emotion-metric-row[data-emotion="${emotion}"]`);
        
        if (pctText && barFill && row) {
            const pctVal = (score * 100).toFixed(1);
            pctText.textContent = `${pctVal}%`;
            barFill.style.width = `${pctVal}%`;
            
            // Toggle highlight styling for dominant emotion
            if (emotion === dominant) {
                row.classList.add('dominant');
            } else {
                row.classList.remove('dominant');
            }
        }
    }
    
    // Play a stress alert beep if dominant emotion implies stress and is highly confident
    const stressEmotions = ['Angry', 'Fearful', 'Sad'];
    if (stressEmotions.includes(dominant) && scores[dominant] > 0.55) {
        const now = Date.now();
        if (now - lastBeepTime > beepCooldownMs) {
            lastBeepTime = now;
            playBeep(650, 0.3); // Play a warning beep at 650Hz
        }
    }
}

function resetMetrics() {
    dominantEmotionBadge.textContent = 'No face detected';
    dominantEmotionBadge.className = 'badge';
    
    const emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised'];
    emotions.forEach(emotion => {
        const pctText = document.getElementById(`pct-${emotion}`);
        const barFill = document.getElementById(`bar-${emotion}`);
        const row = document.querySelector(`.emotion-metric-row[data-emotion="${emotion}"]`);
        
        if (pctText && barFill && row) {
            pctText.textContent = '0%';
            barFill.style.width = '0%';
            row.classList.remove('dominant');
        }
    });
}

// Status Panel Updates
function updateStatus(status, text) {
    statusDot.className = 'dot';
    
    if (status === 'connecting') {
        statusDot.classList.add('idle');
    } else if (status === 'active') {
        statusDot.classList.add('streaming');
    } else if (status === 'error') {
        statusDot.classList.add('error');
    } else {
        statusDot.classList.add('idle');
    }
    
    statusText.textContent = text;
}

function showProcessing(show) {
    if (show) {
        processingInfo.classList.remove('hidden');
    } else {
        processingInfo.classList.add('hidden');
    }
}
