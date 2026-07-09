// State Variables
let currentMode = 'webcam'; // 'webcam' or 'upload'
let streamActive = false;
let webcamStream = null;
let frameLoopActive = false;
let lastFrameTime = 0;
const frameThrottleMs = 100; // Target ~10 FPS for fluid face tracking

// Audio beep alert settings
let lastBeepTime = 0;
const beepCooldownMs = 3000; // 3 seconds cooldown between alerts

// State Variables for Stress Intervention Modal
let isStressModalActive = false;
let consecutiveStressTicks = 0;
let breathingIntervalId = null;

// Telemetry History Graph
let telemetryChart = null;

// Web Audio API Beep Synthesizer (Loud Alarm style)
function playBeep(frequency = 880, duration = 0.5) {
    try {
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        
        // Auto-resume if context is suspended by browser autoplay policy
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }
        
        const oscillator = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        
        // Use 'sawtooth' wave for a buzzy, prominent alarm sound
        oscillator.type = 'sawtooth';
        oscillator.frequency.value = frequency;
        
        // Maintain constant volume and fade out quickly only at the very end
        gainNode.gain.setValueAtTime(0.3, audioCtx.currentTime);
        gainNode.gain.setValueAtTime(0.3, audioCtx.currentTime + duration - 0.15);
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

    // Auto-start webcam when page opens
    startWebcam();
    
    // Initialize telemetry chart
    initChart();
    
    // Fetch logs initially and setup periodic logs telemetry fetcher
    fetchLogs();
    setInterval(fetchLogs, 2500);
    
    // Setup Stress Modal Button close listener
    const btnCloseStress = document.getElementById('btn-close-stress');
    if (btnCloseStress) {
        btnCloseStress.addEventListener('click', closeStressModal);
    }

    // Setup Profile Dropdown Toggle
    const profileTrigger = document.getElementById('profile-trigger');
    const profileDropdown = document.getElementById('profile-dropdown-menu');
    if (profileTrigger && profileDropdown) {
        profileTrigger.addEventListener('click', (e) => {
            e.stopPropagation();
            profileDropdown.classList.toggle('show');
        });
        document.addEventListener('click', (e) => {
            if (!profileDropdown.contains(e.target) && e.target !== profileTrigger) {
                profileDropdown.classList.remove('show');
            }
        });
    }
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
            
            // Adjust camera block size to match the webcam's native aspect ratio
            const container = videoElement.closest('.media-container');
            if (container && videoElement.videoWidth && videoElement.videoHeight) {
                container.style.aspectRatio = `${videoElement.videoWidth} / ${videoElement.videoHeight}`;
            }
            
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
    
    // Reset aspect ratio to default
    const container = videoElement.closest('.media-container');
    if (container) {
        container.style.aspectRatio = '';
    }
    
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

// Webcam Frame Capture Loop (using recursive setTimeout to prevent request queuing/bottlenecks)
function startFrameLoop() {
    if (frameLoopActive) return;
    frameLoopActive = true;
    runFrameCycle();
}

function stopFrameLoop() {
    frameLoopActive = false;
}

async function runFrameCycle() {
    if (!frameLoopActive || !streamActive || currentMode !== 'webcam') {
        frameLoopActive = false;
        return;
    }
    
    const startTime = Date.now();
    
    // Draw video frame to an offscreen canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = videoElement.videoWidth;
    tempCanvas.height = videoElement.videoHeight;
    
    if (tempCanvas.width > 0 && tempCanvas.height > 0) {
        // Ensure overlay canvas dimensions match the active video stream resolution
        if (webcamOverlay.width !== videoElement.videoWidth || webcamOverlay.height !== videoElement.videoHeight) {
            webcamOverlay.width = videoElement.videoWidth;
            webcamOverlay.height = videoElement.videoHeight;
        }
        
        const ctx = tempCanvas.getContext('2d');
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
                
                // Guard clause to prevent updating UI if stream was stopped or mode changed during request
                if (frameLoopActive && streamActive && currentMode === 'webcam') {
                    drawFaces(webcamOverlay, data.faces, true);
                    updateMetrics(data.dominant, data.scores);
                }
            }
        } catch (err) {
            console.error('API Frame processing error:', err);
        } finally {
            showProcessing(false);
        }
    }
    
    // Schedule the next frame cycle, taking processing time into account
    if (frameLoopActive && streamActive && currentMode === 'webcam') {
        const elapsedTime = Date.now() - startTime;
        const nextDelay = Math.max(0, frameThrottleMs - elapsedTime);
        setTimeout(runFrameCycle, nextDelay);
    } else {
        frameLoopActive = false;
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
                // Guard clause to prevent updating UI if user switched mode during request
                if (currentMode !== 'upload') return;
                
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
        
        const stressEmotions = ['Angry', 'Sad', 'Fearful'];
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
    
    // Play stress alarm and track consecutive ticks to trigger the guided breathing modal
    const stressEmotions = ['Angry', 'Fearful', 'Sad'];
    if (stressEmotions.includes(dominant) && scores[dominant] > 0.40) {
        consecutiveStressTicks++;
        
        // Trigger beep alert
        const now = Date.now();
        if (now - lastBeepTime > beepCooldownMs) {
            lastBeepTime = now;
            playBeep(880, 0.5); // Play warning beep
        }
        
        // If user remains stressed for 5 consecutive ticks (~1 second of continuous stress), trigger intervention
        if (consecutiveStressTicks >= 5 && !isStressModalActive) {
            triggerStressIntervention();
        }
    } else {
        consecutiveStressTicks = 0;
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
        processingInfo.classList.remove('invisible');
    } else {
        processingInfo.classList.add('invisible');
    }
}

// Initialize Chart.js stress telemetry graph
function initChart() {
    const ctx = document.getElementById('telemetry-chart');
    if (!ctx) return;
    
    // Set global font family for Chart.js
    Chart.defaults.font.family = "'Outfit', sans-serif";
    Chart.defaults.font.size = 10;
    Chart.defaults.color = '#8e90ab';
    
    telemetryChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [], // Timestamps
            datasets: [{
                label: 'Stress Index %',
                data: [], // Stress level history
                borderColor: '#38bdf8',
                backgroundColor: 'rgba(56, 189, 248, 0.12)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 2.5,
                pointBackgroundColor: '#38bdf8',
                pointBorderColor: 'transparent',
                pointHoverRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(20, 20, 36, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#38bdf8',
                    borderColor: 'rgba(255, 255, 255, 0.08)',
                    borderWidth: 1,
                    padding: 8,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return `Stress Level: ${context.parsed.y.toFixed(0)}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 5 }
                },
                y: {
                    min: 0,
                    max: 100,
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { stepSize: 25 }
                }
            }
        }
    });
}

// Fetch recently logged events from CSV database
async function fetchLogs() {
    try {
        const response = await fetch('/logs');
        if (response.ok) {
            const logs = await response.json();
            updateTelemetryChart(logs);
        }
    } catch (err) {
        console.error("Error fetching logs:", err);
    }
}

// Update the chart dataset dynamically from logs history
function updateTelemetryChart(logs) {
    if (!telemetryChart) return;
    
    if (!logs || logs.length === 0) {
        return;
    }
    
    // We reverse logs since backend `/logs` serves newest first, and line chart progresses left-to-right (oldest-to-newest)
    const reversedLogs = [...logs].reverse();
    
    // Map timestamps to display format (HH:MM:SS)
    const labels = reversedLogs.map(log => {
        try {
            // Split YYYY-MM-DD HH:MM:SS to fetch HH:MM:SS
            return log.timestamp.split(' ')[1] || log.timestamp;
        } catch {
            return log.timestamp;
        }
    });
    
    // Calculate Stress Index
    // Angry, Fearful, Sad are stress emotions (scale based on confidence)
    // Happy, Neutral, Surprised, Disgusted are calm emotions (fixed low baseline)
    const stressEmotions = ['Angry', 'Fearful', 'Sad'];
    const stressData = reversedLogs.map(log => {
        if (stressEmotions.includes(log.emotion)) {
            return log.confidence * 100;
        } else {
            // Calm emotions represent low background stress level (e.g. 10%)
            return 10;
        }
    });
    
    // Check if the latest telemetry point is in high stress (> 40%)
    const latestLog = logs[0];
    const isLatestStressed = latestLog && stressEmotions.includes(latestLog.emotion) && latestLog.confidence > 0.40;
    
    // Dynamic chart accent coloring based on current stress state
    const dataset = telemetryChart.data.datasets[0];
    if (isLatestStressed) {
        dataset.borderColor = '#ff416c';
        dataset.backgroundColor = 'rgba(255, 65, 108, 0.12)';
        dataset.pointBackgroundColor = '#ff416c';
    } else {
        dataset.borderColor = '#38bdf8';
        dataset.backgroundColor = 'rgba(56, 189, 248, 0.12)';
        dataset.pointBackgroundColor = '#38bdf8';
    }
    
    telemetryChart.data.labels = labels;
    dataset.data = stressData;
    telemetryChart.update('none'); // Update without full layout animation to keep performance fast
}

// Show the Guided Breathing Wellness Modal and pause capturing
function triggerStressIntervention() {
    isStressModalActive = true;
    consecutiveStressTicks = 0;
    
    // Stop the webcam capture loop visually
    stopFrameLoop();
    
    // Display the modal
    const modal = document.getElementById('stress-modal');
    if (modal) modal.classList.remove('hidden');
    
    // Play a soft wellness major triad notification chord (C Major)
    playBeep(523.25, 0.35); // C5
    setTimeout(() => playBeep(659.25, 0.35), 120); // E5
    setTimeout(() => playBeep(783.99, 0.6), 240); // G5
    
    // Start breathing guide
    startBreathingGuide();
}

// Close Guided Breathing Wellness Modal and resume capturing
function closeStressModal() {
    const modal = document.getElementById('stress-modal');
    if (modal) modal.classList.add('hidden');
    
    stopBreathingGuide();
    isStressModalActive = false;
    
    // Play a friendly resume sound
    playBeep(659.25, 0.2);
    setTimeout(() => playBeep(880, 0.3), 100);
    
    // Resume the webcam loop
    if (streamActive && currentMode === 'webcam') {
        startFrameLoop();
    }
}

// Guided breathing animation cycle controller (Inhale 4s / Exhale 4s)
function startBreathingGuide() {
    const circle = document.getElementById('breathing-circle');
    const label = document.getElementById('breathing-text');
    if (!circle || !label) return;
    
    let isInhaling = true;
    circle.className = 'breathing-ring-inner expand';
    label.textContent = 'Inhale deeply...';
    label.style.color = '#38bdf8';
    
    breathingIntervalId = setInterval(() => {
        isInhaling = !isInhaling;
        if (isInhaling) {
            circle.className = 'breathing-ring-inner expand';
            label.textContent = 'Inhale deeply...';
            label.style.color = '#38bdf8';
        } else {
            circle.className = 'breathing-ring-inner shrink';
            label.textContent = 'Exhale slowly...';
            label.style.color = '#ff416c';
        }
    }, 4000);
}

// Clear guided breathing cycle timers
function stopBreathingGuide() {
    if (breathingIntervalId) {
        clearInterval(breathingIntervalId);
        breathingIntervalId = null;
    }
}
