// Global variables
let detectionActive = false;
let eyeClosedStartTime = null;
let eyeClosedDuration = 0;
let alertThreshold = 5; // seconds
let alarmAudio = null;
let detectionInterval = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Update threshold display
    const thresholdSlider = document.getElementById('thresholdSlider');
    const thresholdValue = document.getElementById('thresholdValue');
    thresholdSlider.addEventListener('input', function() {
        alertThreshold = parseInt(this.value);
        thresholdValue.textContent = this.value;
    });

    // Update volume display
    const volumeSlider = document.getElementById('volumeSlider');
    const volumeValue = document.getElementById('volumeValue');
    volumeSlider.addEventListener('input', function() {
        volumeValue.textContent = this.value;
    });

    // Initialize alarm sound
    createAlarmSound();

    // Smooth scrolling for navigation
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});

// Create alarm sound using Web Audio API
function createAlarmSound() {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) {
        console.warn('Web Audio API not supported');
        return;
    }
}

// Play alarm sound
function playAlarm() {
    if (!document.getElementById('soundToggle').checked) {
        return;
    }

    const volume = document.getElementById('volumeSlider').value / 100;
    
    // Create audio context
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const audioContext = new AudioContext();
    
    // Create oscillator for beep sound
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.value = 1500; // Hz
    oscillator.type = 'square';
    
    gainNode.gain.value = volume;
    
    oscillator.start();
    
    // Store for stopping later
    if (!window.activeAlarms) window.activeAlarms = [];
    window.activeAlarms.push({ oscillator, audioContext });
    
    // Auto-stop after 2 seconds and restart
    setTimeout(() => {
        oscillator.stop();
        if (detectionActive && document.getElementById('alertBox').style.display !== 'none') {
            playAlarm(); // Restart alarm if still in alert state
        }
    }, 500);
}

// Stop alarm sound
function stopAlarm() {
    if (window.activeAlarms) {
        window.activeAlarms.forEach(alarm => {
            try {
                alarm.oscillator.stop();
                alarm.audioContext.close();
            } catch (e) {
                console.warn('Error stopping alarm:', e);
            }
        });
        window.activeAlarms = [];
    }
}

// Start detection
async function startDetection() {
    try {
        // Request camera access
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 }
            } 
        });
        
        // Create video element
        const videoPlaceholder = document.getElementById('videoPlaceholder');
        videoPlaceholder.innerHTML = '<video id="videoFeed" autoplay playsinline></video>';
        const video = document.getElementById('videoFeed');
        video.srcObject = stream;
        video.style.width = '100%';
        video.style.borderRadius = '10px';
        
        // Update UI
        detectionActive = true;
        document.getElementById('startBtn').disabled = true;
        document.getElementById('stopBtn').disabled = false;
        updateStatus('Active', 'success');
        
        // Start detection simulation
        startDetectionLoop();
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Unable to access camera. Please ensure camera permissions are granted.');
    }
}

// Stop detection
function stopDetection() {
    // Stop camera
    const video = document.getElementById('videoFeed');
    if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
    
    // Reset UI
    const videoPlaceholder = document.getElementById('videoPlaceholder');
    videoPlaceholder.innerHTML = `
        <i class="fas fa-video-slash"></i>
        <p>Camera feed will appear here</p>
        <small>Click "Start Detection" to begin</small>
    `;
    
    detectionActive = false;
    document.getElementById('startBtn').disabled = false;
    document.getElementById('stopBtn').disabled = true;
    updateStatus('Not Started', 'default');
    updateEyeState('N/A', 'default');
    updateTimer(0);
    hideAlert();
    stopAlarm();
    
    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }
}

// Simulated detection loop (in real implementation, this would use Python backend)
function startDetectionLoop() {
    let simulatedEyesClosed = false;
    let randomChangeCounter = 0;
    
    detectionInterval = setInterval(() => {
        if (!detectionActive) return;
        
        // Simulate random eye state changes
        randomChangeCounter++;
        if (randomChangeCounter > 30) { // Change state every ~3 seconds
            simulatedEyesClosed = Math.random() > 0.7; // 30% chance eyes closed
            randomChangeCounter = 0;
        }
        
        if (simulatedEyesClosed) {
            // Eyes closed
            if (eyeClosedStartTime === null) {
                eyeClosedStartTime = Date.now();
            }
            
            eyeClosedDuration = (Date.now() - eyeClosedStartTime) / 1000;
            updateEyeState('Closed', 'warning');
            updateTimer(eyeClosedDuration);
            
            // Check if threshold exceeded
            if (eyeClosedDuration >= alertThreshold) {
                showAlert();
                playAlarm();
            }
        } else {
            // Eyes open
            eyeClosedStartTime = null;
            eyeClosedDuration = 0;
            updateEyeState('Open', 'success');
            updateTimer(0);
            hideAlert();
            stopAlarm();
        }
    }, 100); // Check every 100ms
}

// Update status display
function updateStatus(status, type) {
    const statusValue = document.getElementById('statusValue');
    statusValue.textContent = status;
    statusValue.style.color = type === 'success' ? 'var(--success-color)' : 
                               type === 'warning' ? 'var(--warning-color)' : 
                               type === 'danger' ? 'var(--danger-color)' : 
                               'var(--primary-color)';
}

// Update eye state display
function updateEyeState(state, type) {
    const eyeState = document.getElementById('eyeState');
    eyeState.textContent = state;
    eyeState.style.color = type === 'success' ? 'var(--success-color)' : 
                            type === 'warning' ? 'var(--warning-color)' : 
                            type === 'danger' ? 'var(--danger-color)' : 
                            'var(--gray)';
}

// Update timer display
function updateTimer(seconds) {
    const timerValue = document.getElementById('timerValue');
    timerValue.textContent = seconds.toFixed(1) + 's';
    
    if (seconds >= alertThreshold) {
        timerValue.style.color = 'var(--danger-color)';
    } else if (seconds > 0) {
        timerValue.style.color = 'var(--warning-color)';
    } else {
        timerValue.style.color = 'var(--success-color)';
    }
}

// Show alert
function showAlert() {
    const alertBox = document.getElementById('alertBox');
    alertBox.style.display = 'block';
    updateStatus('ALERT!', 'danger');
}

// Hide alert
function hideAlert() {
    const alertBox = document.getElementById('alertBox');
    alertBox.style.display = 'none';
    if (detectionActive) {
        updateStatus('Active', 'success');
    }
}

// Note: In a production environment, you would need to:
// 1. Set up a Python backend server (Flask/FastAPI)
// 2. Stream video from browser to Python backend
// 3. Process frames with OpenCV on backend
// 4. Send detection results back to frontend via WebSocket
// 5. Update UI based on real detection data

console.log('Smart Driver Hub - Drowsiness Detection System Initialized');
console.log('Note: This is a frontend demo. For full functionality, integrate with Python backend.');
