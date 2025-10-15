// Global variables
let detectionActive = false;
let statusCheckInterval = null;
const API_BASE_URL = 'http://localhost:5000';

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Update threshold display
    const thresholdSlider = document.getElementById('thresholdSlider');
    const thresholdValue = document.getElementById('thresholdValue');
    thresholdSlider.addEventListener('input', function() {
        thresholdValue.textContent = this.value;
        if (detectionActive) {
            updateThreshold(parseInt(this.value));
        }
    });

    // Update volume display
    const volumeSlider = document.getElementById('volumeSlider');
    const volumeValue = document.getElementById('volumeValue');
    volumeSlider.addEventListener('input', function() {
        volumeValue.textContent = this.value;
    });

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

// Start detection
async function startDetection() {
    try {
        // Call backend to start detection
        const response = await fetch(`${API_BASE_URL}/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Update video feed
            const videoPlaceholder = document.getElementById('videoPlaceholder');
            videoPlaceholder.innerHTML = `<img id="videoFeed" src="${API_BASE_URL}/video_feed" style="width: 100%; border-radius: 10px;">`;
            
            // Update UI
            detectionActive = true;
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            updateStatus('Active', 'success');
            
            // Start checking status
            startStatusCheck();
        } else {
            alert('Failed to start detection: ' + data.message);
        }
        
    } catch (error) {
        console.error('Error starting detection:', error);
        alert('Unable to connect to backend server. Please ensure the Flask server is running.');
    }
}

// Stop detection
async function stopDetection() {
    try {
        // Call backend to stop detection
        const response = await fetch(`${API_BASE_URL}/stop`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
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
        
        // Stop checking status
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
            statusCheckInterval = null;
        }
        
    } catch (error) {
        console.error('Error stopping detection:', error);
    }
}

// Start checking status periodically
function startStatusCheck() {
    statusCheckInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/status`);
            const data = await response.json();
            
            // Update UI with status data
            updateEyeState(data.eye_state, data.alert ? 'danger' : (data.timer > 0 ? 'warning' : 'success'));
            updateTimer(data.timer);
            
            // Show/hide alert
            if (data.alert) {
                showAlert();
            } else {
                hideAlert();
            }
            
        } catch (error) {
            console.error('Error fetching status:', error);
        }
    }, 200); // Check every 200ms for smooth updates
}

// Update threshold on backend
async function updateThreshold(threshold) {
    try {
        await fetch(`${API_BASE_URL}/update_threshold`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ threshold })
        });
    } catch (error) {
        console.error('Error updating threshold:', error);
    }
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
    const threshold = parseInt(document.getElementById('thresholdSlider').value);
    timerValue.textContent = seconds.toFixed(1) + 's';
    
    if (seconds >= threshold) {
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

console.log('Smart Driver Hub - Drowsiness Detection System Initialized');
console.log('Backend URL:', API_BASE_URL);
