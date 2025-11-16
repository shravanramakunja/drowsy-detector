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

    // Prepare fake sample datasets and populate selector
    prepareSampleDatasets();
});

// Sample dataset utilities (lightweight faker-like generator)
const sampleProfiles = [];
function randInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function pick(arr) {
    return arr[Math.floor(Math.random() * arr.length)];
}

function makeFakeProfile(i) {
    const makes = ['AutoStar', 'BlueLine', 'RiverMotor', 'ZenCar', 'Orion', 'Nimbus'];
    const models = ['A', 'B', 'C', 'X', 'GT', 'Plus'];
    const types = ['sedan', 'suv', 'hatchback', 'truck', 'coupe', 'electric'];

    const make = pick(makes);
    const model = pick(models) + (i + 1);
    const type = pick(types);

    // Generate realistic-ish ranges based on type
    let eng = 2.0;
    let wt = randInt(1100, 1800);
    let hp = randInt(90, 200);
    if (type === 'suv') { eng = +(2.4 + Math.random()).toFixed(1); wt = randInt(1600, 2300); hp = randInt(130, 260); }
    if (type === 'truck') { eng = +(3.0 + Math.random()*2).toFixed(1); wt = randInt(2000, 3500); hp = randInt(180, 400); }
    if (type === 'hatchback') { eng = +(1.2 + Math.random()).toFixed(1); wt = randInt(900, 1300); hp = randInt(70, 120); }
    if (type === 'coupe') { eng = +(1.8 + Math.random()).toFixed(1); wt = randInt(1200, 1500); hp = randInt(110, 250); }
    if (type === 'electric') { eng = 0.0; wt = randInt(1400, 2200); hp = randInt(120, 350); }

    const transmission = Math.random() > 0.5 ? 0 : 1; // 0=Auto,1=Manual

    return {
        id: `sample-${i}`,
        name: `${make} ${model}`,
        engine: eng,
        weight: wt,
        horsepower: hp,
        transmission,
        type
    };
}

function prepareSampleDatasets() {
    // Create a handful of sample profiles
    sampleProfiles.length = 0;
    for (let i = 0; i < 8; i++) sampleProfiles.push(makeFakeProfile(i));

    const sel = document.getElementById('sampleDataset');
    if (!sel) return;
    // clear existing options except placeholder
    sel.innerHTML = '<option value="">-- Select Sample --</option>';
    sampleProfiles.forEach((p, idx) => {
        const opt = document.createElement('option');
        opt.value = idx;
        opt.textContent = `${p.name} — ${p.type.toUpperCase()} (${p.engine}L, ${p.horsepower}HP)`;
        sel.appendChild(opt);
    });
}

function loadSelectedSample() {
    const sel = document.getElementById('sampleDataset');
    const idx = sel ? sel.value : '';
    if (idx === '') {
        alert('Please select a sample dataset first.');
        return;
    }
    const p = sampleProfiles[parseInt(idx)];
    if (!p) return;
    applyProfileToForm(p);
    // Run prediction automatically for convenience
    runFuelPrediction();
}

function loadRandomSample() {
    const p = pick(sampleProfiles);
    applyProfileToForm(p);
    runFuelPrediction();
}

function applyProfileToForm(p) {
    document.getElementById('engineSize').value = p.engine;
    document.getElementById('vehicleWeight').value = p.weight;
    document.getElementById('horsepower').value = p.horsepower;
    document.getElementById('transmission').value = p.transmission;
    const vt = document.getElementById('vehicleType');
    if (vt) vt.value = p.type;
}

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

// Fuel prediction modal handlers
function openFuelModal() {
    document.getElementById('fuelModal').style.display = 'flex';
    resetFuelModal();
}

function closeFuelModal() {
    document.getElementById('fuelModal').style.display = 'none';
}

function resetFuelModal() {
    document.getElementById('engineSize').value = 2.0;
    document.getElementById('vehicleWeight').value = 1500;
    document.getElementById('horsepower').value = 140;
    document.getElementById('transmission').value = 0;
    document.getElementById('predictResult').style.display = 'none';
    document.getElementById('predictProgress').style.display = 'none';
    document.getElementById('progressFill').style.width = '0%';
}

function runFuelPrediction() {
    // Read inputs (not used for real model; demo only)
    const eng = parseFloat(document.getElementById('engineSize').value) || 2.0;
    const wt = parseFloat(document.getElementById('vehicleWeight').value) || 1500;
    const hp = parseFloat(document.getElementById('horsepower').value) || 140;
    const tx = parseInt(document.getElementById('transmission').value) || 0;
    const vtype = document.getElementById('vehicleType') ? document.getElementById('vehicleType').value : 'sedan';

    // Show progress
    const progress = document.getElementById('predictProgress');
    const fill = document.getElementById('progressFill');
    progress.style.display = 'block';
    fill.style.width = '0%';
    document.getElementById('predictResult').style.display = 'none';

    let p = 0;
    const timer = setInterval(() => {
        p += Math.random() * 20;
        if (p >= 100) p = 100;
        fill.style.width = p + '%';
        if (p >= 100) {
            clearInterval(timer);
            // Compute a dummy km/l value based on inputs and vehicle type
            const typeMultiplier = (() => {
                switch (vtype) {
                    case 'suv': return 0.85;
                    case 'truck': return 0.7;
                    case 'hatchback': return 1.05;
                    case 'coupe': return 0.95;
                    case 'electric': return 1.4;
                    default: return 1.0; // sedan
                }
            })();

            const baseCity = Math.max(4, (110 - (wt / 120)) / (eng / 1.6));
            const baseHighway = Math.max(7, baseCity * 1.2 + hp / 220);
            const city = (baseCity * typeMultiplier * (tx === 1 ? 1.05 : 1.0)).toFixed(1);
            const highway = (baseHighway * typeMultiplier * (tx === 1 ? 1.03 : 1.0)).toFixed(1);
            const overall = (((parseFloat(city) + parseFloat(highway)) / 2)).toFixed(1);

            document.getElementById('fuelValue').textContent = overall + ' km/l';
            document.getElementById('cityVal').textContent = city + ' km/l';
            document.getElementById('highwayVal').textContent = highway + ' km/l';
            document.getElementById('predictResult').style.display = 'block';

            // Append or update vehicle type label
            const resultBox = document.getElementById('predictResult');
            let existing = resultBox.querySelector('.vehicle-type');
            if (!existing) {
                existing = document.createElement('div');
                existing.className = 'vehicle-type';
                existing.style.marginTop = '8px';
                resultBox.appendChild(existing);
            }
            existing.innerHTML = '<small>Vehicle Type: <strong>' + vtype.charAt(0).toUpperCase() + vtype.slice(1) + '</strong></small>';
        }
    }, 300);
}
