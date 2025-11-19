from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import pygame
from threading import Thread
import time
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import random

# Get the directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, 'web')

# Initialize Flask app
app = Flask(__name__, 
            template_folder=WEB_DIR,
            static_folder=WEB_DIR,
            static_url_path='')
CORS(app)

# Initialize pygame for alarm
pygame.mixer.init()

class DrowsinessDetector:
    def __init__(self):
        # Eye detection parameters
        self.EYE_CLOSED_THRESHOLD = 0.15
        self.EYE_CLOSED_SECONDS = 5
        self.ASSUMED_FPS = 30
        self.EYE_CLOSED_FRAMES = self.EYE_CLOSED_SECONDS * self.ASSUMED_FPS
        
        # Initialize counters
        self.frame_counter = 0
        self.alarm_on = False
        self.start_time = None
        self.eyes_closed_time = 0
        self.eye_state = "N/A"
        self.status = "Not Started"
        self.camera = None
        self.is_running = False
        
        # Load Haar Cascade classifiers
        print("[INFO] Loading face and eye detectors...")
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise Exception("Error loading cascade classifiers")
    
    def detect_eyes(self, face_gray, face_region):
        """Detect eyes within a face region"""
        eyes = self.eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        return eyes
    
    def play_alarm(self):
        """Play alarm sound"""
        try:
            frequency = 1500
            duration = 500
            sample_rate = 22050
            period = int(sample_rate / frequency)
            amplitude = 8192
            
            samples = []
            for i in range(int(duration * sample_rate / 1000)):
                if i % period < period / 2:
                    samples.append(amplitude)
                else:
                    samples.append(-amplitude)
            
            sound_array = np.array(samples, dtype=np.int16)
            stereo_array = np.column_stack((sound_array, sound_array))
            sound = pygame.sndarray.make_sound(stereo_array)
            sound.set_volume(1.0)
            sound.play(-1)
            print("[ALARM] Playing alarm sound!")
        except Exception as e:
            print(f"[WARNING] Could not play alarm sound: {e}")
    
    def stop_alarm(self):
        """Stop alarm sound"""
        pygame.mixer.stop()
    
    def start_camera(self, camera_index=0):
        """Start the camera"""
        if self.camera is not None:
            return True
        
        for index in [camera_index, 0, 1, 2]:
            print(f"[INFO] Trying camera index {index}...")
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.camera = cap
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"[INFO] Successfully opened camera {index}")
                self.is_running = True
                self.status = "Active"
                return True
            cap.release()
        
        print("[ERROR] Could not open any camera")
        return False
    
    def stop_camera(self):
        """Stop the camera"""
        self.is_running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.stop_alarm()
        self.status = "Stopped"
        self.eye_state = "N/A"
        self.eyes_closed_time = 0
        print("[INFO] Camera stopped")
    
    def process_frame(self, frame):
        """Process a single frame for drowsiness detection"""
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        # Process each face
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Region of interest for eyes (upper half of face)
            roi_gray = gray[y:y + int(h * 0.6), x:x + w]
            roi_color = frame[y:y + int(h * 0.6), x:x + w]
            
            # Detect eyes
            eyes = self.detect_eyes(roi_gray, (x, y, w, int(h * 0.6)))
            
            # Check if eyes are detected
            if len(eyes) >= 2:
                # Eyes are open
                self.frame_counter = 0
                self.eyes_closed_time = 0
                self.start_time = None
                
                if self.alarm_on:
                    self.alarm_on = False
                    self.stop_alarm()
                
                # Draw rectangles around eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                
                self.eye_state = "Open"
                status_color = (0, 255, 0)
            else:
                # Eyes not detected (possibly closed)
                if self.start_time is None:
                    self.start_time = time.time()
                
                # Calculate how long eyes have been closed
                self.eyes_closed_time = time.time() - self.start_time
                self.frame_counter += 1
                
                # Trigger alarm if eyes closed for more than threshold
                if self.eyes_closed_time >= self.EYE_CLOSED_SECONDS:
                    if not self.alarm_on:
                        self.alarm_on = True
                        Thread(target=self.play_alarm, daemon=True).start()
                    
                    # Draw warning on frame
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                    status_color = (0, 0, 255)
                else:
                    status_color = (0, 165, 255)
                
                self.eye_state = f"Closed ({self.eyes_closed_time:.1f}s)"
        
        # Display status on frame
        status_text = "ALERT!" if self.alarm_on else "Active"
        color = (0, 0, 255) if self.alarm_on else (0, 255, 0)
        cv2.putText(frame, f"Status: {status_text}", (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display eye state
        cv2.putText(frame, f"Eyes: {self.eye_state}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def generate_frames(self):
        """Generate frames for video streaming"""
        while self.is_running:
            if self.camera is None:
                break
            
            success, frame = self.camera.read()
            if not success:
                print("[ERROR] Failed to read frame")
                break
            
            # Process frame
            frame = self.process_frame(frame)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            frame = buffer.tobytes()
            
            # Yield frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    def get_status_data(self):
        """Get current status data"""
        return {
            'status': self.status,
            'eye_state': self.eye_state,
            'timer': round(self.eyes_closed_time, 1),
            'alert': self.alarm_on,
            'threshold': self.EYE_CLOSED_SECONDS
        }
    
    def update_threshold(self, seconds):
        """Update the alert threshold"""
        self.EYE_CLOSED_SECONDS = seconds
        print(f"[INFO] Threshold updated to {seconds} seconds")

# Create global detector instance
detector = DrowsinessDetector()

# ==================== FUEL EFFICIENCY PREDICTOR ====================

class FuelEfficiencyPredictor:
    def __init__(self):
        self.model_city = None
        self.model_highway = None
        self.model_combined = None
        self.is_trained = False
        self.label_encoders = {}
        self.vehicle_type_mapping = {
            'sedan': 'Midsize Cars',
            'suv': 'Sport Utility Vehicle',
            'hatchback': 'Compact Cars',
            'truck': 'Standard Pickup Trucks',
            'coupe': 'Two Seaters',
            'electric': 'Midsize Cars'
        }
        self.train_model()
    
    def load_and_preprocess_data(self):
        """Load fuel.csv dataset and preprocess for training"""
        print("[INFO] Loading dataset from fuel.csv...")
        
        try:
            dataset_path = os.path.join(BASE_DIR, 'dataset', 'fuel.csv')
            df = pd.read_csv(dataset_path)
            
            print(f"[INFO] Loaded {len(df)} records from dataset")
            
            # Select relevant features and clean data
            df_clean = df[[
                'engine_cylinders',
                'engine_displacement',
                'transmission_type',
                'class',
                'drive',
                'fuel_type',
                'city_mpg_ft1',
                'highway_mpg_ft1',
                'miles_per_gallon'
            ]].copy()
            
            # Remove rows with missing critical values
            df_clean = df_clean.dropna(subset=['city_mpg_ft1', 'highway_mpg_ft1', 'miles_per_gallon'])
            df_clean = df_clean[df_clean['city_mpg_ft1'] > 0]
            df_clean = df_clean[df_clean['highway_mpg_ft1'] > 0]
            
            # Convert MPG to km/l (1 MPG ≈ 0.425 km/l)
            df_clean['city_kml'] = df_clean['city_mpg_ft1'] * 0.425144
            df_clean['highway_kml'] = df_clean['highway_mpg_ft1'] * 0.425144
            df_clean['combined_kml'] = df_clean['miles_per_gallon'] * 0.425144
            
            # Fill missing values
            df_clean['engine_cylinders'] = df_clean['engine_cylinders'].fillna(4)
            df_clean['engine_displacement'] = df_clean['engine_displacement'].fillna(2.0)
            df_clean['transmission_type'] = df_clean['transmission_type'].fillna('Automatic')
            df_clean['class'] = df_clean['class'].fillna('Midsize Cars')
            df_clean['drive'] = df_clean['drive'].fillna('2-Wheel Drive')
            df_clean['fuel_type'] = df_clean['fuel_type'].fillna('Regular')
            
            print(f"[INFO] Preprocessed {len(df_clean)} valid records")
            return df_clean
            
        except Exception as e:
            print(f"[ERROR] Failed to load dataset: {e}")
            print("[INFO] Falling back to synthetic data generation...")
            return None
    
    def train_model(self):
        """Train Random Forest models using real dataset"""
        print("[INFO] Training Random Forest models...")
        
        # Load real dataset
        df = self.load_and_preprocess_data()
        
        if df is None or len(df) < 100:
            print("[WARNING] Insufficient data for training")
            self.is_trained = False
            return
        
        # Prepare features
        # Encode categorical variables
        self.label_encoders['transmission'] = LabelEncoder()
        self.label_encoders['class'] = LabelEncoder()
        self.label_encoders['drive'] = LabelEncoder()
        self.label_encoders['fuel_type'] = LabelEncoder()
        
        df['transmission_encoded'] = self.label_encoders['transmission'].fit_transform(
            df['transmission_type'].astype(str)
        )
        df['class_encoded'] = self.label_encoders['class'].fit_transform(
            df['class'].astype(str)
        )
        df['drive_encoded'] = self.label_encoders['drive'].fit_transform(
            df['drive'].astype(str)
        )
        df['fuel_encoded'] = self.label_encoders['fuel_type'].fit_transform(
            df['fuel_type'].astype(str)
        )
        
        # Feature matrix
        X = df[[
            'engine_cylinders',
            'engine_displacement',
            'transmission_encoded',
            'class_encoded',
            'drive_encoded',
            'fuel_encoded'
        ]].values
        
        # Target variables
        y_city = df['city_kml'].values
        y_highway = df['highway_kml'].values
        y_combined = df['combined_kml'].values
        
        # Train models
        print("[INFO] Training City MPG model...")
        self.model_city = RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model_city.fit(X, y_city)
        
        print("[INFO] Training Highway MPG model...")
        self.model_highway = RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model_highway.fit(X, y_highway)
        
        print("[INFO] Training Combined MPG model...")
        self.model_combined = RandomForestRegressor(
            n_estimators=150,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model_combined.fit(X, y_combined)
        
        self.is_trained = True
        print(f"[INFO] Models trained successfully on {len(df)} samples!")
        print(f"[INFO] Feature importance (Combined): {self.model_combined.feature_importances_}")
    
    def predict(self, engine, weight, horsepower, transmission, vehicle_type):
        """Predict fuel efficiency for given vehicle parameters"""
        if not self.is_trained:
            return None
        
        try:
            # Map frontend vehicle type to dataset class
            vehicle_class = self.vehicle_type_mapping.get(vehicle_type, 'Midsize Cars')
            
            # Estimate cylinders from engine displacement
            cylinders = 4
            if engine >= 4.0:
                cylinders = 8
            elif engine >= 2.5:
                cylinders = 6
            
            # Determine drive type (simplified)
            drive = '2-Wheel Drive'
            if vehicle_type in ['suv', 'truck']:
                drive = '4-Wheel or All-Wheel Drive'
            
            # Fuel type
            fuel_type = 'Electricity' if vehicle_type == 'electric' else 'Regular'
            
            # Transmission type
            trans_type = 'Manual' if transmission == 1 else 'Automatic'
            
            # Encode categorical features
            try:
                trans_encoded = self.label_encoders['transmission'].transform([trans_type])[0]
            except:
                trans_encoded = 0
            
            try:
                class_encoded = self.label_encoders['class'].transform([vehicle_class])[0]
            except:
                class_encoded = 0
            
            try:
                drive_encoded = self.label_encoders['drive'].transform([drive])[0]
            except:
                drive_encoded = 0
            
            try:
                fuel_encoded = self.label_encoders['fuel_type'].transform([fuel_type])[0]
            except:
                fuel_encoded = 0
            
            # Create feature vector
            features = np.array([[
                cylinders,
                engine,
                trans_encoded,
                class_encoded,
                drive_encoded,
                fuel_encoded
            ]])
            
            # Predict
            city = float(self.model_city.predict(features)[0])
            highway = float(self.model_highway.predict(features)[0])
            combined = float(self.model_combined.predict(features)[0])
            
            # Ensure realistic values
            city = max(4.0, min(40.0, city))
            highway = max(5.0, min(50.0, highway))
            combined = max(4.5, min(45.0, combined))
            
            return {
                'city': round(city, 1),
                'highway': round(highway, 1),
                'overall': round(combined, 1)
            }
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return None

# Initialize fuel predictor
print("\n" + "="*60)
print("Fuel Efficiency Predictor - Machine Learning Module")
print("="*60)
fuel_predictor = FuelEfficiencyPredictor()
print("="*60 + "\n")

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory(WEB_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory(WEB_DIR, path)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(detector.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_detection():
    """Start the detection"""
    try:
        if detector.start_camera():
            return jsonify({'success': True, 'message': 'Detection started'})
        else:
            return jsonify({'success': False, 'message': 'Failed to start camera'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/stop', methods=['POST'])
def stop_detection():
    """Stop the detection"""
    try:
        detector.stop_camera()
        return jsonify({'success': True, 'message': 'Detection stopped'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/status')
def get_status():
    """Get current status"""
    return jsonify(detector.get_status_data())

@app.route('/update_threshold', methods=['POST'])
def update_threshold():
    """Update the alert threshold"""
    try:
        data = request.get_json()
        threshold = int(data.get('threshold', 5))
        detector.update_threshold(threshold)
        return jsonify({'success': True, 'threshold': threshold})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/predict_fuel', methods=['POST'])
def predict_fuel_efficiency():
    """Predict fuel efficiency using Random Forest model"""
    try:
        data = request.get_json()
        
        engine = float(data.get('engine', 2.0))
        weight = int(data.get('weight', 1500))
        horsepower = int(data.get('horsepower', 140))
        transmission = int(data.get('transmission', 0))
        vehicle_type = data.get('vehicle_type', 'sedan')
        
        # Predict using ML model
        prediction = fuel_predictor.predict(engine, weight, horsepower, transmission, vehicle_type)
        
        if prediction:
            return jsonify({
                'success': True,
                'prediction': prediction,
                'model': 'Random Forest',
                'message': 'Prediction completed successfully'
            })
        else:
            return jsonify({'success': False, 'message': 'Model not trained'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Smart Driver Hub - Drowsiness Detection Backend")
    print("=" * 60)
    print("\n[INFO] Starting Flask server...")
    print("[INFO] Access the web interface at: http://localhost:5000")
    print("[INFO] Press Ctrl+C to stop the server\n")
    
    try:
        # Disable the debug reloader to avoid watching site-packages (watchdog restarts)
        app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down server...")
        detector.stop_camera()
        pygame.quit()
