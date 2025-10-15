import cv2
import numpy as np
import pygame
from threading import Thread
import time

# Initialize pygame for alarm sound
pygame.mixer.init()

class DrowsinessDetector:
    def __init__(self):
        # Eye detection parameters
        self.EYE_CLOSED_THRESHOLD = 0.15  # Ratio of eye height to face height
        self.EYE_CLOSED_FRAMES = 15  # Consecutive frames to trigger alarm
        
        # Initialize counters
        self.frame_counter = 0
        self.alarm_on = False
        
        # Load Haar Cascade classifiers (comes with OpenCV)
        print("[INFO] Loading face and eye detectors...")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
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
        # Create a simple beep sound
        frequency = 2500  # Hz
        duration = 1000   # milliseconds
        
        # Generate square wave
        sample_rate = 22050
        period = int(sample_rate / frequency)
        amplitude = 4096
        
        samples = []
        for i in range(int(duration * sample_rate / 1000)):
            if i % period < period / 2:
                samples.append(amplitude)
            else:
                samples.append(-amplitude)
        
        # Convert to pygame sound
        sound_array = np.array(samples, dtype=np.int16)
        sound = pygame.sndarray.make_sound(sound_array)
        sound.play(-1)  # Loop the sound
        
    def stop_alarm(self):
        """Stop alarm sound"""
        pygame.mixer.stop()
    
    def detect_drowsiness(self):
        """Main detection loop"""
        print("[INFO] Starting video stream...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Could not open webcam")
            return
        
        print("[INFO] Camera initialized. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame")
                break
            
            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.detector(gray, 0)
            
            # Process each face
            for face in faces:
                # Get facial landmarks
                shape = self.predictor(gray, face)
                shape = self.shape_to_numpy(shape)
                
                # Extract eye coordinates
                left_eye = shape[self.LEFT_EYE_START:self.LEFT_EYE_END]
                right_eye = shape[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
                
                # Calculate EAR for both eyes
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                
                # Average EAR for both eyes
                ear = (left_ear + right_ear) / 2.0
                
                # Draw eye contours
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
                
                # Check if EAR is below threshold
                if ear < self.EAR_THRESHOLD:
                    self.frame_counter += 1
                    
                    # If eyes closed for sufficient frames, trigger alarm
                    if self.frame_counter >= self.EAR_CONSEC_FRAMES:
                        if not self.alarm_on:
                            self.alarm_on = True
                            # Play alarm in separate thread
                            Thread(target=self.play_alarm, daemon=True).start()
                        
                        # Draw warning on frame
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.frame_counter = 0
                    if self.alarm_on:
                        self.alarm_on = False
                        self.stop_alarm()
                
                # Display EAR value
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Draw face rectangle
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Display status
            status = "ALERT!" if self.alarm_on else "Active"
            color = (0, 0, 255) if self.alarm_on else (0, 255, 0)
            cv2.putText(frame, f"Status: {status}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Show frame
            cv2.imshow("Drowsiness Detection", frame)
            
            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        print("[INFO] Cleaning up...")
        self.stop_alarm()
        cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

def main():
    print("=" * 50)
    print("Real-Time Drowsiness Detection System")
    print("=" * 50)
    print("\nInstructions:")
    print("1. Make sure you have good lighting")
    print("2. Position your face clearly in front of the camera")
    print("3. The system will alert you if drowsiness is detected")
    print("4. Press 'q' to quit\n")
    
    try:
        detector = DrowsinessDetector()
        detector.detect_drowsiness()
    except FileNotFoundError:
        print("\n[ERROR] Could not find 'shape_predictor_68_face_landmarks.dat'")
        print("Please download it from:")
        print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("Extract and place it in the same directory as this script.")
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
