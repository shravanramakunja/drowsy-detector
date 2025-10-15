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
        try:
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
            
            # Convert to pygame sound (stereo format)
            sound_array = np.array(samples, dtype=np.int16)
            # Create stereo array by duplicating mono to 2 channels
            stereo_array = np.column_stack((sound_array, sound_array))
            sound = pygame.sndarray.make_sound(stereo_array)
            sound.play(-1)  # Loop the sound
        except Exception as e:
            print(f"[WARNING] Could not play alarm sound: {e}")
        
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
                    # Eyes are open - reset counter
                    self.frame_counter = 0
                    if self.alarm_on:
                        self.alarm_on = False
                        self.stop_alarm()
                    
                    # Draw rectangles around eyes
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    
                    status_text = "Eyes Open"
                    status_color = (0, 255, 0)
                else:
                    # Eyes not detected (possibly closed)
                    self.frame_counter += 1
                    
                    if self.frame_counter >= self.EYE_CLOSED_FRAMES:
                        if not self.alarm_on:
                            self.alarm_on = True
                            # Play alarm in separate thread
                            Thread(target=self.play_alarm, daemon=True).start()
                        
                        # Draw warning on frame
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    status_text = f"Eyes Closed ({self.frame_counter})"
                    status_color = (0, 0, 255) if self.alarm_on else (0, 165, 255)
                
                # Display eye status
                cv2.putText(frame, status_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            
            # Display overall status
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
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
