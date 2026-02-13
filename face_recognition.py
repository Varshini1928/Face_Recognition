import cv2
import numpy as np
import os
from PIL import Image
import pickle

class FaceRecognitionSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_data = []
        self.labels = []
        self.label_dict = {}
        self.current_id = 0
        
    def add_face_data(self, name, num_samples=50):
        """Add new face data for training"""
        print(f"\n=== Adding face data for {name} ===")
        print("Look at the camera. Press 's' to capture, 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        face_samples = []
        sample_count = 0
        
        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )
            
            for (x, y, w, h) in faces:
                sample_count += 1
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (200, 200))
                face_samples.append(face_resized)
                
                # Draw rectangle and show progress
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Samples: {sample_count}/{num_samples}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            cv2.imshow('Add Face Data', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        
        if len(face_samples) > 0:
            # Store face data
            if name not in self.label_dict.values():
                self.label_dict[self.current_id] = name
                self.current_id += 1
            
            # Add to training data
            label_id = list(self.label_dict.keys())[
                list(self.label_dict.values()).index(name)
            ]
            
            for face in face_samples:
                self.face_data.append(face)
                self.labels.append(label_id)
            
            print(f"✓ Added {len(face_samples)} samples for {name}")
            self.train_model()
        else:
            print("✗ No face data captured")
            
    def train_model(self):
        """Train the face recognition model"""
        if len(self.face_data) > 0:
            print("Training model...")
            self.recognizer.train(self.face_data, np.array(self.labels))
            print("✓ Model trained successfully")
        else:
            print("No training data available")
            
    def save_model(self, filename='face_model.yml', labels_file='labels.pkl'):
        """Save the trained model and labels"""
        if len(self.face_data) > 0:
            self.recognizer.write(filename)
            with open(labels_file, 'wb') as f:
                pickle.dump(self.label_dict, f)
            print(f"✓ Model saved to {filename}")
            
    def load_model(self, filename='face_model.yml', labels_file='labels.pkl'):
        """Load a trained model"""
        if os.path.exists(filename) and os.path.exists(labels_file):
            self.recognizer.read(filename)
            with open(labels_file, 'rb') as f:
                self.label_dict = pickle.load(f)
            print(f"✓ Model loaded from {filename}")
            print(f"Known faces: {list(self.label_dict.values())}")
            return True
        else:
            print("No saved model found")
            return False
            
    def recognize_faces(self):
        """Real-time face recognition"""
        if len(self.face_data) == 0 and not self.load_model():
            print("No model available. Please add face data first.")
            return
            
        print("\n=== Face Recognition Started ===")
        print("Press 'q' to quit, 'a' to add new face")
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )
            
            for (x, y, w, h) in faces:
                # Extract and preprocess face
                face_roi = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_roi, (200, 200))
                
                # Predict
                label_id, confidence = self.recognizer.predict(face_resized)
                
                # Get name from label
                if confidence < 100:  # Threshold for recognition
                    name = self.label_dict.get(label_id, "Unknown")
                    color = (0, 255, 0)  # Green for recognized
                    text = f"{name} ({100-confidence:.1f}%)"
                else:
                    name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown
                    text = "Unknown"
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, text, (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Display info
            cv2.putText(frame, "Press 'q' to quit, 'a' to add face", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Face Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                cap.release()
                cv2.destroyAllWindows()
                name = input("Enter name for new person: ")
                self.add_face_data(name)
                # Restart recognition
                cap = cv2.VideoCapture(0)
                
        cap.release()
        cv2.destroyAllWindows()
        
    def list_known_faces(self):
        """Display all known faces"""
        if self.label_dict:
            print("\n=== Known Faces ===")
            for id, name in self.label_dict.items():
                print(f"ID {id}: {name}")
        else:
            print("No faces in database")
            
    def delete_face(self, name):
        """Delete a face from the database"""
        if name in self.label_dict.values():
            # Find and remove label
            label_id = list(self.label_dict.keys())[
                list(self.label_dict.values()).index(name)
            ]
            del self.label_dict[label_id]
            
            # Remove all training data for this label
            indices_to_keep = [i for i, label in enumerate(self.labels) if label != label_id]
            self.face_data = [self.face_data[i] for i in indices_to_keep]
            self.labels = [self.labels[i] for i in indices_to_keep]
            
            # Retrain model
            self.train_model()
            print(f"✓ Deleted {name} from database")
        else:
            print(f"✗ {name} not found in database")

def main():
    print("=" * 50)
    print("     FACE RECOGNITION SYSTEM")
    print("=" * 50)
    
    fr_system = FaceRecognitionSystem()
    
    while True:
        print("\n=== MAIN MENU ===")
        print("1. Add new face data")
        print("2. Start face recognition")
        print("3. List known faces")
        print("4. Delete a face")
        print("5. Save model")
        print("6. Load model")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ")
        
        if choice == '1':
            name = input("Enter person's name: ")
            samples = input("Number of samples (default 50): ")
            samples = int(samples) if samples else 50
            fr_system.add_face_data(name, samples)
            
        elif choice == '2':
            fr_system.recognize_faces()
            
        elif choice == '3':
            fr_system.list_known_faces()
            
        elif choice == '4':
            name = input("Enter name to delete: ")
            fr_system.delete_face(name)
            
        elif choice == '5':
            fr_system.save_model()
            
        elif choice == '6':
            fr_system.load_model()
            
        elif choice == '7':
            print("\nThank you for using Face Recognition System!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    # Check if OpenCV is installed with face module
    try:
        cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        print("\n✗ OpenCV face module not found!")
        print("\nPlease install required packages:")
        print("pip install opencv-python")
        print("pip install opencv-contrib-python")
        print("pip install numpy pillow")
        exit()
    
    main()
