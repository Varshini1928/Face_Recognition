"""
Face Recognition Engine
Core module for face detection, embedding extraction, and identification.
"""

import os
import cv2
import numpy as np
import pickle
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class FaceDetector:
    """Handles face detection using OpenCV's DNN or Haar Cascade."""

    def __init__(self, method='dnn'):
        self.method = method
        self._load_detector()

    def _load_detector(self):
        if self.method == 'dnn':
            prototxt = Path("models/deploy.prototxt")
            caffemodel = Path("models/res10_300x300_ssd_iter_140000.caffemodel")
            if prototxt.exists() and caffemodel.exists():
                self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
                logger.info("Loaded DNN face detector")
            else:
                logger.warning("DNN model files not found, falling back to Haar Cascade")
                self.method = 'haar'
                self._load_haar()
        else:
            self._load_haar()

    def _load_haar(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.cascade = cv2.CascadeClassifier(cascade_path)
        logger.info("Loaded Haar Cascade face detector")

    def detect(self, frame, conf_threshold=0.5):
        """Detect faces in a frame. Returns list of (x, y, w, h) bounding boxes."""
        if self.method == 'dnn':
            return self._detect_dnn(frame, conf_threshold)
        return self._detect_haar(frame)

    def _detect_dnn(self, frame, conf_threshold):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                      (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces

    def _detect_haar(self, frame):
        # Guard: frame must be large enough for detection to work
        h, w = frame.shape[:2]
        if h < 60 or w < 60:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            faces = self.cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                maxSize=(max(31, w - 1), max(31, h - 1))
            )
        except cv2.error:
            return []
        return list(faces) if len(faces) > 0 else []


class FaceEmbedder:
    """
    Extracts 128-d face embeddings using a lightweight CNN.
    Uses OpenCV's face recognizer or a pretrained FaceNet-style model.
    """

    EMBEDDING_SIZE = 128

    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        """
        Build a lightweight CNN for face embedding.
        In production, swap with FaceNet / ArcFace weights.
        """
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, Model, Input

            inp = Input(shape=(96, 96, 3))
            x = layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D()(x)

            x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D()(x)

            x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D()(x)

            x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.GlobalAveragePooling2D()(x)

            x = layers.Dense(512, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(self.EMBEDDING_SIZE)(x)
            x = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)

            model = Model(inp, x, name='FaceEmbedder')
            model_path = Path("models/face_embedder.h5")
            if model_path.exists():
                model.load_weights(str(model_path))
                logger.info("Loaded pretrained embedder weights")
            else:
                logger.warning("No pretrained weights found — embeddings will be random until trained")
            return model
        except ImportError:
            logger.warning("TensorFlow not available — using LBPH fallback")
            return None

    def preprocess(self, face_img):
        """Resize and normalize a face crop."""
        face = cv2.resize(face_img, (96, 96))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        face = (face - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return np.expand_dims(face, axis=0)

    def get_embedding(self, face_img):
        """Return a 128-d L2-normalized embedding vector."""
        if self.model is None:
            # LBPH fallback — returns a dummy embedding
            gray = cv2.cvtColor(cv2.resize(face_img, (96, 96)), cv2.COLOR_BGR2GRAY)
            return gray.flatten()[:self.EMBEDDING_SIZE].astype(np.float32)
        processed = self.preprocess(face_img)
        embedding = self.model.predict(processed, verbose=0)[0]
        return embedding


class FaceDatabase:
    """Stores and retrieves face embeddings with person identities."""

    def __init__(self, db_path="data/embeddings/face_db.pkl"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.database = {}   # {name: [embedding1, embedding2, ...]}
        self._load()

    def _load(self):
        if self.db_path.exists():
            with open(self.db_path, 'rb') as f:
                self.database = pickle.load(f)
            total = sum(len(v) for v in self.database.values())
            logger.info(f"Loaded face DB: {len(self.database)} identities, {total} embeddings")

    def save(self):
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.database, f)
        logger.info(f"Saved face DB to {self.db_path}")

    def add_face(self, name, embedding):
        name = name.strip().title()
        if name not in self.database:
            self.database[name] = []
        self.database[name].append(embedding)
        self.save()
        logger.info(f"Added embedding for '{name}' (total: {len(self.database[name])})")

    def remove_person(self, name):
        name = name.strip().title()
        if name in self.database:
            del self.database[name]
            self.save()
            return True
        return False

    def get_all_names(self):
        return list(self.database.keys())

    def find_match(self, embedding, threshold=0.6):
        """
        Compare embedding against DB using cosine similarity.
        Returns (name, confidence) or ('Unknown', 0.0).
        """
        best_name = "Unknown"
        best_sim = 0.0

        for name, embeddings in self.database.items():
            sims = [self._cosine_similarity(embedding, e) for e in embeddings]
            avg_sim = float(np.mean(sims))
            if avg_sim > best_sim:
                best_sim = avg_sim
                best_name = name

        if best_sim < threshold:
            return "Unknown", best_sim
        return best_name, best_sim

    @staticmethod
    def _cosine_similarity(a, b):
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class FaceRecognitionSystem:
    """
    Top-level facade that wires detector + embedder + database together.
    Use this class for all operations.
    """

    def __init__(self, detector_method='haar', recognition_threshold=0.6):
        logger.info("Initializing Face Recognition System...")
        self.detector = FaceDetector(method=detector_method)
        self.embedder = FaceEmbedder()
        self.db = FaceDatabase()
        self.threshold = recognition_threshold
        self.recognition_log = []
        logger.info("System ready.")

    # ------------------------------------------------------------------ #
    #  Registration                                                        #
    # ------------------------------------------------------------------ #

    def register_from_image(self, image_path, name):
        """Register a person from a single image file."""
        img = cv2.imread(str(image_path))
        if img is None:
            return False, f"Cannot read image: {image_path}"
        faces = self.detector.detect(img)
        if not faces:
            return False, "No face detected in image"
        if len(faces) > 1:
            return False, f"Multiple faces ({len(faces)}) detected — use an image with one person"
        x, y, w, h = faces[0]
        crop = img[y:y+h, x:x+w]
        embedding = self.embedder.get_embedding(crop)
        self.db.add_face(name, embedding)
        return True, f"Registered '{name}' successfully"

    def register_from_webcam(self, name, num_samples=10, camera_id=0):
        """Capture multiple frames from webcam to register a person."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return False, "Cannot open camera"

        captured = 0
        logger.info(f"Capturing {num_samples} samples for '{name}'. Press 'q' to quit early.")

        while captured < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            faces = self.detector.detect(frame)
            display = frame.copy()

            for (x, y, w, h) in faces:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(display, f"Capturing {captured}/{num_samples}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if len(faces) == 1:
                    crop = frame[y:y+h, x:x+w]
                    embedding = self.embedder.get_embedding(crop)
                    self.db.add_face(name, embedding)
                    captured += 1

            cv2.putText(display, f"Registering: {name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
            cv2.imshow("Face Registration", display)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return True, f"Registered '{name}' with {captured} samples"

    # ------------------------------------------------------------------ #
    #  Recognition                                                         #
    # ------------------------------------------------------------------ #

    def recognize_image(self, image_path):
        """Recognize faces in a single image. Returns annotated frame + results."""
        img = cv2.imread(str(image_path))
        if img is None:
            return None, []
        return self._recognize_frame(img)

    def _recognize_frame(self, frame):
        faces = self.detector.detect(frame)
        results = []
        for (x, y, w, h) in faces:
            # Cast to native Python int — numpy int32 is not JSON serializable
            x, y, w, h = int(x), int(y), int(w), int(h)
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                continue
            embedding = self.embedder.get_embedding(crop)
            name, confidence = self.db.find_match(embedding, self.threshold)
            results.append({
                'name': name,
                'confidence': round(float(confidence) * 100, 1),
                'bbox': (x, y, w, h),
                'timestamp': datetime.now().isoformat()
            })
            # Draw annotation
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{name} ({confidence*100:.1f}%)"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        return frame, results

    def run_live_recognition(self, camera_id=0, save_log=True):
        """Run real-time face recognition from webcam. Press 'q' to quit."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error("Cannot open camera")
            return

        logger.info("Starting live recognition. Press 'q' to quit, 's' to screenshot.")
        frame_count = 0
        fps = 0
        prev_time = datetime.now()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            annotated, results = self._recognize_frame(frame)

            # FPS counter
            now = datetime.now()
            elapsed = (now - prev_time).total_seconds()
            if elapsed > 0:
                fps = 1.0 / elapsed
            prev_time = now

            if save_log:
                self.recognition_log.extend(results)

            # HUD
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(annotated, f"Faces: {len(results)}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            known = [r['name'] for r in results if r['name'] != 'Unknown']
            if known:
                cv2.putText(annotated, "Detected: " + ", ".join(set(known)),
                            (10, annotated.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Face Recognition System", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = f"logs/screenshot_{ts}.jpg"
                os.makedirs("logs", exist_ok=True)
                cv2.imwrite(path, annotated)
                logger.info(f"Screenshot saved: {path}")

        cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Session ended. Processed {frame_count} frames.")
        if save_log:
            self._save_log()

    def _save_log(self):
        if not self.recognition_log:
            return
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"logs/recognition_log_{ts}.txt"
        with open(path, 'w') as f:
            for entry in self.recognition_log:
                f.write(f"{entry['timestamp']} | {entry['name']} | {entry['confidence']}% | bbox={entry['bbox']}\n")
        logger.info(f"Recognition log saved: {path}")

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def list_registered(self):
        names = self.db.get_all_names()
        if not names:
            print("No persons registered yet.")
        else:
            print(f"\n{'='*40}")
            print(f"  Registered Persons ({len(names)} total)")
            print(f"{'='*40}")
            for name in names:
                count = len(self.db.database[name])
                print(f"  • {name:<25} ({count} sample{'s' if count > 1 else ''})")
            print(f"{'='*40}\n")

    def delete_person(self, name):
        ok = self.db.remove_person(name)
        msg = f"Deleted '{name}'" if ok else f"'{name}' not found in database"
        logger.info(msg)
        return ok
