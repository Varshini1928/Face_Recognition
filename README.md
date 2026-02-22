# 🎭 Face Recognition System

> CNN-based real-time face recognition with webcam support, web dashboard, and model training pipeline.

---

## 📁 Project Structure

```
face_recognition_system/
├── main.py                     ← CLI entry point (start here)
├── web_app.py                  ← Flask web dashboard
├── requirements.txt
│
├── src/
│   ├── face_engine.py          ← Core: Detector + Embedder + Database + System
│   ├── train_model.py          ← CNN training pipeline
│   └── evaluate.py             ← Accuracy metrics + confusion matrix
│
├── data/
│   ├── raw/                    ← Training images (one folder per person)
│   │   ├── Alice/
│   │   │   ├── photo1.jpg
│   │   │   └── photo2.jpg
│   │   └── Bob/
│   │       └── photo1.jpg
│   ├── processed/              ← Auto-generated preprocessed data
│   └── embeddings/
│       └── face_db.pkl         ← Live recognition database (auto-created)
│
├── models/
│   ├── face_embedder.h5        ← Trained CNN weights (auto-saved after training)
│   └── label_map.pkl           ← Class label mapping
│
├── logs/
│   ├── training_history.png    ← Accuracy/Loss curves
│   ├── confusion_matrix.png    ← Evaluation matrix
│   └── recognition_log_*.txt   ← Live session logs
│
└── web/
    └── templates/
        └── index.html          ← Web dashboard UI
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run interactive CLI
```bash
python main.py
```

### 3. Or launch the web dashboard directly
```bash
python web_app.py
# Then open http://localhost:5000
```

---

## 🚀 Command-Line Flags

```bash
# Live webcam recognition (no menu)
python main.py --live

# Recognize faces in an image file
python main.py --image path/to/photo.jpg

# Register a person (interactive flow)
python main.py --register

# Launch web dashboard
python main.py --web

# Custom threshold and detector
python main.py --threshold 0.65 --detector haar
```

---

## 🧠 How It Works

### Pipeline Overview
```
Camera / Image
      │
      ▼
 Face Detector          ← OpenCV Haar Cascade or DNN SSD
      │ (bounding boxes)
      ▼
 Face Embedder          ← Lightweight CNN → 128-d L2 vector
      │ (embedding)
      ▼
 Cosine Similarity      ← Compare against database embeddings
      │
      ▼
 Identity + Confidence
```

### Components

| Component | Description |
|-----------|-------------|
| `FaceDetector` | Detects face bounding boxes via Haar Cascade (fast) or DNN SSD (more accurate) |
| `FaceEmbedder` | CNN that maps a face crop → 128-d L2-normalized embedding vector |
| `FaceDatabase` | Pickle-backed store of {name: [embeddings]}; cosine similarity matching |
| `FaceRecognitionSystem` | Top-level facade wiring all three components |

---

## 🏋️ Training Your Own Model

### 1. Prepare data
```
data/raw/
    Alice/   ← 20–50 face photos per person (frontal, varied lighting)
    Bob/
    Carol/
```

### 2. Train
```bash
python src/train_model.py --data_dir data/raw --epochs 40
```

Or from the CLI menu → option **[7]**.

The best model is auto-saved to `models/face_embedder.h5`.

### 3. Tips for high accuracy
- Use **20+ images per person** with varied lighting, angles, and expressions
- Images should be **cropped close to the face** (not full body shots)
- Run training with `--epochs 50` for larger datasets
- The system includes automatic augmentation (flip, brightness, rotation) for small datasets

---

## 🌐 Web Dashboard Features

| Tab | Feature |
|-----|---------|
| **Dashboard** | System stats overview |
| **Live Feed** | MJPEG webcam stream with real-time recognition overlay |
| **Recognize** | Upload any image → get annotated result with confidence scores |
| **Register** | Add persons via file upload or webcam snapshot |
| **Manage** | View and delete registered persons |

---

## 📊 Evaluation

Run accuracy evaluation on your labeled dataset:

```bash
python src/evaluate.py
```

Outputs:
- Per-class precision / recall / F1
- Overall accuracy
- Confusion matrix PNG (`logs/confusion_matrix.png`)
- JSON report (`logs/evaluation_results.json`)

---

## 🔧 Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recognition_threshold` | `0.6` | Cosine similarity cutoff (lower = stricter) |
| `detector_method` | `haar` | `'haar'` or `'dnn'` |
| `num_samples` | `10` | Webcam registration captures |
| `embedding_size` | `128` | CNN output dimension |

Adjust threshold:
```python
from src.face_engine import FaceRecognitionSystem
frs = FaceRecognitionSystem(recognition_threshold=0.65)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| Face Detection | OpenCV (Haar Cascade / DNN SSD) |
| Deep Learning | TensorFlow / Keras CNN |
| Embeddings | L2-normalized 128-d cosine space |
| Web Backend | Flask |
| Frontend | Vanilla JS + CSS3 |
| Storage | Pickle (face DB), HDF5 (model weights) |

---

## 📈 Expected Performance

| Dataset Size | Expected Accuracy |
|-------------|-------------------|
| 5 persons × 10 imgs | ~75–82% |
| 10 persons × 20 imgs | ~82–88% |
| 20 persons × 30+ imgs | ~87–93% |

*Accuracy improves significantly with more diverse training images.*

---

## 🐞 Troubleshooting

**No camera detected:**
```bash
# Test camera index
python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened())"
```

**TensorFlow not found:**  
The system gracefully falls back to a lightweight LBPH-style mode. For full CNN support:
```bash
pip install tensorflow  # GPU
# or
pip install tensorflow-cpu  # CPU only
```

**Low accuracy:**
- Ensure face images are well-lit and front-facing
- Add more training samples per person
- Lower the threshold slightly: `--threshold 0.55`