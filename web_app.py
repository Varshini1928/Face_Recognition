"""
web_app.py
Flask web dashboard for Face Recognition System.
Provides REST API + a browser UI for registration, live feed, and management.
"""

import os
import cv2
import base64
import json
import logging
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from io import BytesIO

from flask import (Flask, render_template, request, jsonify,
                   Response, send_from_directory)
from werkzeug.utils import secure_filename

# Add src directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from face_engine import FaceRecognitionSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
app.config['UPLOAD_FOLDER'] = 'data/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global system instance
frs = FaceRecognitionSystem(detector_method='haar', recognition_threshold=0.6)

# Camera state
camera_lock = threading.Lock()
camera = None
camera_active = False


# ------------------------------------------------------------------ #
#  Camera Management                                                   #
# ------------------------------------------------------------------ #

def get_camera():
    global camera, camera_active
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            camera_active = camera.isOpened()
    return camera


def release_camera():
    global camera, camera_active
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            camera_active = False


def generate_frames():
    """Video stream generator for MJPEG."""
    cap = get_camera()
    while True:
        with camera_lock:
            if cap is None or not cap.isOpened():
                break
            ret, frame = cap.read()
        if not ret:
            break
        annotated, results = frs._recognize_frame(frame)
        _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


# ------------------------------------------------------------------ #
#  Routes — Pages                                                      #
# ------------------------------------------------------------------ #

@app.route('/')
def index():
    return render_template('index.html')


# ------------------------------------------------------------------ #
#  Routes — Video                                                      #
# ------------------------------------------------------------------ #

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    cam = get_camera()
    if cam and cam.isOpened():
        return jsonify({'status': 'ok', 'message': 'Camera started'})
    return jsonify({'status': 'error', 'message': 'Cannot open camera'}), 500


@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    release_camera()
    return jsonify({'status': 'ok', 'message': 'Camera stopped'})


# ------------------------------------------------------------------ #
#  Routes — Recognition                                               #
# ------------------------------------------------------------------ #

@app.route('/api/recognize', methods=['POST'])
def recognize():
    """Recognize faces in an uploaded image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    annotated, results = frs.recognize_image(filepath)
    if annotated is None:
        return jsonify({'error': 'Could not process image'}), 400

    _, buffer = cv2.imencode('.jpg', annotated)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'status': 'ok',
        'image': img_b64,
        'faces': results,
        'count': len(results)
    })


@app.route('/api/recognize/base64', methods=['POST'])
def recognize_base64():
    """Recognize faces from a base64-encoded image (e.g., webcam snapshot)."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    img_data = data['image'].split(',')[-1]  # strip data:image/...;base64, prefix
    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Invalid image data'}), 400

    annotated, results = frs._recognize_frame(frame)
    _, buffer = cv2.imencode('.jpg', annotated)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'status': 'ok',
        'image': 'data:image/jpeg;base64,' + img_b64,
        'faces': results,
        'count': len(results)
    })


# ------------------------------------------------------------------ #
#  Routes — Registration                                              #
# ------------------------------------------------------------------ #

@app.route('/api/register', methods=['POST'])
def register():
    """Register a person from uploaded image(s)."""
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({'error': 'Name is required'}), 400

    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'No images provided'}), 400

    success_count = 0
    errors = []

    for file in files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        ok, msg = frs.register_from_image(filepath, name)
        if ok:
            success_count += 1
        else:
            errors.append(msg)

    if success_count > 0:
        return jsonify({
            'status': 'ok',
            'message': f"Registered '{name}' with {success_count} image(s)",
            'errors': errors
        })
    return jsonify({'status': 'error', 'errors': errors}), 400


@app.route('/api/register/snapshot', methods=['POST'])
def register_snapshot():
    """Register a person from a webcam snapshot (base64)."""
    data = request.get_json()
    name = data.get('name', '').strip()
    img_b64 = data.get('image', '')

    if not name or not img_b64:
        return jsonify({'error': 'Name and image required'}), 400

    img_data = img_b64.split(',')[-1]
    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Invalid image'}), 400

    faces = frs.detector.detect(frame)
    if not faces:
        return jsonify({'error': 'No face detected in snapshot'}), 400
    if len(faces) > 1:
        return jsonify({'error': 'Multiple faces detected'}), 400

    x, y, w, h = faces[0]
    crop = frame[y:y+h, x:x+w]
    embedding = frs.embedder.get_embedding(crop)
    frs.db.add_face(name, embedding)

    return jsonify({'status': 'ok', 'message': f"Snapshot registered for '{name}'"})


# ------------------------------------------------------------------ #
#  Routes — Database Management                                       #
# ------------------------------------------------------------------ #

@app.route('/api/persons', methods=['GET'])
def list_persons():
    names = frs.db.get_all_names()
    persons = [{
        'name': n,
        'samples': len(frs.db.database[n])
    } for n in names]
    return jsonify({'persons': persons, 'count': len(persons)})


@app.route('/api/persons/<name>', methods=['DELETE'])
def delete_person(name):
    ok = frs.delete_person(name)
    if ok:
        return jsonify({'status': 'ok', 'message': f"Deleted '{name}'"})
    return jsonify({'status': 'error', 'message': f"'{name}' not found"}), 404


@app.route('/api/stats', methods=['GET'])
def stats():
    names = frs.db.get_all_names()
    total_samples = sum(len(frs.db.database[n]) for n in names)
    return jsonify({
        'registered_persons': len(names),
        'total_samples': total_samples,
        'threshold': frs.threshold,
        'detector': frs.detector.method
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Face Recognition Web App on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)