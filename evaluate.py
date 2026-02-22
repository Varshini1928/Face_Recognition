"""
evaluate.py
Evaluate recognition accuracy, generate confusion matrix and metrics report.
"""

import cv2
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def evaluate_on_dataset(system, test_dir="data/raw", split_ratio=0.2):
    """
    Evaluate the face recognition system on a labeled test directory.
    
    Args:
        system: FaceRecognitionSystem instance
        test_dir: directory with subdirectories per person
        split_ratio: fraction of images to use for testing
    """
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import random

    test_dir = Path(test_dir)
    y_true, y_pred = [], []

    logger.info(f"Evaluating on dataset: {test_dir}")

    for person_dir in sorted(test_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        true_name = person_dir.name
        img_paths = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
        random.shuffle(img_paths)
        test_images = img_paths[:max(1, int(len(img_paths) * split_ratio))]

        for img_path in test_images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = system.detector.detect(img)
            if not faces:
                y_true.append(true_name)
                y_pred.append("No Face Detected")
                continue
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            crop = img[y:y+h, x:x+w]
            embedding = system.embedder.get_embedding(crop)
            pred_name, confidence = system.db.find_match(embedding, system.threshold)
            y_true.append(true_name)
            y_pred.append(pred_name)

    if not y_true:
        logger.error("No test images processed!")
        return {}

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    labels = sorted(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    logger.info(f"\n{'='*50}")
    logger.info(f"  EVALUATION RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"  Total test samples: {len(y_true)}")
    logger.info(f"  Overall Accuracy:   {accuracy*100:.2f}%")
    logger.info(f"\n{report}")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'accuracy': round(accuracy * 100, 2),
        'total_samples': len(y_true),
        'y_true': y_true,
        'y_pred': y_pred,
        'labels': labels,
        'confusion_matrix': cm.tolist(),
        'report': report
    }

    import os
    os.makedirs("logs", exist_ok=True)
    result_path = "logs/evaluation_results.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {result_path}")

    # Plot confusion matrix
    _plot_confusion_matrix(cm, labels)
    return results


def _plot_confusion_matrix(cm, labels):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels))))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('True', fontsize=12)
        plt.tight_layout()
        plt.savefig("logs/confusion_matrix.png", dpi=150)
        logger.info("Confusion matrix saved to logs/confusion_matrix.png")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not plot confusion matrix: {e}")


def benchmark_speed(system, num_frames=100, img_size=(640, 480)):
    """Benchmark detection + recognition speed."""
    import time
    logger.info(f"Benchmarking speed on {num_frames} synthetic frames ({img_size})...")
    times_detect, times_embed = [], []

    for _ in range(num_frames):
        frame = np.random.randint(0, 255, (*img_size[::-1], 3), dtype=np.uint8)
        t0 = time.perf_counter()
        faces = system.detector.detect(frame)
        t1 = time.perf_counter()
        times_detect.append(t1 - t0)

        if faces:
            x, y, w, h = faces[0]
            crop = frame[y:y+h, x:x+w] if h > 0 and w > 0 else frame[:50, :50]
            t2 = time.perf_counter()
            system.embedder.get_embedding(crop)
            t3 = time.perf_counter()
            times_embed.append(t3 - t2)

    avg_detect = np.mean(times_detect) * 1000
    avg_embed = np.mean(times_embed) * 1000 if times_embed else 0
    total_fps = 1000 / (avg_detect + avg_embed) if (avg_detect + avg_embed) > 0 else 0

    logger.info(f"\n  Detection:  {avg_detect:.1f} ms/frame")
    logger.info(f"  Embedding:  {avg_embed:.1f} ms/frame")
    logger.info(f"  Est. FPS:   {total_fps:.1f}")
    return {'detect_ms': avg_detect, 'embed_ms': avg_embed, 'fps': total_fps}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from face_engine import FaceRecognitionSystem

    system = FaceRecognitionSystem()
    print("\n1. Evaluate on dataset")
    print("2. Benchmark speed")
    choice = input("Choose [1/2]: ").strip()

    if choice == '1':
        evaluate_on_dataset(system)
    elif choice == '2':
        benchmark_speed(system)