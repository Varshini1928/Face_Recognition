"""
train_model.py
Train / fine-tune the CNN face embedder on your own dataset.

Dataset layout expected:
    data/raw/
        ├── Alice/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── Bob/
            ├── img1.jpg
            └── img2.jpg
"""

import os
import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Data Loading                                                        #
# ------------------------------------------------------------------ #

def load_dataset(data_dir="data/raw", img_size=(96, 96)):
    """Load images from a folder-per-person structure."""
    data_dir = Path(data_dir)
    X, y, label_map = [], [], {}
    label_id = 0

    for person_dir in sorted(data_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        label_map[label_id] = name
        images_loaded = 0

        for img_path in person_dir.iterdir():
            if img_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            X.append(img)
            y.append(label_id)
            images_loaded += 1

        logger.info(f"  Loaded {images_loaded} images for '{name}'")
        label_id += 1

    return np.array(X), np.array(y), label_map


def augment_image(img):
    """Simple augmentation: flip, brightness, rotation."""
    import random
    # Horizontal flip
    if random.random() > 0.5:
        img = img[:, ::-1, :]
    # Brightness jitter
    factor = random.uniform(0.7, 1.3)
    img = np.clip(img * factor, 0, 1)
    return img


def augment_dataset(X, y, multiplier=3):
    """Expand dataset via augmentation."""
    X_aug, y_aug = [X], [y]
    for _ in range(multiplier - 1):
        X_aug.append(np.array([augment_image(x) for x in X]))
        y_aug.append(y)
    return np.concatenate(X_aug), np.concatenate(y_aug)


# ------------------------------------------------------------------ #
#  Model                                                               #
# ------------------------------------------------------------------ #

def build_model(num_classes, embedding_size=128):
    """
    Lightweight CNN with a classification head for training,
    embedding branch for inference.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model, Input

        inp = Input(shape=(96, 96, 3), name='input')

        # Feature extraction
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

        # Embedding layer
        embedding = layers.Dense(embedding_size, name='embedding')(x)
        embedding_normalized = layers.Lambda(
            lambda t: tf.math.l2_normalize(t, axis=1),
            name='embedding_norm'
        )(embedding)

        # Classification head
        output = layers.Dense(num_classes, activation='softmax', name='classification')(embedding_normalized)

        model = Model(inp, output, name='FaceRecognizer')
        return model

    except ImportError:
        logger.error("TensorFlow is required for training. Install via: pip install tensorflow")
        return None


# ------------------------------------------------------------------ #
#  Training                                                            #
# ------------------------------------------------------------------ #

def train(data_dir="data/raw", epochs=30, batch_size=32, val_split=0.2):
    try:
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelBinarizer
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return

    logger.info("=" * 50)
    logger.info("  Face Recognition CNN Training")
    logger.info("=" * 50)

    # Load data
    logger.info(f"\nLoading dataset from '{data_dir}'...")
    X, y, label_map = load_dataset(data_dir)

    if len(X) == 0:
        logger.error("No images found! Check data/raw/ directory structure.")
        return

    num_classes = len(label_map)
    logger.info(f"\nDataset: {len(X)} images, {num_classes} classes")

    if len(X) < 20:
        logger.info("Small dataset detected — applying augmentation (3x)")
        X, y = augment_dataset(X, y, multiplier=3)
        logger.info(f"After augmentation: {len(X)} images")

    # Encode labels
    lb = LabelBinarizer()
    y_enc = lb.fit_transform(y)
    if num_classes == 2:
        from tensorflow.keras.utils import to_categorical
        y_enc = to_categorical(y, num_classes)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=val_split, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

    # Build model
    model = build_model(num_classes)
    if model is None:
        return

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Callbacks
    os.makedirs("models", exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/face_embedder.h5",
            monitor='val_accuracy', save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.TensorBoard(log_dir='logs/tensorboard')
    ]

    # Train
    logger.info("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"\nFinal Validation Accuracy: {val_acc * 100:.2f}%")
    logger.info(f"Model saved to models/face_embedder.h5")

    # Save label map
    import pickle
    with open("models/label_map.pkl", 'wb') as f:
        pickle.dump(label_map, f)
    logger.info("Label map saved to models/label_map.pkl")

    # Plot training history
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['accuracy'], label='Train Acc')
        ax1.plot(history.history['val_accuracy'], label='Val Acc')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Val Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        plt.tight_layout()
        os.makedirs("logs", exist_ok=True)
        plt.savefig("logs/training_history.png")
        logger.info("Training plot saved to logs/training_history.png")
    except Exception:
        pass

    return history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train Face Recognition CNN')
    parser.add_argument('--data_dir', default='data/raw', help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    train(data_dir=args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
