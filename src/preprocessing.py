"""Image and dataset preprocessing shared by the API, the retraining pipeline,
and the notebook.
"""
import io
import os

import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical

IMAGE_SIZE = (28, 28)
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def _image_to_array(img):
    """Converts a PIL image to a normalized (28, 28) float32 array."""
    img = img.convert("L").resize(IMAGE_SIZE)
    return np.array(img).astype("float32") / 255.0


def preprocess_image(image_bytes):
    """Preprocesses raw uploaded image bytes for model.predict().

    Returns an array of shape (1, 28, 28, 1).
    """
    img = Image.open(io.BytesIO(image_bytes))
    arr = _image_to_array(img)
    return arr.reshape(1, 28, 28, 1)


def preprocess_image_from_path(image_path):
    """Preprocesses an image file on disk for model.predict().

    Returns an array of shape (1, 28, 28, 1).
    """
    img = Image.open(image_path)
    arr = _image_to_array(img)
    return arr.reshape(1, 28, 28, 1)


def load_and_preprocess_dataset(data_dir):
    """Loads a labeled image dataset from data_dir/{0..9}/*.png style folders.

    Returns (X, y): X is shape (n, 28, 28, 1) normalized, y is one-hot shape (n, 10).
    """
    images = []
    labels = []

    class_dirs = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()
    )

    for class_dir in class_dirs:
        label = int(class_dir)
        class_path = os.path.join(data_dir, class_dir)
        for filename in sorted(os.listdir(class_path)):
            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue
            img_path = os.path.join(class_path, filename)
            img = Image.open(img_path)
            images.append(_image_to_array(img))
            labels.append(label)

    if not images:
        raise ValueError(f"No valid images found under {data_dir}")

    X = np.array(images, dtype="float32").reshape(-1, 28, 28, 1)
    y = to_categorical(np.array(labels), num_classes=10)
    return X, y


def preprocess_raw_arrays(X_raw, y_raw):
    """Preprocesses raw MNIST-style arrays (uint8 images, integer labels).

    Returns (X, y): X is shape (n, 28, 28, 1) normalized, y is one-hot shape (n, 10).
    """
    X = X_raw.astype("float32") / 255.0
    X = X.reshape(-1, 28, 28, 1)
    y = to_categorical(y_raw, num_classes=10)
    return X, y
