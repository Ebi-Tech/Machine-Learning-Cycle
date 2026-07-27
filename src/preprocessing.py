"""Image and dataset preprocessing shared by the API, the retraining pipeline,
and the notebook.
"""
import io
import os

import numpy as np
from PIL import Image, ImageFilter
from tensorflow.keras.utils import to_categorical

IMAGE_SIZE = (28, 28)
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
SAFE_IMAGE_MODES = ("L", "RGB", "RGBA", "1", "P")


def _otsu_threshold_array(arr):
    """Blurs a 0-255 grayscale array to smooth out paper texture and camera
    noise, then applies Otsu's method to binarize it: sweeps every possible
    split point and picks the one that best separates dark background from
    bright strokes (minimizes the spread within each side), producing a
    clean black-and-white image instead of a noisy grayscale scan.
    """
    img = Image.fromarray(arr.astype("uint8"))
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    img_array = np.array(img)

    hist, _ = np.histogram(img_array.flatten(), bins=256, range=(0, 256))
    total = img_array.size
    sum_all = np.sum(np.arange(256) * hist)
    sum_bg, weight_bg = 0.0, 0
    max_variance, threshold = 0, 0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > max_variance:
            max_variance = variance
            threshold = t

    return ((img_array > threshold) * 255).astype("float32")


def _image_to_array(img, invert_if_light=False, apply_thresholding=False):
    """Converts a PIL image to a normalized (28, 28) float32 array.

    MNIST digits are white strokes on a black background, but a real
    scanned or photographed submission is dark ink on white paper. Fed in
    as-is, that light background reads as a canvas full of bright strokes,
    so when invert_if_light is set, a light-background image (mean pixel
    value above 127) is inverted back to MNIST's dark-background format
    before normalizing.

    Unusual PIL modes (CMYK, LAB, I, F, and similar) don't always convert
    to grayscale cleanly in one step, so anything outside the common modes
    is normalized to RGB first.

    apply_thresholding runs Gaussian blur + Otsu binarization before
    normalizing, for real-world photos/scans that carry paper texture and
    lighting gradients MNIST never has. It's off by default: retraining
    data walked by load_and_preprocess_dataset goes through this same
    function, and reshaping already-clean images through a background
    separation step built for noisy photos would be pointless work at
    best and lossy at worst.
    """
    if img.mode not in SAFE_IMAGE_MODES:
        img = img.convert("RGB")
    img = img.convert("L")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img).astype("float32")
    if invert_if_light and arr.mean() > 127:
        arr = 255.0 - arr
    if apply_thresholding:
        arr = _otsu_threshold_array(arr)
    return arr / 255.0


def preprocess_image(image_bytes):
    """Preprocesses raw uploaded image bytes for model.predict().

    Returns an array of shape (1, 28, 28, 1).

    Raises ValueError if image_bytes isn't a format PIL can open (a PDF,
    text file, script, or other non-image upload), so the caller can
    surface a clean 400 instead of an unhandled server error.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception:
        raise ValueError("Uploaded file is not a valid image. Accepted formats: PNG, JPG, JPEG.")

    arr = _image_to_array(img, invert_if_light=True, apply_thresholding=True)
    return arr.reshape(1, 28, 28, 1)


def preprocess_image_from_path(image_path):
    """Preprocesses an image file on disk for model.predict().

    Returns an array of shape (1, 28, 28, 1).
    """
    img = Image.open(image_path)
    arr = _image_to_array(img, invert_if_light=True, apply_thresholding=True)
    return arr.reshape(1, 28, 28, 1)


def load_and_preprocess_dataset(data_dir):
    """Loads a labeled image dataset from data_dir/{0..9}/*.png style folders.

    Returns (X, y): X is shape (n, 28, 28, 1) normalized, y is one-hot shape
    (n, 10). If no valid images are found, returns two empty arrays instead
    of raising, so callers like retrain_model can handle "nothing to train
    on" as a normal, expected outcome rather than an exception.
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
            images.append(_image_to_array(img, invert_if_light=True))
            labels.append(label)

    if len(images) == 0:
        return np.array([]), np.array([])

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
