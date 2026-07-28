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
    """Blurs the image to smooth out paper texture and camera noise, then
    finds the best black/white cutoff automatically (Otsu's method)
    instead of using one fixed threshold for every image.
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

    MNIST digits are white strokes on black, but a real photo is usually
    dark ink on white paper. invert_if_light flips a light-background
    image back to MNIST's format.

    Unusual PIL modes (CMYK, LAB, etc.) don't always convert to grayscale
    cleanly, so anything outside the common modes goes through RGB first.

    apply_thresholding blurs and cleans up the image for real photos with
    paper texture and lighting glare. It's off by default because
    load_and_preprocess_dataset uses this same function for retraining
    data, which is already clean and doesn't need it.
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
    """Raises ValueError if image_bytes isn't a format PIL can open (a PDF,
    text file, or other non-image upload), so the caller can return a
    clean 400 instead of crashing.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception:
        raise ValueError("Uploaded file is not a valid image. Accepted formats: PNG, JPG, JPEG.")

    arr = _image_to_array(img, invert_if_light=True, apply_thresholding=True)
    return arr.reshape(1, 28, 28, 1)


def preprocess_image_from_path(image_path):
    img = Image.open(image_path)
    arr = _image_to_array(img, invert_if_light=True, apply_thresholding=True)
    return arr.reshape(1, 28, 28, 1)


def load_and_preprocess_dataset(data_dir):
    """Loads a labeled image dataset from data_dir/{0..9}/*.png style folders.

    Returns two empty arrays if no valid images are found, instead of
    raising, so retrain_model can treat "nothing to train on" as a normal
    outcome instead of an error.
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
    X = X_raw.astype("float32") / 255.0
    X = X.reshape(-1, 28, 28, 1)
    y = to_categorical(y_raw, num_classes=10)
    return X, y
