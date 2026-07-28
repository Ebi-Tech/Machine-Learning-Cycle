"""
ONNX-based prediction for lightweight cloud deployment.
Uses onnxruntime instead of TensorFlow for inference.

Preprocessing is duplicated here (not imported from src.preprocessing)
because that module imports tensorflow.keras.utils at the top level.
requirements-render.txt deliberately excludes TensorFlow to stay within
Render's free-tier memory limit, so pulling in src.preprocessing would
crash this deployment at import time. Keep this file's preprocessing in
sync with src/preprocessing.py's _image_to_array if that logic changes.
"""
import io

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter

IMAGE_SIZE = (28, 28)
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


def _preprocess_image_array(img):
    """Inverts light-background images to MNIST's dark-background format.
    Matches src.preprocessing._image_to_array(invert_if_light=True).

    Unusual PIL modes (CMYK, LAB, etc.) don't always convert to grayscale
    cleanly, so anything outside the common modes goes through RGB first.
    """
    if img.mode not in SAFE_IMAGE_MODES:
        img = img.convert("RGB")
    img = img.convert("L")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img).astype("float32")
    if arr.mean() > 127:
        arr = 255.0 - arr
    arr = _otsu_threshold_array(arr)
    arr = arr / 255.0
    return arr.reshape(1, 28, 28, 1)


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

    return _preprocess_image_array(img)


def preprocess_image_from_path(image_path):
    img = Image.open(image_path)
    return _preprocess_image_array(img)


def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session


def predict_single_onnx(session, image_bytes, confidence_threshold=0.7):
    """A near-blank image is rejected before it reaches the model. The
    model always picks some digit, even for an empty image, often with
    high confidence, so blank images need to be caught separately.
    """
    processed = preprocess_image(image_bytes)
    processed = processed.astype(np.float32)

    active_pixels = int((processed > 0.1).sum())
    if active_pixels < 10:
        return {
            "predicted_digit": None,
            "confidence": 0.0,
            "probabilities": [0.0] * 10,
            "is_confident": False,
            "warning": "The uploaded image appears to be blank or does not contain a visible digit.",
        }

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: processed})[0]

    probabilities = output[0]

    # Some ONNX exports give raw scores instead of probabilities, so convert if needed.
    if np.any(probabilities < 0) or np.abs(np.sum(probabilities) - 1.0) > 0.01:
        exp_preds = np.exp(probabilities - np.max(probabilities))
        probabilities = exp_preds / np.sum(exp_preds)

    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    return {
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "probabilities": [float(p) for p in probabilities],
        "is_confident": confidence >= confidence_threshold,
    }


def predict_from_path_onnx(session, image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return predict_single_onnx(session, image_bytes)
