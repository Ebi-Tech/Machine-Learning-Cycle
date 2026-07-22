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
from PIL import Image

IMAGE_SIZE = (28, 28)


def _preprocess_image_array(img):
    """Converts a PIL image to a normalized (1, 28, 28, 1) float32 array,
    inverting light backgrounds (mean > 127) to MNIST's dark-background
    format, matching src.preprocessing._image_to_array(invert_if_light=True).
    """
    img = img.convert("L").resize(IMAGE_SIZE)
    arr = np.array(img).astype("float32")
    if arr.mean() > 127:
        arr = 255.0 - arr
    arr = arr / 255.0
    return arr.reshape(1, 28, 28, 1)


def preprocess_image(image_bytes):
    """Preprocesses raw uploaded image bytes for inference."""
    img = Image.open(io.BytesIO(image_bytes))
    return _preprocess_image_array(img)


def preprocess_image_from_path(image_path):
    """Preprocesses an image file on disk for inference."""
    img = Image.open(image_path)
    return _preprocess_image_array(img)


def load_onnx_model(model_path):
    """Load an ONNX model and return the inference session."""
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session


def predict_single_onnx(session, image_bytes):
    """
    Run prediction using ONNX runtime.
    Takes an ONNX session and raw image bytes.
    Returns dict with predicted_digit, confidence, and probabilities.
    """
    processed = preprocess_image(image_bytes)
    processed = processed.astype(np.float32)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: processed})[0]

    probabilities = output[0]

    # Apply softmax if the output isn't already probabilities
    # (some ONNX exports produce logits)
    if np.any(probabilities < 0) or np.abs(np.sum(probabilities) - 1.0) > 0.01:
        exp_preds = np.exp(probabilities - np.max(probabilities))
        probabilities = exp_preds / np.sum(exp_preds)

    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    return {
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "probabilities": [float(p) for p in probabilities],
    }


def predict_from_path_onnx(session, image_path):
    """Run prediction from a file path using ONNX runtime."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    return predict_single_onnx(session, image_bytes)
