"""Prediction helpers shared by the API and batch/demo scripts."""
from src.preprocessing import preprocess_image, preprocess_image_from_path


def _predict_from_array(model, image_array):
    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_digit = int(probabilities.argmax())
    return {
        "predicted_digit": predicted_digit,
        "confidence": float(probabilities[predicted_digit]),
        "probabilities": [float(p) for p in probabilities],
    }


def predict_single(model, image_bytes):
    """Predicts a digit from raw uploaded image bytes.

    Returns a dict with predicted_digit, confidence, and probabilities.
    """
    image_array = preprocess_image(image_bytes)
    return _predict_from_array(model, image_array)


def predict_from_path(model, image_path):
    """Predicts a digit from an image file on disk.

    Returns a dict with predicted_digit, confidence, and probabilities.
    """
    image_array = preprocess_image_from_path(image_path)
    return _predict_from_array(model, image_array)
