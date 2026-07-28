"""Prediction helpers shared by the API and batch/demo scripts."""
from src.preprocessing import preprocess_image, preprocess_image_from_path


def _predict_from_array(model, image_array, confidence_threshold=0.7):
    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_digit = int(probabilities.argmax())
    confidence = float(probabilities[predicted_digit])
    return {
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "probabilities": [float(p) for p in probabilities],
        "is_confident": confidence >= confidence_threshold,
    }


def predict_single(model, image_bytes, confidence_threshold=0.7):
    """A near-blank image is rejected before it reaches the model. The
    model always picks some digit, even for an empty image, often with
    high confidence, so blank images need to be caught separately.
    """
    image_array = preprocess_image(image_bytes)

    active_pixels = int((image_array > 0.1).sum())
    if active_pixels < 10:
        return {
            "predicted_digit": None,
            "confidence": 0.0,
            "probabilities": [0.0] * 10,
            "is_confident": False,
            "warning": "The uploaded image appears to be blank or does not contain a visible digit.",
        }

    return _predict_from_array(model, image_array, confidence_threshold)


def predict_from_path(model, image_path):
    image_array = preprocess_image_from_path(image_path)
    return _predict_from_array(model, image_array)
