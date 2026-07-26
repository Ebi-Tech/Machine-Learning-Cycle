"""Flask API serving the handwritten digit classifier.

Run from the project root with: python app.py
"""
import fcntl
import json
import logging
import os
import shutil
import time
import zipfile
from datetime import datetime

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from werkzeug.utils import secure_filename

from src.model import get_model_info, retrain_model
from src.preprocessing import load_and_preprocess_dataset, preprocess_raw_arrays
from src.prediction import predict_single

MODEL_PATH = "models/digit_classifier.h5"
RETRAIN_DIR = "data/retrain"
EVAL_METRICS_PATH = "models/eval_metrics.json"
RETRAIN_LOG_PATH = "models/retrain_log.json"
TRAIN_DATA_PATH = "data/train"
TEST_DATA_PATH = "data/test"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}
MAX_PREDICT_FILE_SIZE = 10 * 1024 * 1024  # 10MB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("digit-api")

tf.get_logger().setLevel("ERROR")

os.makedirs(RETRAIN_DIR, exist_ok=True)

START_TIME = time.time()
model = keras.models.load_model(MODEL_PATH)
logger.info(f"Loaded model from {MODEL_PATH}")

app = Flask(__name__)
CORS(app)


@app.route("/")
def serve_frontend():
    return send_from_directory("frontend", "index.html")


@app.route("/frontend/<path:filename>")
def serve_static(filename):
    return send_from_directory("frontend", filename)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _compute_and_save_eval_metrics(current_model):
    """Evaluates current_model on the held-out test set and writes
    models/eval_metrics.json for the /visualizations endpoint.
    """
    X_test_raw = np.load(os.path.join(TEST_DATA_PATH, "X_test.npy"))
    y_test_raw = np.load(os.path.join(TEST_DATA_PATH, "y_test.npy"))
    X_test, _ = preprocess_raw_arrays(X_test_raw, y_test_raw)

    y_pred_proba = current_model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    report = classification_report(y_test_raw, y_pred, output_dict=True)
    per_class_f1 = [report[str(c)]["f1-score"] for c in range(10)]
    cm = confusion_matrix(y_test_raw, y_pred).tolist()

    y_train_raw = np.load(os.path.join(TRAIN_DATA_PATH, "y_train.npy"))
    class_distribution = np.bincount(y_train_raw, minlength=10).tolist()

    metrics = {
        "class_distribution": class_distribution,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "updated_at": datetime.now().isoformat(),
    }

    os.makedirs(os.path.dirname(EVAL_METRICS_PATH), exist_ok=True)
    with open(EVAL_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def _append_retrain_log(entry):
    """Appends a retrain attempt (promoted or rejected) to
    models/retrain_log.json, so /retrain-history can return a full audit
    trail of every retraining attempt.

    Under multi-replica deployment, two containers can both be handling a
    /retrain call and writing to this same volume-mounted file at once. An
    unlocked read-modify-write would let one write silently clobber the
    other, so the whole read-append-write cycle holds an exclusive file
    lock (fcntl.flock), which blocks other writers until it's released.
    """
    os.makedirs(os.path.dirname(RETRAIN_LOG_PATH), exist_ok=True)
    with open(RETRAIN_LOG_PATH, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        try:
            log = json.load(f)
        except (json.JSONDecodeError, ValueError):
            log = []
        log.append(entry)
        f.seek(0)
        f.truncate()
        json.dump(log, f, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)


def _handle_zip_upload(zip_file):
    tmp_dir = os.path.join(RETRAIN_DIR, f"_tmp_zip_{int(time.time() * 1000)}")
    os.makedirs(tmp_dir, exist_ok=True)
    zip_path = os.path.join(tmp_dir, "upload.zip")
    zip_file.save(zip_path)

    per_class_counts = {}
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        base_dir = tmp_dir
        top_level_dirs = [
            d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
        ]
        has_digit_dirs = any(d.isdigit() for d in top_level_dirs)

        if not has_digit_dirs and len(top_level_dirs) == 1:
            # Common case: someone right-clicked a folder and hit "Compress"
            # on macOS, which wraps the 0-9 folders in a parent directory.
            # Look one level deeper before giving up.
            nested_dir = os.path.join(base_dir, top_level_dirs[0])
            nested_entries = [
                d for d in os.listdir(nested_dir) if os.path.isdir(os.path.join(nested_dir, d))
            ]
            if any(d.isdigit() for d in nested_entries):
                base_dir = nested_dir

        for entry in sorted(os.listdir(base_dir)):
            entry_path = os.path.join(base_dir, entry)
            if not os.path.isdir(entry_path) or not entry.isdigit():
                continue

            label_dir = os.path.join(RETRAIN_DIR, entry)
            os.makedirs(label_dir, exist_ok=True)

            saved = 0
            for filename in sorted(os.listdir(entry_path)):
                if not allowed_file(filename):
                    continue
                src_path = os.path.join(entry_path, filename)
                dst_path = os.path.join(label_dir, secure_filename(filename))
                shutil.move(src_path, dst_path)
                saved += 1

            if saved:
                per_class_counts[entry] = per_class_counts.get(entry, 0) + saved
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not per_class_counts:
        return jsonify({
            "error": "Zip did not contain any valid class subdirectories (0-9) with images."
        }), 400

    return jsonify({
        "uploaded": sum(per_class_counts.values()),
        "per_class": per_class_counts,
        "upload_dir": f"{RETRAIN_DIR}/",
    }), 200


@app.route("/health", methods=["GET"])
def health():
    try:
        info = get_model_info(MODEL_PATH)
        return jsonify({
            "status": "healthy",
            "model_loaded": model is not None,
            "model_path": MODEL_PATH,
            "model_last_trained": info["last_modified"],
            "uptime_seconds": int(time.time() - START_TIME),
            "model_parameters": info["parameters"],
        }), 200
    except Exception as e:
        logger.error(f"/health failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded. Include a file under the 'file' field."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": f"Invalid file format. Allowed: {sorted(ALLOWED_EXTENSIONS)}"}), 400

        image_bytes = file.read()
        if len(image_bytes) > MAX_PREDICT_FILE_SIZE:
            return jsonify({"error": "Image file too large. Maximum size: 10MB."}), 400

        start = time.time()
        result = predict_single(model, image_bytes)
        elapsed_ms = (time.time() - start) * 1000

        response = {
            "predicted_digit": result["predicted_digit"],
            "confidence": round(result["confidence"], 4),
            "probabilities": result["probabilities"],
            "processing_time_ms": round(elapsed_ms, 2),
        }
        if not result.get("is_confident", True):
            response["warning"] = "Low confidence prediction. The uploaded image may not be a handwritten digit."

        return jsonify(response), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"/predict failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "files" not in request.files:
            return jsonify({"error": "No files uploaded. Include files under the 'files' field."}), 400

        files = request.files.getlist("files")
        if not files or all(f.filename == "" for f in files):
            return jsonify({"error": "No files uploaded."}), 400

        # Bulk mode: a single .zip carries its own 0-9 subdirectory structure.
        if len(files) == 1 and files[0].filename.lower().endswith(".zip"):
            return _handle_zip_upload(files[0])

        label = request.form.get("label")
        if label is None or not label.isdigit() or not (0 <= int(label) <= 9):
            return jsonify({"error": "A valid 'label' form field (digit 0-9) is required."}), 400

        label_dir = os.path.join(RETRAIN_DIR, label)
        os.makedirs(label_dir, exist_ok=True)

        saved = 0
        for f in files:
            if f.filename == "" or not allowed_file(f.filename):
                continue
            f.save(os.path.join(label_dir, secure_filename(f.filename)))
            saved += 1

        if saved == 0:
            return jsonify({"error": "No valid image files were uploaded."}), 400

        return jsonify({
            "uploaded": saved,
            "label": label,
            "upload_dir": f"{label_dir}/",
        }), 200
    except Exception as e:
        logger.error(f"/upload failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/retrain", methods=["POST"])
def retrain():
    global model
    try:
        class_dirs = [
            d for d in os.listdir(RETRAIN_DIR)
            if os.path.isdir(os.path.join(RETRAIN_DIR, d)) and d.isdigit()
            and os.listdir(os.path.join(RETRAIN_DIR, d))
        ]
        if not class_dirs:
            return jsonify({"error": "No training data available. Upload images first using the /upload endpoint."}), 400

        X_new, y_new = load_and_preprocess_dataset(RETRAIN_DIR)

        start = time.time()
        result = retrain_model(MODEL_PATH, X_new, y_new, epochs=5, batch_size=32)
        elapsed = time.time() - start

        response = {
            "status": "promoted" if result["promoted"] else "rejected",
            "promoted": result["promoted"],
            "baseline_accuracy": round(result["baseline_accuracy"], 4),
            "new_accuracy": round(result["new_accuracy"], 4),
            "accuracy_change": round(result["accuracy_change"], 4),
            "samples_used": result["samples_used"],
            "epochs_run": result["epochs_run"],
            "training_time_seconds": round(elapsed, 2),
            "reason": result["reason"],
        }

        if result["promoted"]:
            model = keras.models.load_model(MODEL_PATH)
            logger.info(f"Reloaded model from {MODEL_PATH}: retrain promoted")
            try:
                _compute_and_save_eval_metrics(model)
            except Exception as metrics_error:
                logger.error(f"Failed to update eval metrics after retraining: {metrics_error}")
        else:
            logger.info(f"Retrain rejected, original model kept: {result['reason']}")

        _append_retrain_log({**response, "timestamp": datetime.now().isoformat()})

        return jsonify(response), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"/retrain failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/retrain-history", methods=["GET"])
def retrain_history():
    try:
        if not os.path.exists(RETRAIN_LOG_PATH):
            return jsonify([]), 200

        with open(RETRAIN_LOG_PATH) as f:
            log = json.load(f)
        return jsonify(log), 200
    except Exception as e:
        logger.error(f"/retrain-history failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/visualizations", methods=["GET"])
def visualizations():
    try:
        if not os.path.exists(EVAL_METRICS_PATH):
            return jsonify({
                "error": "Model has not been evaluated yet. Train or retrain the model to generate visualization data."
            }), 404

        with open(EVAL_METRICS_PATH) as f:
            metrics = json.load(f)
        return jsonify(metrics), 200
    except Exception as e:
        logger.error(f"/visualizations failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
