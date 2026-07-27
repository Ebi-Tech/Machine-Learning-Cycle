"""
Lightweight Flask API for Render cloud deployment.
Uses ONNX Runtime instead of TensorFlow for inference within 512MB RAM.
Retraining is disabled on this deployment - use the Docker setup locally.
"""
import os
import json
import sqlite3
import time
import logging
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from src.prediction_onnx import load_onnx_model, predict_single_onnx

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Paths
MODEL_PATH = "models/digit_classifier.onnx"
EVAL_METRICS_PATH = "models/eval_metrics.json"
DB_PATH = "models/uploads.db"
MAX_PREDICT_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            label INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            upload_timestamp TEXT NOT NULL,
            used_in_retrain INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()


init_db()

# Load ONNX model at startup
logger.info(f"Loading ONNX model from {MODEL_PATH}")
onnx_session = load_onnx_model(MODEL_PATH)
logger.info("ONNX model loaded successfully")

# Track server start time
SERVER_START_TIME = time.time()


@app.route("/")
def serve_frontend():
    return send_from_directory("frontend", "index.html")


@app.route("/frontend/<path:filename>")
def serve_static(filename):
    return send_from_directory("frontend", filename)


@app.route("/health", methods=["GET"])
def health():
    try:
        model_mtime = os.path.getmtime(MODEL_PATH)
        model_size = os.path.getsize(MODEL_PATH)
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "model_path": MODEL_PATH,
            "model_last_trained": datetime.fromtimestamp(model_mtime).isoformat(),
            "uptime_seconds": round(time.time() - SERVER_START_TIME, 1),
            "model_parameters": 242442,  # Known from model architecture
            "runtime": "onnxruntime",
            "deployment": "render-free-tier"
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided. Send an image with key 'file'."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename."}), 400

        start_time = time.time()
        image_bytes = file.read()
        if len(image_bytes) > MAX_PREDICT_FILE_SIZE:
            return jsonify({"error": "Image file too large. Maximum size: 10MB."}), 400

        result = predict_single_onnx(onnx_session, image_bytes)
        elapsed_ms = (time.time() - start_time) * 1000

        response = {
            "predicted_digit": result["predicted_digit"],
            "confidence": round(result["confidence"], 4),
            "probabilities": result["probabilities"],
            "processing_time_ms": round(elapsed_ms, 1),
        }
        if not result.get("is_confident", True):
            response["warning"] = result.get(
                "warning",
                "Low confidence prediction. The uploaded image may not be a handwritten digit.",
            )

        return jsonify(response)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/upload", methods=["POST"])
def upload():
    return jsonify({
        "error": "Upload is not available on the Render free-tier deployment due to memory constraints. Use the Docker setup locally for full functionality including upload and retraining.",
        "docker_command": "docker compose up -d --scale app=1"
    }), 501


@app.route("/retrain", methods=["POST"])
def retrain():
    return jsonify({
        "error": "Retraining is not available on the Render free-tier deployment due to memory constraints. Use the Docker setup locally for full functionality including upload and retraining.",
        "docker_command": "docker compose up -d --scale app=1"
    }), 501


@app.route("/uploads", methods=["GET"])
def get_uploads():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM training_uploads ORDER BY upload_timestamp DESC LIMIT 100")
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        summary = {}
        for row in rows:
            label = str(row["label"])
            if label not in summary:
                summary[label] = 0
            summary[label] += 1

        return jsonify({
            "total_uploads": len(rows),
            "per_class": summary,
            "recent": rows[:20],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/visualizations", methods=["GET"])
def visualizations():
    try:
        if not os.path.exists(EVAL_METRICS_PATH):
            return jsonify({"error": "No evaluation data available yet."}), 404
        with open(EVAL_METRICS_PATH, "r") as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Visualizations failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/retrain-history", methods=["GET"])
def retrain_history():
    log_path = "models/retrain_log.json"
    try:
        if not os.path.exists(log_path):
            return jsonify([])
        with open(log_path, "r") as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify([])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
