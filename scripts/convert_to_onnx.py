"""
Convert digit_classifier.h5 to digit_classifier.onnx for lightweight cloud deployment.
Run from project root:
    python scripts/convert_to_onnx.py
"""
import os
import subprocess
import sys
import numpy as np

# Stop TensorFlow from printing its startup warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

model = tf.keras.models.load_model("models/digit_classifier.h5")

# tf2onnx's from_keras() crashes on this model (KeyError) because of how
# Keras 3 names its tensors. Using the notebook's SavedModel export instead
# avoids the bug, since tf2onnx's command-line tool handles that format fine.
output_path = "models/digit_classifier.onnx"
savedmodel_path = "models/digit_classifier_savedmodel"

subprocess.run(
    [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", savedmodel_path,
        "--output", output_path,
        "--opset", "13",
    ],
    check=True,
)

test_input = np.random.rand(1, 28, 28, 1).astype(np.float32)

tf_pred = model.predict(test_input, verbose=0)

import onnxruntime as ort
session = ort.InferenceSession(output_path)
input_name = session.get_inputs()[0].name
onnx_pred = session.run(None, {input_name: test_input})[0]

max_diff = np.max(np.abs(tf_pred - onnx_pred))
print(f"Saved ONNX model to {output_path}")
print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")
print(f"  Max prediction difference between TF and ONNX: {max_diff:.8f}")
if max_diff < 1e-5:
    print("  Verification PASSED: predictions are identical.")
else:
    print("  WARNING: predictions differ more than expected.")
