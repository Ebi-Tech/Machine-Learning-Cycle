"""CNN architecture, training, and retraining logic shared by the notebook
and the API.
"""
import os
from datetime import datetime

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

from src.preprocessing import preprocess_raw_arrays

TEST_X_PATH = "data/test/X_test.npy"
TEST_Y_PATH = "data/test/y_test.npy"


def build_enhanced_cnn():
    """Builds the CNN architecture chosen in the notebook (Experiment 2)."""
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dropout(0.5),

        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ], name="enhanced_cnn")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=64):
    training_callbacks = [
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=training_callbacks,
        verbose=2,
    )
    return history.history


def retrain_model(model_path, X_new, y_new, epochs=5, batch_size=32, tolerance=0.005):
    """Retrains a saved model on new data, but only keeps the result if
    it's not much worse than the original.

    Bad or too little new data can quietly wreck a working model, so the
    retrained version only overwrites model_path if its test accuracy
    stays within `tolerance` of the original. Otherwise the original file
    is left alone and the new weights are thrown away.

    A saved model isn't ready to keep training right after loading, so it
    gets recompiled with a fresh optimizer first.
    """
    if len(X_new) == 0 or len(y_new) == 0:
        return {
            "promoted": False,
            "baseline_accuracy": 0.0,
            "new_accuracy": 0.0,
            "accuracy_change": 0.0,
            "samples_used": 0,
            "epochs_run": 0,
            "reason": "No valid training images found in the upload directory.",
        }

    model = keras.models.load_model(model_path)

    X_test_raw = np.load(TEST_X_PATH)
    y_test_raw = np.load(TEST_Y_PATH)
    X_test, y_test = preprocess_raw_arrays(X_test_raw, y_test_raw)

    _, baseline_accuracy = model.evaluate(X_test, y_test, verbose=0)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(
        X_new, y_new,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    _, new_accuracy = model.evaluate(X_test, y_test, verbose=0)

    baseline_accuracy = float(baseline_accuracy)
    new_accuracy = float(new_accuracy)
    accuracy_change = new_accuracy - baseline_accuracy
    promoted = new_accuracy >= (baseline_accuracy - tolerance)

    if promoted:
        model.save(model_path)
        reason = (
            f"Retrained model accuracy ({new_accuracy:.4f}) is within tolerance "
            f"of baseline ({baseline_accuracy:.4f}). Model promoted."
        )
    else:
        reason = (
            f"Retrained model accuracy ({new_accuracy:.4f}) regressed beyond "
            f"{tolerance * 100:.1f}% tolerance from baseline ({baseline_accuracy:.4f}). "
            f"Original model kept."
        )

    return {
        "promoted": bool(promoted),
        "baseline_accuracy": baseline_accuracy,
        "new_accuracy": new_accuracy,
        "accuracy_change": accuracy_change,
        "samples_used": int(len(X_new)),
        "epochs_run": len(history.history["loss"]),
        "reason": reason,
    }


_model_info_cache = {"mtime": None, "params": None}


def get_model_info(model_path):
    """Used by the health endpoint, which calls this on every request.
    Loading the full model every time would be slow, so the parameter
    count is cached and only recomputed when the model file changes.
    """
    if not os.path.exists(model_path):
        return {
            "exists": False,
            "size_bytes": None,
            "last_modified": None,
            "parameters": None,
        }

    size_bytes = os.path.getsize(model_path)
    mtime = os.path.getmtime(model_path)
    last_modified = datetime.fromtimestamp(mtime).isoformat()

    if _model_info_cache["mtime"] != mtime:
        model = keras.models.load_model(model_path)
        _model_info_cache["mtime"] = mtime
        _model_info_cache["params"] = model.count_params()

    return {
        "exists": True,
        "size_bytes": size_bytes,
        "last_modified": last_modified,
        "parameters": _model_info_cache["params"],
    }
