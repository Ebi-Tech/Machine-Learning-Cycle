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
    """Builds and compiles the Experiment 2 architecture (the notebook's
    selected best model): Conv-BatchNorm-Pool-Dropout x2, then a deeper conv
    block, dropout, and a dense head.
    """
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
    """Trains model with early stopping and LR reduction on plateau.

    Returns the training history dict (history.history).
    """
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
    """Fine-tunes a saved model on newly collected data, gated by a safety
    check against the held-out MNIST test set.

    A retrain on bad or too-small a batch of new data can quietly wreck a
    deployed model, so the retrained model is only promoted (saved over
    model_path) if its accuracy on data/test/ stays within `tolerance` of
    the pre-retrain baseline. If it regresses beyond that, the original
    model file is left untouched and the fine-tuned weights are discarded.

    A saved model's optimizer is not fit-ready after reload, so it is
    recompiled with a fresh Adam optimizer before fine-tuning.

    Returns a dict with promoted, baseline_accuracy, new_accuracy,
    accuracy_change, samples_used, epochs_run, reason.
    """
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


def get_model_info(model_path):
    """Returns file size, last modified timestamp, and parameter count for
    a saved model, for use by the health endpoint.
    """
    if not os.path.exists(model_path):
        return {
            "exists": False,
            "size_bytes": None,
            "last_modified": None,
            "parameters": None,
        }

    size_bytes = os.path.getsize(model_path)
    last_modified = datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
    model = keras.models.load_model(model_path)

    return {
        "exists": True,
        "size_bytes": size_bytes,
        "last_modified": last_modified,
        "parameters": model.count_params(),
    }
