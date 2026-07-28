"""
Locust load test for the Handwritten Digit Recognition API.
Simulates users uploading digit images for prediction.

Usage:
    # Headless mode (for capturing results):
    locust -f locust/locustfile.py --host=http://localhost:8080 --headless \
        -u 50 -r 10 --run-time 60s --csv=locust/results

    # Web UI mode (for interactive testing):
    locust -f locust/locustfile.py --host=http://localhost:8080
"""

import io
import os
import numpy as np
from PIL import Image
from locust import HttpUser, task, between, events


# Filled in once before the test starts, not per user.
TEST_IMAGES = []


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    global TEST_IMAGES

    # Real images give more realistic inference times than random noise.
    mnist_path = os.path.join(os.path.dirname(__file__), "..", "data", "test", "X_test.npy")
    if os.path.exists(mnist_path):
        X_test = np.load(mnist_path)
        # 20 images spread across the set, not just the first 20
        indices = list(range(0, 10000, 500))[:20]
        for idx in indices:
            img = Image.fromarray(X_test[idx])
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            TEST_IMAGES.append(buf.getvalue())
        print(f"Loaded {len(TEST_IMAGES)} real MNIST test images for load testing.")
    else:
        rng = np.random.default_rng(42)
        for _ in range(20):
            img_array = np.zeros((28, 28), dtype=np.uint8)
            # A few random bright blocks, standing in for pen strokes
            for _ in range(5):
                x, y = rng.integers(5, 23), rng.integers(5, 23)
                img_array[y-2:y+2, x-2:x+2] = rng.integers(180, 255)
            img = Image.fromarray(img_array)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            TEST_IMAGES.append(buf.getvalue())
        print(f"Generated {len(TEST_IMAGES)} synthetic test images for load testing.")


class DigitPredictionUser(HttpUser):
    """Simulates a user who repeatedly submits digit images for prediction."""
    wait_time = between(0.5, 2)

    @task(10)
    def predict_digit(self):
        # Weighted 10x: predicting images is the main thing this API does.
        img_bytes = TEST_IMAGES[np.random.randint(len(TEST_IMAGES))]
        self.client.post(
            "/predict",
            files={"file": ("digit.png", io.BytesIO(img_bytes), "image/png")},
            name="/predict",
        )

    @task(2)
    def check_health(self):
        # Weighted 2x, like a monitoring tool or the frontend polling uptime.
        self.client.get("/health", name="/health")

    @task(1)
    def get_visualizations(self):
        # Weighted 1x, like a user opening the dashboard occasionally.
        self.client.get("/visualizations", name="/visualizations")
