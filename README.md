# Handwritten Digit Recognition Pipeline

An end-to-end machine learning pipeline for automated grading of handwritten numerical answers in distance learning. Built as an extension of a previous student outcome prediction study (OULAD dataset) into the image domain.

## Demo & Deployment

- **Video Demo:** [YouTube Link](YOUR_YOUTUBE_LINK_HERE)
- **Live Deployment:** [https://machine-learning-cycle.onrender.com](https://machine-learning-cycle.onrender.com)

> Note: The Render deployment uses ONNX Runtime for inference within the 512MB free-tier memory limit. Retraining is available only through the local Docker setup. The cloud deployment supports prediction, health monitoring, and visualizations.

## Project Description

This project demonstrates the full ML lifecycle: data acquisition, preprocessing, model training, evaluation, deployment, monitoring, and retraining. A custom CNN classifies handwritten digits (0-9) from the MNIST dataset, served through a Flask API with a browser-based dashboard.

Key engineering decisions:

- **Custom CNN over transfer learning:** MNIST images are 28x28 grayscale. MobileNet and similar architectures expect 224x224 RGB input, requiring wasteful upscaling. A lightweight custom CNN (242,442 parameters) is faster to train, faster at inference, and produces a smaller Docker image for scaling.
- **Background inversion in preprocessing:** Real handwritten submissions arrive as dark ink on white paper (the opposite of MNIST's white-on-black format). The preprocessing pipeline detects light backgrounds and inverts automatically, so the model handles real-world input without retraining on augmented data.
- **Retraining safety gate:** A user can upload new images and trigger retraining through the UI. Before promoting a retrained model, the system evaluates it against the held-out test set. If accuracy drops more than 0.5 percentage points from baseline, the retrained model is rejected and the original is kept. This prevents bad or insufficient data from degrading the deployed model.
- **Dual deployment path:** The full TensorFlow setup runs locally and in Docker (supports training, retraining, and inference). A separate ONNX Runtime path powers the cloud deployment (inference only, under 100MB memory).

## Architecture

```
User -> Browser Dashboard (HTML/JS/Tailwind/DaisyUI)
        |
        v
Flask API (app.py)
  /health            model status, uptime, parameters
  /predict           single image classification
  /upload            bulk image upload for retraining
  /retrain           trigger retraining with safety gate
  /retrain-history    audit log of all retrain attempts
  /visualizations    dataset stats, F1 scores, confusion matrix
        |
        v
CNN Model (digit_classifier.h5 / .onnx)
        |
        v
MNIST Dataset (60,000 train / 10,000 test)
```

## Model Performance

Trained and evaluated across three experiments (see `notebook/handwritten_digit_recognition.ipynb`, Section 7.6):

| Experiment | Architecture | Test Accuracy | Macro F1 |
|------------|-------------|---------------|----------|
| 1. Baseline CNN | Conv-Conv-Dense | 99.18% | 0.9917 |
| 2. Enhanced CNN (selected) | Conv-BN-Dropout-Conv-BN-Dropout-Conv-BN-Dense | 99.39% | 0.9939 |
| 3. Enhanced + Augmentation | Same as Exp 2 + ImageDataGenerator | 99.14% | 0.9914 |

Experiment 2 was selected: BatchNorm and Dropout improved generalization without the diminishing returns of augmentation on an already-clean dataset.

The deployed model continues to evolve through the retraining safety gate described above. Its live accuracy may differ slightly from the table above as a result of legitimate retraining events after this initial evaluation; every retrain attempt, promoted or rejected, is recorded in `models/retrain_log.json` and visible in the dashboard's History tab.

## Flood Request Simulation (Locust)

Tested with 100 concurrent users, 20 users/second spawn rate, 60-second duration per configuration. All requests hit the /predict endpoint with real MNIST test images through an nginx load balancer.

| Containers | Median Latency | Avg Latency | Throughput | Max Latency | Failures |
|-----------|---------------|-------------|------------|-------------|----------|
| 1 | 930 ms | 951 ms | 44.3 req/s | 1,689 ms | 0 |
| 2 | 380 ms | 410 ms | 58.7 req/s | 1,086 ms | 0 |
| 4 | 250 ms | 307 ms | 62.9 req/s | 1,554 ms | 0 |

Scaling from 1 to 2 containers reduced median latency by 59% and increased throughput by 33%. Scaling from 2 to 4 gave a further 34% latency reduction with 7% more throughput. The diminishing returns from 2 to 4 are expected: all containers share the same CPU on a single host, so the bottleneck shifts from application concurrency to hardware.

Zero failures across all three configurations.

Full Locust CSV results are in `locust/results/`.

## Technologies

- **Model:** TensorFlow/Keras (custom CNN, 3 conv layers, BatchNorm, Dropout)
- **API:** Flask, Gunicorn
- **Frontend:** HTML, JavaScript, Tailwind CSS, DaisyUI, Chart.js
- **Containerization:** Docker, Docker Compose, Nginx (reverse proxy and load balancer)
- **Cloud:** Render (ONNX Runtime inference), Dockerfile.render
- **Load Testing:** Locust
- **Dataset:** MNIST (70,000 images, 28x28 grayscale, 10 classes)

## Project Structure

```
Machine-Learning-Cycle/
├── README.md
├── notebook/
│   └── handwritten_digit_recognition.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   └── prediction_onnx.py
├── data/
│   ├── train/       (generated by notebook)
│   ├── test/        (generated by notebook)
│   └── retrain/     (populated at runtime via uploads)
├── models/
│   ├── digit_classifier.h5
│   ├── digit_classifier.onnx
│   └── eval_metrics.json
├── frontend/
│   ├── index.html
│   └── script.js
├── locust/
│   ├── locustfile.py
│   ├── run_scaling_test.sh
│   └── results/
├── scripts/
│   ├── generate_eval_metrics.py
│   └── convert_to_onnx.py
├── app.py
├── app_render.py
├── Dockerfile
├── Dockerfile.render
├── docker-compose.yml
├── nginx.conf
├── requirements.txt
└── requirements-render.txt
```

## Setup Instructions

### Prerequisites

- Python 3.11
- Docker and Docker Compose (for containerized deployment)
- pip

### Local Development

```bash
# Clone the repository
git clone https://github.com/Ebi-Tech/Machine-Learning-Cycle.git
cd Machine-Learning-Cycle

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the notebook first to generate training/test data
# Open notebook/handwritten_digit_recognition.ipynb and run all cells

# Generate evaluation metrics
python scripts/generate_eval_metrics.py

# Start the server
python app.py

# Open http://localhost:5000 in your browser
```

### Docker Deployment (with scaling)

```bash
# Generate eval metrics if not already present
python scripts/generate_eval_metrics.py

# Build the Docker image
docker compose build

# Run with 1 container
docker compose up -d --scale app=1
# Access at http://localhost:8080

# Scale to 2 containers
docker compose up -d --scale app=2

# Scale to 4 containers
docker compose up -d --scale app=4

# Check running containers
docker compose ps

# Stop everything
docker compose down
```

### Run Locust Load Tests

```bash
# Automated scaling test (runs 1, 2, and 4 containers sequentially)
chmod +x locust/run_scaling_test.sh
./locust/run_scaling_test.sh

# Or run manually against a running deployment
locust -f locust/locustfile.py --host=http://localhost:8080
```

### Cloud Deployment (Render)

The Render deployment uses a separate lightweight Dockerfile (`Dockerfile.render`) with ONNX Runtime instead of TensorFlow to fit within the 512MB free-tier memory limit.

To deploy on Render:

1. Connect the GitHub repo on render.com
2. Set the Dockerfile path to `./Dockerfile.render`
3. Deploy

The ONNX model is pre-converted and committed to the repo. To regenerate it:

```bash
pip install tf2onnx onnxruntime
python scripts/convert_to_onnx.py
```

## Constraints and Limitations

- **Real-world photo preprocessing:** The model achieves 99.25% accuracy on MNIST-formatted images (white digit on black background, 28x28 grayscale, single centered digit). Several preprocessing steps were built to bridge the gap between phone photos and MNIST format: automatic background inversion for dark-on-light images, Gaussian blur for noise reduction, and Otsu thresholding for contrast cleanup. These work well for clean, high-contrast images on white paper. However, photos on colored or textured paper, digits that are small in the frame, or images with lighting gradients still produce unreliable predictions. The root cause is that resizing a large photo to 28x28 pixels destroys fine stroke detail when the digit occupies a small portion of the frame, and background noise survives the downscaling. A full solution would require digit localization (bounding-box detection and cropping before resize), which was explored but not completed. For reliable predictions, upload images with a single digit on a plain white or black background, filling most of the frame.
- **Render free tier (512MB RAM):** Full TensorFlow cannot run within this limit. The cloud deployment uses ONNX Runtime for inference only. Retraining requires the local or Docker setup.
- **Cold starts:** Render's free tier sleeps after 15 minutes of inactivity. The first request after inactivity takes 30 to 60 seconds.
- **Model staleness across replicas:** In the multi-container Docker setup, a promoted retrain updates one container's in-memory model. Other replicas continue serving the previous model until restarted. This is inherent to the architecture and acceptable for demonstration purposes.
- **Data not tracked in git:** The `data/train/` and `data/test/` directories contain `.npy` files generated by running the notebook. They are gitignored to keep the repo lightweight. Run the notebook to regenerate them.
