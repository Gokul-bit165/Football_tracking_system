# Football Player & Ball Tracking System

A state-of-the-art AI-powered research pipeline for tracking players and the ball in broadcast football videos. The system combines modern object detection, motion-centric multi-object tracking, camera motion compensation, keypoint-based pitch homography projection, team color classification, possession estimation, and short-horizon trajectory forecasting.

For information on team roles, task splits, branching, and MLflow logging policies, see [CONTRIBUTING.md](file:///c:/Users/gokul/football_tracker/football-tracking-system/CONTRIBUTING.md).

---

## Project Structure

```text
football-tracking-system/
├── configs/
│   ├── data/                  # Dataset yaml configs
│   ├── models/                # Model architecture configs
│   ├── train/                 # Training hyperparameters per experiment
│   └── experiments/           # configs for Exp1 through Exp7
├── data/
│   ├── raw/                   # Original downloaded datasets (Gitignored)
│   ├── interim/               # Master frame extractions & COCO annotations (Gitignored)
│   └── processed/             # Cleaned YOLO splits
├── src/
│   ├── data/                  # Preprocessing, phash deduplication, blur filtering
│   ├── detection/             # YOLO wrappers & custom P2 stride-4 heads
│   ├── tracking/              # ByteTrack, BoT-SORT, OC-SORT wrappers & Kalman filters
│   ├── team_classification/   # CLIP & CNN jersey classification
│   ├── calibration/           # Keypoint models & RANSAC homography projection
│   ├── possession/            # Nearest-player & Bayesian possession estimation
│   ├── trajectory/            # GRU & Transformer predictors
│   ├── pipeline/              # End-to-end video processing orchestration
│   └── eval/                  # Metric computation (mAP, HOTA, IDF1, etc.)
├── experiments/
│   └── mlflow/                # MLflow tracking stores & local artifacts (Gitignored)
├── deployment/
│   ├── docker/                # Training & inference Dockerfiles
│   ├── onnx_export.py         # Model export script
│   └── tensorrt_build.py      # TensorRT compiler configurations
├── dashboard/
│   ├── backend/               # FastAPI WebSocket inference server
│   └── frontend/              # Interactive analytics dashboard UI
├── tests/                     # Unit & integration testing suites
└── notebooks/                 # Exploratory Jupyter notebooks
```

---

## Getting Started

### 1. Repository Setup

Clone the repository and navigate to the project directory:
```bash
git clone <repository_url>
cd football-tracking-system
```

### 2. Environment Setup

#### Option A: Virtual Environment (Standard Python)
Create and activate a clean virtual environment, then install dependencies:
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.venv\Scripts\Activate.ps1
# Activate on Windows CMD:
.venv\Scripts\activate
# Activate on Unix/Git Bash:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Conda Environment (Recommended for GPU support)
If you have a GPU-enabled environment pre-configured (e.g. named `ai`), activate it directly:
```bash
conda activate ai
# Install dependencies if any are missing
pip install -r requirements.txt
```

### 3. Environment Variables Configuration
Copy the sample environment variables and configure them:
```bash
cp .env.example .env
```
Inside `.env`, set `MLFLOW_TRACKING_URI` to point to the shared tracking server.

### 4. MLflow tracking server

To launch the MLflow tracking server locally:
- **Unix Shell / Git Bash / WSL:**
  ```bash
  chmod +x start_mlflow_server.sh
  ./start_mlflow_server.sh
  ```
- **Windows Command Prompt / PowerShell:**
  ```cmd
  start_mlflow_server.bat
  ```

Once running, the MLflow UI will be accessible at: `http://localhost:5000` (or `http://<shared-host>:5000`).

---

## Baseline Checkpoint Handoff
An initial baseline model directory is located under `models/baseline/`.
- Person A will drop a YOLOv8n baseline model (`.pt` weights) here.
- Person B will load it to run tracker evaluations.
- This directory is gitignored to avoid pushing large binary files.
