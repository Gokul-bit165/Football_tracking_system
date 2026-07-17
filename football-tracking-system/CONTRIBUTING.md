# Contributing Guidelines & Team Collaboration Flow

This document details the branching model, task distribution, and integration plan for our Football Player/Ball Tracking Research Project.

---

## 1. 3-Way Collaborative Task Split

To work in parallel immediately without overlapping commits and merge conflicts, the project tasks are split into three dedicated feature branches off the `develop` branch:

### Person A: Detection Training
- **Branch:** `feature/detection-training`
- **Scope:**
  - Benchmark detectors: YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO12, and RT-DETR.
  - Implement and integrate the stride-4 P2 small-object detection head to improve ball class recall.
  - Set up dataset mix configurations and augmentation options.
- **MLflow Logging:**
  - Log all runs, metrics, and models to the MLflow experiment name: **`detection-benchmark`**
- **Artifact Handoff:**
  - Person A will save checkpoints/runs. Once a baseline YOLOv8n detector checkpoint is ready, Person A drops it in `models/baseline/yolov8n_baseline.pt` (this path is gitignored, see `models/baseline/README.md`) so Person B can begin tracker benchmarking.

---

### Person B: Tracking Experiments
- **Branch:** `feature/tracking-experiments`
- **Scope:**
  - Benchmark trackers: DeepSORT, ByteTrack, BoT-SORT (with Camera Motion Compensation), and OC-SORT (specifically for ball tracking).
  - Use a frozen detector checkpoint to isolate tracking performance. Configure the path to the detector checkpoint under the training configuration (e.g. `configs/train/`).
- **MLflow Logging:**
  - Log all runs, tracking parameters, and tracking metrics (HOTA, IDF1, MOTA, MOTP) to the MLflow experiment name: **`tracking-benchmark`**

---

### Person C: Semantic & Downstream System
- **Branch:** `feature/semantic-system`
- **Scope:**
  - **Team Classification:** Implement the lightweight CNN classifier and zero-shot CLIP-based classification on player jersey crops.
  - **Possession Estimation:** Build the temporal window nearest-player method and the probabilistic Bayesian possession state estimator.
  - **Trajectory Prediction:** Implement the real-time GRU predictor (for short horizons) and the sequence-to-sequence Transformer predictor (for offline play tracking).
  - **Evaluation Harness:** Write the full metrics harness (`src/eval/`) for evaluating detection, tracking, and system-level performance against a small sample dataset.
- **MLflow Logging:**
  - Log all module tests and runs to the MLflow experiment name: **`semantic-system-dev`**

---

## 2. Integration and Full Ablation Study

Once Person A has a winning detection checkpoint and Person B has finalized the tracking configuration:

1. **Merge Branch:** Both Person A and Person B merge their feature branches back into `develop`, resolve conflicts, and then branch or merge into `integration/full-pipeline`.
2. **End-to-End Evaluation:** Execute the Exp7 configuration (combining YOLO11-P2 + BoT-SORT + OC-SORT + CLIP team classifier + Bayesian possession + GRU trajectory prediction).
3. **MLflow Logging:** Log the final full-system ablation run to the MLflow experiment name: **`full-ablation-study`**

---

## 3. Branching Guidelines

- **`main`**: Production/deployment-ready pipeline.
- **`develop`**: Active integration branch.
- **`feature/*`**: Work on specific features. Do not commit directly to `develop` or `main`.
- Always open a Pull Request (PR) to merge features into `develop` and request reviews from the other collaborators.
