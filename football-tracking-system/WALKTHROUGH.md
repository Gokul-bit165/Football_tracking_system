# Football Tracking System — Project Walkthrough

> **Author:** Gokul  
> **GitHub:** [Gokul-bit165/Football_tracking_system](https://github.com/Gokul-bit165/Football_tracking_system)  
> **Tech Stack:** Python · YOLOv11m · ByteTrack · Kalman Filter · OpenCV · Chart.js  

---

## Overview

This project builds a **multi-stage computer vision pipeline** that transforms raw broadcast football footage into structured, actionable analytics — similar in concept to commercial systems like **Hawk-Eye**, **StatsBomb**, and **SkillCorner**, but built entirely from scratch using open-source tools.

Given a raw match video, the system automatically:
- Detects and tracks every player, goalkeeper, referee, and the ball
- Assigns persistent identity IDs across the entire video
- Classifies each player into their correct team by jersey color
- Projects all pixel positions onto a real-world 2D pitch map (meters)
- Detects ball possession, passes, and interceptions
- Computes per-player and per-team statistics
- Generates positional heatmaps
- Renders a real-time bird's-eye minimap alongside the annotated video
- Outputs an interactive web analytics dashboard

---

## Pipeline Architecture

```
Raw Video Input
      │
      ▼
┌─────────────────┐
│  Stage 1        │  YOLO11m Object Detection
│  Detection      │  4 classes: ball · player · goalkeeper · referee
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 2        │  ByteTrack Multi-Object Tracker
│  Tracking       │  Persistent track IDs across frames
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 3        │  HSV K-Means Color Classifier
│  Team Class.    │  Jersey crop → Team A (White) vs Team B (Red)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 4        │  RANSAC Homography + Temporal Smoothing
│  Pitch Proj.    │  Pixel (u,v) → Real-world meters (x,y) on 2D pitch
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 5        │  Kalman Filter (constant velocity model)
│  Ball Smoothing │  Smooth noisy ball detections, handle occlusions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 6        │  Temporal Window Possession (majority voting)
│  Possession     │  State machine: possession → pass → new owner
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 7        │  PlayerStatsTracker
│  Statistics     │  Distance · Speed · Possession · Passes · Intercepts
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 8        │  HeatmapGenerator (Gaussian density accumulation)
│  Heatmaps       │  Per-team positional density on standard 105×68m pitch
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 9        │  MinimapRenderer + Stats HUD + Web Dashboard
│  Visualization  │  Annotated video + bird's-eye minimap + charts
└─────────────────┘
```

---

## Stage 1 — Object Detection (YOLO11m)

### Model
We use **YOLOv11m** (medium), trained on a custom football dataset using **Ultralytics**.

The model detects **4 classes**:

| Class ID | Label | Color in Output |
|----------|-------|----------------|
| 0 | ball | Orange dot |
| 1 | goalkeeper | Cyan box |
| 2 | player | White or Red box |
| 3 | referee | Green box |

### Training

Two training runs were conducted:

**Run 1 — Baseline** (`yolo11m_baseline-4`):
- 20 epochs · image size 640 · Adam optimizer
- mAP50: 0.821

**Run 2 — Improved v2** (`yolo11m_v2_improved`):
- 35 epochs (early stopped at 30, best checkpoint at epoch 20)
- Image size 1280 · AdamW · LR: 0.005 · box loss weight: 9.0
- Heavy augmentation: mosaic, HSV jitter, scale, flip
- All experiments tracked with **MLflow**

**Final Validation Results (v2 improved):**

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| **all** | 0.954 | 0.800 | **0.873** | **0.629** |
| ball | 0.913 | 0.529 | 0.588 | 0.285 |
| goalkeeper | 0.982 | 0.769 | 0.954 | 0.766 |
| player | 0.962 | 0.982 | **0.989** | **0.805** |
| referee | 0.960 | 0.919 | 0.961 | 0.659 |

> [!NOTE]
> The ball has the lowest mAP (0.588) due to its tiny size (~10×10 px) and motion blur at broadcast resolution — an expected challenge for anchor-based detectors.

---

## Stage 2 — Multi-Object Tracking (ByteTrack)

### Problem
YOLO produces a fresh set of bounding boxes per frame with no temporal memory. Without tracking, every player gets a new random ID each frame — making trajectories and statistics impossible.

### ByteTrack
**ByteTrack** is a state-of-the-art SORT-family tracker that:
- Associates detections to existing tracks via **IoU matching**
- Unlike other trackers, uses **both high-confidence and low-confidence detections**, making it robust against occlusion and brief disappearances
- Maintains **persistent track IDs** (e.g. Player #18 keeps ID 18 for their entire visible trajectory)

```python
results = model.track(frame, persist=True, tracker="bytetrack.yaml")
```

---

## Stage 3 — Team Color Classification

### Problem
YOLO detects the class "player" but has no concept of which team the player belongs to. This needs to be inferred from jersey color.

### Method: HSV K-Means Jersey Classifier

For each detected player bounding box:

1. **Crop the upper 45%** of the box (jersey region only, avoiding shorts/pitch)
2. **Convert to HSV** color space (more robust to lighting changes than RGB)
3. **Mask out green grass** using an HSV green range filter `[H:35-85, S:40-255]`
4. Run **K-Means clustering (k=2)** to find the 2 dominant colors in the jersey patch
5. Select the **largest cluster center** as the dominant jersey color
6. Compute **weighted HSV Euclidean distance** to reference team colors:
   - Team A reference: White `[H=0, S=15, V=240]`
   - Team B reference: Red `[H=0, S=240, V=200]`
7. Assign to the **closer team** — strict binary classification

> Goalkeepers (class 1) and Referees (class 3) are directly labeled from YOLO's class output and bypass color classification entirely.

---

## Stage 4 — Homography / Pitch Projection

### Problem
Pixel coordinates are perspective-distorted. A player 5 m away from another in reality may appear very close or far on screen depending on camera angle. To compute real distances, speeds, and heatmaps, pixel positions must be mapped to **real-world metre coordinates**.

### Method: RANSAC Homography

We define a set of **manually calibrated pitch keypoints** — visible field markings (corner flags, penalty spots, centre spot) — mapped to their known real-world positions on a standard **105 × 68 m FIFA pitch**.

Using OpenCV's `findHomography` with **RANSAC** outlier rejection:
```python
H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_threshold=5.0)
```

Any pixel coordinate can then be projected:
```python
p_homog = H @ [u, v, 1].T
x_m, y_m = p_homog[0] / p_homog[2], p_homog[1] / p_homog[2]
```

**Temporal smoothing** via exponential moving average (α=0.85) prevents jitter in H between frames.

---

## Stage 5 — Ball Smoothing (Kalman Filter)

### Problem
Raw ball detection coordinates are noisy (±10-20 px jitter), and the ball frequently disappears behind players for 5-10 frames.

### Solution
A **constant-velocity Kalman filter** with state vector `[x, y, vx, vy]`:
- **Predict step**: extrapolate expected position using velocity during missed detections
- **Update step**: correct the prediction using the measured detection position
- **Result**: smooth, stable ball trajectory even during short occlusions

---

## Stage 6 — Ball Possession & Pass Detection

### Possession Algorithm (Temporal Window Voting)

```
Per frame:
  1. Find closest player to the ball (Euclidean distance in metre space)
  2. Add to a sliding window buffer (last 8 frames)
  3. Majority vote → only assign possession if player appears in ≥50% of window
  4. Referees are excluded from possession
```

### Pass / Interception State Machine

```
[Player P1 has possession]
        ↓  P1 loses ball (kicks/releases)
[Pass candidate: (P1, TeamA, frame=F)]
        ↓  New player P2 closest to ball
[Pass completed]
   TeamA == TeamB? → "success" pass
   TeamA != TeamB? → "intercepted"
```

Every detected event is stored with: `from_id`, `to_id`, `team_from`, `team_to`, `status`, `frame_start`, `frame_end`.

---

## Stage 7 — Player Statistics

The `PlayerStatsTracker` accumulates per-player metrics across the full video:

| Stat | Calculation Method |
|------|--------------------|
| **Distance (m)** | Sum of frame-to-frame metre displacements (clamped at 12 m/s to remove noise) |
| **Top Speed (m/s)** | Maximum single-frame displacement ÷ dt |
| **Avg Speed (m/s)** | Mean of all valid per-frame speed samples (SMA-5 smoothed) |
| **Possession Time (s)** | Possession frames ÷ FPS |
| **Passes Made** | Count of events where player = sender |
| **Passes Received** | Count of events where player = receiver (non-intercepted) |
| **Interceptions** | Count of events where player received an intercepted ball |

---

## Stage 8 — Positional Heatmaps

For each frame, each player's metre position is accumulated into a 2D density map:

```python
blob[py, px] = 1.0
blob = GaussianBlur(blob, sigma=1.5m)
team_density += blob
```

At video end:
- Density map is normalised to `[0, 255]`
- Rendered with `cv2.COLORMAP_JET` (blue = low, red = high density)
- Overlaid on a rendered 2D pitch canvas with all FIFA markings

---

## Stage 9 — Visualization & Dashboard

### Annotated Video (`output_full_pipeline.mp4`)
- **Bounding boxes**: color-coded by team (White / Red / Cyan / Green) with track ID
- **Ball marker**: glowing orange dot at Kalman-smoothed position
- **Stats HUD**: top-right overlay — possession %, distance, passes per team
- **Bird's-eye minimap**: side-by-side 2D pitch showing all player positions + ball in real-time

### Web Analytics Dashboard (`dashboard_standalone.html`)
Fully self-contained (no server, no CDN, no CORS issues):
- Chart.js **embedded inline**
- JSON stats **embedded as a JS variable**
- Heatmap images **embedded as base64 PNG**

**Dashboard panels:**
- Possession % KPIs and animated possession bar
- Doughnut chart (Team A vs Team B possession split)
- Horizontal bar chart: Top 10 players by distance
- Horizontal bar chart: Top 10 players by top speed
- Team A and Team B full-pitch heatmaps
- Filterable per-player stats table (sortable by team)

---

## Match Results (Sample Video)

> **Video:** `Copy of A1606b0e6_0 (14).mp4` — ~30 sec broadcast footage  
> **Resolution:** 1920 × 1080 · **FPS:** 25 · **Frames:** 749  

| Metric | Team A (White) | Team B (Red) |
|--------|:-------------:|:-----------:|
| **Ball Possession** | **71.9%** | 28.1% |
| **Total Distance** | 1,440 m | 1,029 m |
| **Passes Made** | 7 | 4 |
| **Interceptions** | 2 | 3 |

**Total events detected:** 11 · **Unique track IDs:** 138

---

## Repository Structure

```
football-tracking-system/
│
├── run_pipeline.py              ← Master entry point (all 9 stages)
├── generate_dashboard.py        ← Self-contained dashboard generator
├── dashboard_standalone.html    ← Output: embedded analytics dashboard
├── WALKTHROUGH.md               ← This document
│
├── src/
│   ├── calibration/
│   │   ├── homography.py        ← RANSAC homography + EMA smoothing
│   │   └── pitch_projection.py  ← Pixel→metres, distance, speed
│   ├── team_classification/
│   │   └── color_classifier.py  ← HSV K-Means jersey classifier
│   ├── tracking/
│   │   ├── bytetrack_wrapper.py ← ByteTrack integration
│   │   └── kalman_ball_filter.py← Constant-velocity Kalman filter
│   ├── possession/
│   │   └── temporal_window.py   ← Possession voting + pass state machine
│   ├── stats/
│   │   └── player_stats.py      ← Per-player and team stats accumulator
│   └── viz/
│       ├── heatmap.py           ← Gaussian heatmap on 2D pitch
│       └── minimap.py           ← Real-time bird's-eye minimap
│
├── configs/train/
│   ├── yolo11m_train.yaml       ← Baseline training config
│   └── yolo11m_v2_improved.yaml ← Improved training config (v2)
│
├── data/processed/v1/cleaned_dataset/
│   └── data.yaml                ← 4-class dataset (915 train / 37 val images)
│
└── experiments/mlflow/          ← MLflow experiment tracking DB
```

---

## How to Run

### Run the full pipeline on a new video

```bash
# Edit VIDEO_PATH in run_pipeline.py first, then:
python run_pipeline.py
```

**Outputs:**
| File | Description |
|------|-------------|
| `output_full_pipeline.mp4` | Annotated video + minimap |
| `output_heatmap_A.png` | Team A positional heatmap |
| `output_heatmap_B.png` | Team B positional heatmap |
| `output_stats.json` | Full match statistics JSON |

### Generate the analytics dashboard

```bash
python generate_dashboard.py
# Opens dashboard_standalone.html automatically
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **YOLO11m over YOLO11s** | Better small-object mAP (ball, distant players) |
| **ByteTrack over SORT/DeepSORT** | No re-ID model needed; robust to low-confidence detections during occlusion |
| **HSV K-Means over CNN classifier** | Zero-shot, zero training data needed — works on any new match with any two-team color scheme |
| **Kalman filter over naive smoothing** | Principled velocity prediction handles multi-frame ball occlusions |
| **Manual homography calibration** | Reliable and fast for fixed broadcast camera angles |
| **Self-contained HTML dashboard** | No CORS issues, no server, works offline — double-click to open |

---

## Limitations & Future Work

| Limitation | Future Improvement |
|------------|--------------------|
| Ball mAP50 = 0.588 | Add dedicated ball-only training data; use deformable DETR |
| Static homography (fixed camera) | Auto pitch keypoint detection using semantic segmentation |
| Color classification can fail in shadow/poor lighting | Fine-tune a lightweight CNN jersey classifier |
| Per-clip stats only | Add video stitching for full-match accumulation |
| Numeric player IDs (not real names) | Add jersey number OCR to resolve real player identities |

---

*Built entirely with open-source tools. No licensed sports data APIs were used.*
