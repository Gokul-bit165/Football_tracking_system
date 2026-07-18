# Technical Deep Dive: Methods, Math, and Architecture

This document provides a comprehensive breakdown of the core technologies, algorithms, and mathematical concepts underpinning our Football Tracking System.

---

## 1. Object Detection: YOLO11m
### Why YOLO11?
We chose **YOLO11** (specifically the medium profile `yolo11m`) because of its state-of-the-art balance between inference speed and detection accuracy (mAP). Compared to older architectures (like YOLOv8 or YOLOv5):
- **C3k2 Blocks**: Enhanced cross-stage partial network blocks that extract richer features with fewer computational steps.
- **Anchor-Free Design**: Direct coordinates regression of bounding box boundaries, which drastically improves detection convergence and handling of varying scales (players near vs. far).
- **SPPF (Spatial Pyramid Pooling - Fast)**: Pools multi-scale features without loss of speed, critical for detecting the tiny scale of the ball and larger player bounding boxes in the same frame.

### Custom Training Details:
* **Feature Resolution (`imgsz: 1280`)**: Standard YOLO uses 640px. Because the ball is tiny, it often gets downsampled to less than a single pixel in deep feature maps. Running at 1280px preserves structural details (the ball goes from $6\times 6$ to $12\times 12$ pixels in input space), allowing the network's convolutional kernels to detect it.
* **Loss Functions Optimization**:
  - **CIoU (Complete Intersection over Union) Loss**: Penalizes aspect ratio and scale differences between predicted and ground truth boxes rather than just coordinate offsets.
  - **DFL (Distribution Focal Loss)**: Optimizes the probability distribution of box borders, crucial for handling occlusion boundaries when players overlap.
  - **Custom Loss Weighting**: We set the box loss regression weight to `9.0` (up from default `7.5`) to penalize imperfect bounding box crops of jersey colors.

---

## 2. Multi-Object Tracking: ByteTrack
### Why ByteTrack instead of SORT/DeepSORT?
* **SORT** fails when a player is occluded (e.g. one player runs behind another) because it discards detections below a strict confidence threshold (e.g. 0.3).
* **DeepSORT** relies on an additional Re-ID convolutional neural network to match visual features. This adds significant latency (~15ms per player) and struggles under changes in body posture or shadows.
* **ByteTrack** solves this mathematically using a **double association logic**:
  1. It first matches highly confident bounding boxes ($\text{conf} > 0.5$) using **Intersection over Union (IoU)** association via Hungarian algorithm matching.
  2. For unmatched tracks, it attempts a second association pass against low-confidence detections ($0.1 < \text{conf} < 0.5$).
  3. This association uses a **Kalman Filter state prediction** to predict where the player *should* be, and matches them to the weak detection at that location.

This allows our system to track players through dense clusters and physical contact without losing their persistent identity track ID.

---

## 3. Team Classification: HSV K-Means
### The Method:
Rather than training a heavy Deep Learning classifier for kits (which would require a massive labeled dataset of kits and fail on new matches), we designed a zero-shot physical approach:
1. **Upper Crop**: We crop the top 45% of the bounding box to isolate the jersey and exclude socks, boots, or pitch background.
2. **Grass Masking**: Convert the cropped image to the **HSV (Hue, Saturation, Value)** color space. Green grass pixels are mathematically removed using an upper/lower HSV threshold:
   $$\text{Green Range: } H \in [35, 85], S \in [40, 255], V \in [40, 255]$$
3. **K-Means Clustering ($k=2$)**: We fit K-Means to the remaining non-masked pixels. The cluster with the largest coordinate count represents the dominant jersey color.
4. **Weighted HSV Distance**: We compute the distance to pre-defined reference team colors ($C_{\text{ref}}$) using a weighted Euclidean metric to prioritize Hue over Saturation/Value:
   $$D = \sqrt{w_h (H - H_{\text{ref}})^2 + w_s (S - S_{\text{ref}})^2 + w_v (V - V_{\text{ref}})^2}$$
5. **Strict Binary Assignment**: Every player is mapped to the closer team vector, ensuring stable team labels even in shadow zones.

---

## 4. Homography & 2D Projection
### The Mathematics of Perspective:
Broadcast footage is shot from an elevated, angled camera. Pixel movement is non-linear relative to physical distance. To resolve this, we use a **Homography Matrix** ($H$), a $3\times3$ transformation matrix that maps coordinates from the 2D image plane to the 2D pitch plane.

Given a point in the image plane $(u, v)$ and its corresponding point on the real-world pitch $(x, y)$:
$$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$
$$x = \frac{x'}{w'}, \quad y = \frac{y'}{w'}$$

### Robust Estimation via RANSAC:
To estimate the 8 degrees of freedom in $H$, we select known landmarks (corners, circles, lines). Since manual pixel selection can introduce error, we compute $H$ using **RANSAC (Random Sample Consensus)**. RANSAC randomly selects subsets of points, computes candidate matrices, and counts "inliers" (points that map correctly within a tolerance threshold). It selects the matrix that maximizes the inlier count, discarding human selection errors.

---

## 5. Ball Tracking: Kalman Filter & Hybrid Drawing
### The Ball State Space Model:
The ball is modeled as a linear dynamic system with state vector $X_t$ and measurement vector $Z_t$:
$$X_t = \begin{bmatrix} x_t & y_t & v_{x,t} & v_{y,t} \end{bmatrix}^T$$
$$Z_t = \begin{bmatrix} z_{x,t} & z_{y,t} \end{bmatrix}^T$$

1. **Prediction Step (State Extrapolation)**:
   $$X_{t|t-1} = F X_{t-1|t-1}$$
   $$P_{t|t-1} = F P_{t-1|t-1} F^T + Q$$
   Where $F$ is the transition matrix (constant velocity model) and $Q$ is the process noise covariance.
2. **Update Step (Measurement Correction)**:
   $$K_t = P_{t|t-1} H^T (H P_{t|t-1} H^T + R)^{-1}$$
   $$X_{t|t} = X_{t|t-1} + K_t (Z_t - H X_{t|t-1})$$
   $$P_{t|t} = (I - K_t H) P_{t|t-1}$$
   Where $K$ is the Kalman gain, $R$ is the measurement noise covariance, and $H$ is the measurement mapping matrix.

### The Hybrid Rendering Solution:
Football velocity is discontinuous (kicks). To prevent the Kalman velocity state from dragging the visual dot away from the actual ball position:
- **During Detection**: We bypass the state position and draw the raw visual coordinates directly (`bx, by`), applying a light **Exponential Moving Average (EMA)** to filter out coordinate quantization noise:
  $$P_{\text{drawn}, t} = 0.85 P_{\text{detected}, t} + 0.15 P_{\text{drawn}, t-1}$$
- **During Occlusion**: We fall back on $X_{t|t-1}$ (the Kalman prediction) to estimate where the ball traveled, preventing the dot from disappearing when blocked by a player.

---

## 6. Game Events State Machine
### Temporal Window Possession
A simple distance check is highly unstable due to pixel noise. We implement a **temporal majority vote window** of size $W_f = 8$ frames:
$$\text{Owner}_t = \text{Mode}(\text{ClosestPlayer}_{t-7}, \dots, \text{ClosestPlayer}_t)$$
Possession is only shifted to a player if they are the closest to the ball for the majority of the sliding window.

### Pass Detection
A state transition tracks possession changes:
```
[Team A Player P1 Has Possession]
               ↓
     [Ball distance from all players > Proximity Threshold]
               ↓
    [Pass Candidate Registered]
               ↓
  [Possession secured by Player P2]
         ├── If Team(P2) == Team(P1) ──> Pass Successful
         └── If Team(P2) != Team(P1) ──> Interception
```
This state machine ignores referee coordinates to prevent false possession data.
