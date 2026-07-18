# Hackathon Presentation Guide: The Story of Our Football Tracking System

Here is a structured, compelling narrative you can use for your hackathon pitch. It is designed to take the judges on a journey: from the initial problem to the engineering hurdles, the solutions, and the final results.

---

##  Slide 1: The Vision (The "What" and "Why")
* **The Pitch:** Broadcast football videos are a goldmine of data, but raw footage is unstructured. Professional systems like Hawk-Eye or StatsBomb cost millions and require multi-camera hardware. Our goal was to build a system that extracts professional-grade, 2D top-down positional analytics from **any single broadcast video camera feed** using state-of-the-art open-source AI.
* **The Value:** Democratizing elite sports analytics for local clubs, academy players, and content creators.

---

## Slide 2: The Core Approach (The Architecture)
We built a **9-stage pipeline** designed to go from raw pixels to structured sports data:
1. **Detection (YOLO11m)**: Custom-trained to find players, goalkeepers, referees, and the ball.
2. **Tracking (ByteTrack)**: Maintains player identities across frame changes.
3. **Team Classification (HSV K-Means)**: Automatically separates teams based on upper jersey colors.
4. **Homography (RANSAC & EMA)**: Map perspective-distorted pixel coordinates to actual metric pitch positions.
5. **Trajectory Smoothing (Kalman Filtering)**: Mitigate small object jitter and player occlusions.
6. **Possession State Machine**: Temporal sliding window logic to determine who has the ball.
7. **Pass/Interception Detection**: Tracks sequence transfers between teams.
8. **Positional Heatmaps**: 2D Gaussian density plots showing spatial patterns.
9. **Interactive Dashboard**: A self-contained web portal using Chart.js to inspect metrics.

---

## Slide 3: Engineering Hurdles & How We Overcame Them
*Judges love details about what failed and how you solved it. Here are the 4 key technical stories of this project:*

### Story 1: The Tiny Object Dilemma (Ball Detection)
* **The Problem:** The football is a tiny object (~10x10 pixels) and moves at high speed, causing severe motion blur. Standard object detection confidence thresholds (0.25) missed the ball in over half of the frames.
* **Our Approach:** Instead of a single inference pass, we implemented a **dual-pass inference pipeline**. We run tracking for players at `0.20` confidence, and run a separate, targeted detector for the ball at a very low threshold of `0.10`.
* **The Result:** Reclaimed over 40% of missed ball detections while ignoring background noise by using box-size constraints.

### Story 2: The Kalman Filter Inertia Lag (Unsynced Ball Dot)
* **The Problem:** Constant-velocity Kalman filters expect smooth movements. But when the ball is kicked, it experiences instant acceleration. The filter's velocity memory caused the tracked orange dot to lag behind the physical ball by 15-20 pixels, looking unsynced.
* **Our Approach:** We built a **Hybrid Renderer**. When the ball is detected, we draw the visual dot directly at the raw pixel coordinates (with a light 2-frame exponential moving average to prevent subpixel jitter). We feed this coordinate into the Kalman filter to keep its state updated. We fall back to the Kalman filter predictions **only when the ball is hidden/occluded**.
* **The Result:** 100% precise alignment on the ball during active play, with seamless interpolation when the ball goes behind players.

### Story 3: The YOLO State Collision Bug
* **The Problem:** We originally tried to optimize the script by running `model.predict()` and `model.track()` on the same YOLO object instance. This caused ByteTrack's internal memory state to corrupt, resetting all player IDs to 0.
* **Our Approach:** We decoupled the pipeline, initializing **two separate, isolated YOLO instances** (one dedicated solely to tracking players, and one dedicated to prediction classes for the ball).
* **The Result:** Player identities stayed completely stable across the entire video, allowing correct accumulation of distance and speed.

### Story 4: The 2D Projection Transform (Homography)
* **The Problem:** Cameras tilt and zoom, and broadcast angles distort distance. 10 pixels on the far side of the pitch represents a much larger distance than 10 pixels close to the camera.
* **Our Approach:** We mapped key physical pitch coordinates (corners, goal lines, center spot) to a standard 105m x 68m FIFA pitch template. Using RANSAC Homography combined with exponential matrix smoothing, we transformed camera pixels into actual meter coordinates.
* **The Result:** Accurate tracking of player distance covered and instantaneous speeds.

---

## Slide 4: Results & Impact
* **Accuracy:** Our custom-tuned YOLO11m v2 model achieved an overall player tracking **mAP50 of 0.873** (with player class reaching **0.989**).
* **The Data:** From a simple 30-second clip, we successfully extracted **49 unique game events** (passes, interceptions, possession switches) along with team heatmaps.
* **The Interface:** Generated a self-contained, responsive dark-mode dashboard that works entirely offline, bypassing CORS security restrictions.

---

## Pitch Tips:
1. **Start with the "Why":** Show a 3-second raw video clip vs. your output video side-by-side with the minimap. Let them see the visual impact first.
2. **Emphasize Custom Solutions:** Mention that you didn't just wrap an API; you handled Kalman noise tuning, K-Means kit classification, and manual homography matrix computation.
3. **Show, Don't Just Tell:** Keep the standalone dashboard open and scroll through the interactive player cards.
