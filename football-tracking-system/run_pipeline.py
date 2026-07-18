"""
Full Football Tracking Pipeline.

Stages:
  1. YOLO Detection (ball, player, goalkeeper, referee)
  2. ByteTrack Tracking (persistent IDs)
  3. Team Color Classification (Red vs White)
  4. Homography / Pitch Projection (pixel -> meters on 2D pitch)
  5. Ball Possession Detection (temporal window voting)
  6. Pass / Interception Detection (possession state machine)
  7. Player Statistics (distance, speed, possession time, passes)
  8. Heatmaps (per-team Gaussian density on 2D pitch)
  9. Overlay: video frame + minimap + stats HUD

Output:
  - output_full_pipeline.mp4  : annotated video with minimap + stats HUD
  - output_heatmap_A.png      : Team A positional heatmap
  - output_heatmap_B.png      : Team B positional heatmap
  - output_stats.json         : Per-player and per-team statistics JSON
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import json
import numpy as np
from ultralytics import YOLO

from src.team_classification.color_classifier import ColorTeamClassifier
from src.tracking.kalman_ball_filter import KalmanBallFilter
from src.calibration.homography import HomographyEstimator
from src.calibration.pitch_projection import PitchProjector
from src.possession.temporal_window import TemporalWindowPossession
from src.stats.player_stats import PlayerStatsTracker
from src.viz.heatmap import HeatmapGenerator
from src.viz.minimap import MinimapRenderer

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
VIDEO_PATH  = "Copy of A1606b0e6_0 (14).mp4"
OUTPUT_VIDEO = "output_full_pipeline.mp4"
OUTPUT_HM_A  = "output_heatmap_A.png"
OUTPUT_HM_B  = "output_heatmap_B.png"
OUTPUT_STATS = "output_stats.json"
MODEL_PATH   = "runs/detect/runs/yolo11m_v2_improved/weights/best.pt"

# Pitch dimensions (standard FIFA)
PITCH_LEN = 105.0
PITCH_WID = 68.0

# Class IDs from model: {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
CLS_BALL = 0
CLS_GK   = 1
CLS_PLAY = 2
CLS_REF  = 3

# HSV reference colors for team classification
TEAM_HSV = {
    "Team A": np.array([0, 15, 240], dtype=np.float32),    # White kit
    "Team B": np.array([0, 240, 200], dtype=np.float32),   # Red kit
}

# ─────────────────────────────────────────────
# HARDCODED HOMOGRAPHY KEYPOINTS
# These map known pixel positions in the video
# to their real-world pitch meter coordinates.
# Adjust these per video if camera angle differs.
# ─────────────────────────────────────────────
# Format: pixel (x, y) -> pitch (x_m, y_m)
# Using visible pitch markings from the match:
#   top-left corner (~top of frame left)
#   top-right corner
#   bottom-left corner
#   bottom-right corner
#   centre spot
SRC_KEYPOINTS = np.array([
    [240,  62],   # top-left corner area
    [1700,  55],  # top-right corner area
    [100,  930],  # bottom-left corner area
    [1830, 920],  # bottom-right corner area
    [960,  490],  # centre spot
], dtype=np.float32)

DST_KEYPOINTS = np.array([
    [0.0,   0.0],                     # top-left
    [PITCH_LEN, 0.0],                 # top-right
    [0.0,   PITCH_WID],               # bottom-left
    [PITCH_LEN, PITCH_WID],           # bottom-right
    [PITCH_LEN / 2, PITCH_WID / 2],  # centre
], dtype=np.float32)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def draw_hud(frame: np.ndarray, stats_tracker: PlayerStatsTracker,
             possession_id, frame_idx: int, fps: float,
             team_a_poss_s: float, team_b_poss_s: float) -> np.ndarray:
    """Draw HUD overlay in top-right corner of the video frame."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 320, 180
    x0 = w - panel_w - 10
    y0 = 10

    # Semi-transparent dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (10, 10, 10), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    ts = stats_tracker.get_team_summary()
    total_poss = team_a_poss_s + team_b_poss_s
    pct_a = (team_a_poss_s / total_poss * 100) if total_poss > 0 else 50.0
    pct_b = 100.0 - pct_a

    time_s = frame_idx / fps
    mins = int(time_s // 60)
    secs = int(time_s % 60)

    lines = [
        (f"Time: {mins:02d}:{secs:02d}", (200, 200, 200)),
        ("", None),
        (f"Team A (White)", (255, 255, 255)),
        (f"  Poss: {pct_a:.1f}%  Passes: {ts.get('Team A', {}).get('passes_made', 0)}", (200, 200, 200)),
        (f"  Dist: {ts.get('Team A', {}).get('total_distance_m', 0.0):.0f}m", (200, 200, 200)),
        ("", None),
        (f"Team B (Red)", (80, 80, 255)),
        (f"  Poss: {pct_b:.1f}%  Passes: {ts.get('Team B', {}).get('passes_made', 0)}", (200, 200, 200)),
        (f"  Dist: {ts.get('Team B', {}).get('total_distance_m', 0.0):.0f}m", (200, 200, 200)),
    ]

    for i, (text, color) in enumerate(lines):
        if text and color:
            cv2.putText(frame, text, (x0 + 8, y0 + 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # Possession bar
    bar_y = y0 + panel_h - 15
    bar_x0 = x0 + 8
    bar_w = panel_w - 16
    bar_h = 10
    fill_a = int(bar_w * pct_a / 100)
    cv2.rectangle(frame, (bar_x0, bar_y), (bar_x0 + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x0, bar_y), (bar_x0 + fill_a, bar_y + bar_h), (255, 255, 255), -1)
    cv2.rectangle(frame, (bar_x0 + fill_a, bar_y), (bar_x0 + bar_w, bar_y + bar_h), (50, 50, 255), -1)

    return frame


def draw_label(frame, x1, y1, x2, y2, label, box_color):
    """Draw bounding box with label."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    y_lbl = max(y1, label_size[1] + 10)
    cv2.rectangle(frame,
                  (x1, y_lbl - label_size[1] - 5),
                  (x1 + label_size[0], y_lbl + 2),
                  box_color, cv2.FILLED)
    tc = (0, 0, 0) if box_color == (255, 255, 255) else (255, 255, 255)
    cv2.putText(frame, label, (x1, y_lbl - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, tc, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_full_pipeline():
    print("=" * 60)
    print("  Football Tracking Pipeline — All Stages")
    print("=" * 60)

    # ── Load model ───────────────────────────
    print(f"[1/9] Loading YOLO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # ── Init modules ─────────────────────────
    print("[2/9] Initialising pipeline modules...")
    color_clf   = ColorTeamClassifier(team_colors=TEAM_HSV)
    kf_ball     = KalmanBallFilter(dt=0.04, process_noise=1.5, measurement_noise=1.5)
    homo_est    = HomographyEstimator(reproj_threshold=5.0, smooth_alpha=0.85)
    projector   = PitchProjector(pitch_length=PITCH_LEN, pitch_width=PITCH_WID)
    possession  = TemporalWindowPossession(proximity_threshold=60.0, window_frames=8)
    stats_trk   = PlayerStatsTracker(fps=25.0)
    heatmap_gen = HeatmapGenerator(scale=8.0, sigma=1.5)
    minimap     = MinimapRenderer(scale=5.0)

    # ── Compute static homography ─────────────
    print("[3/9] Computing homography matrix...")
    H = homo_est.estimate(SRC_KEYPOINTS, DST_KEYPOINTS)
    H = homo_est.smooth(H)
    if H is None:
        print("  WARNING: Homography estimation failed. Pitch projection disabled.")

    # ── Open video ───────────────────────────
    print(f"[4/9] Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"  ERROR: Could not open {VIDEO_PATH}")
        return

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Reconfigure stats tracker fps
    stats_trk.fps = fps

    # Minimap dimensions
    mm_h = minimap.H
    mm_w = minimap.W

    # Output frame size: video width + minimap width side-by-side
    out_w = W + mm_w
    out_h = max(H_v, mm_h)

    print(f"[5/9] Processing {total} frames ({W}x{H_v} @ {fps:.1f} FPS)...")
    print(f"      Output: {OUTPUT_VIDEO} ({out_w}x{out_h})")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (out_w, out_h))

    # Possession time accumulators
    team_poss_frames = {"Team A": 0, "Team B": 0}

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── YOLO tracking ────────────────────
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        player_positions  = {}   # pixel coords: {pid: (cx_px, cy_px)}
        player_pitch_pos  = {}   # meter coords: {pid: (x_m, y_m)}
        player_teams      = {}   # {pid: team}
        ball_px           = None # ball pixel center
        ball_m            = None # ball meter pos

        if results and results[0].boxes is not None:
            boxes      = results[0].boxes
            xyxys      = boxes.xyxy.cpu().numpy()
            classes    = boxes.cls.cpu().numpy().astype(int)
            confs      = boxes.conf.cpu().numpy()
            track_ids  = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [0]*len(classes)

            ball_candidates = []

            for i in range(len(classes)):
                cls_id  = classes[i]
                x1, y1, x2, y2 = map(int, xyxys[i])
                tid     = track_ids[i]
                conf    = confs[i]

                # Foot-point (base of bounding box)
                foot_px = ((x1 + x2) // 2, y2)

                if cls_id == CLS_PLAY:  # player
                    crop = frame[y1:y2, x1:x2]
                    hsv  = color_clf.get_dominant_hsv(crop)
                    d_A  = color_clf.hsv_distance(hsv, TEAM_HSV["Team A"])
                    d_B  = color_clf.hsv_distance(hsv, TEAM_HSV["Team B"])
                    team = "Team A" if d_A < d_B else "Team B"
                    color = (255, 255, 255) if team == "Team A" else (0, 0, 255)
                    lbl   = f"{'W' if team == 'Team A' else 'R'}{tid}"
                    draw_label(frame, x1, y1, x2, y2, lbl, color)

                    player_positions[tid] = (foot_px[0], foot_px[1])
                    player_teams[tid]     = team

                elif cls_id == CLS_GK:  # goalkeeper
                    draw_label(frame, x1, y1, x2, y2, f"GK{tid}", (255, 255, 0))
                    player_positions[tid] = (foot_px[0], foot_px[1])
                    player_teams[tid]     = "Goalkeeper"

                elif cls_id == CLS_REF:  # referee
                    draw_label(frame, x1, y1, x2, y2, f"Rf{tid}", (0, 200, 0))
                    player_positions[tid] = (foot_px[0], foot_px[1])
                    player_teams[tid]     = "Referee"

                elif cls_id == CLS_BALL:
                    ball_candidates.append((conf, ((x1 + x2) / 2, (y1 + y2) / 2)))

            # ── Ball tracking ─────────────────
            if ball_candidates:
                ball_candidates.sort(reverse=True, key=lambda x: x[0])
                bx, by = ball_candidates[0][1]
                smoothed = kf_ball.update([bx, by])
                ball_px  = (int(smoothed[0]), int(smoothed[1]))
                cv2.circle(frame, ball_px, 5, (0, 140, 255), -1)
                cv2.circle(frame, ball_px, 8, (0, 180, 255), 1)

            # ── Pitch projection ──────────────
            if H is not None:
                for pid, pix in player_positions.items():
                    pm = projector.project_point(pix, H)
                    player_pitch_pos[pid] = pm

                if ball_px is not None:
                    ball_m = projector.project_point(ball_px, H)

            # ── Stats: update positions ───────
            for pid in player_positions:
                pm   = player_pitch_pos.get(pid)
                team = player_teams.get(pid, "Unknown")
                stats_trk.update_position(pid, pm, team)

            # ── Possession ────────────────────
            # Use pixel proximity if no homography, else meter proximity
            poss_positions = player_pitch_pos if H is not None else player_positions
            poss_ball      = ball_m if H is not None else ball_px
            curr_poss, new_events = possession.update(poss_positions, poss_ball, player_teams)

            stats_trk.update_possession(curr_poss)

            if curr_poss is not None:
                team_of_poss = player_teams.get(curr_poss, "")
                if team_of_poss in team_poss_frames:
                    team_poss_frames[team_of_poss] += 1

            # ── Pass events ───────────────────
            for event in new_events:
                stats_trk.update_pass_event(event)

            # ── Heatmap accumulation ──────────
            for pid, pm in player_pitch_pos.items():
                team = player_teams.get(pid, "Unknown")
                heatmap_gen.add_position(pid, team, pm)

        else:
            curr_poss = None

        # ── HUD overlay ───────────────────────
        poss_s_a = team_poss_frames["Team A"] / fps
        poss_s_b = team_poss_frames["Team B"] / fps
        frame = draw_hud(frame, stats_trk, curr_poss, frame_idx, fps, poss_s_a, poss_s_b)

        # ── Minimap render ────────────────────
        mm_frame = minimap.render(
            player_pitch_pos, player_teams,
            ball_pos_m=ball_m,
            possession_id=curr_poss,
            frame_num=frame_idx
        )

        # ── Compose output frame ─────────────
        out_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        out_frame[:H_v, :W] = frame
        out_frame[:mm_h, W:W + mm_w] = mm_frame

        writer.write(out_frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{total} ({frame_idx/total*100:.1f}%)...")

    cap.release()
    writer.release()
    print(f"\n[6/9] Video saved: {OUTPUT_VIDEO}")

    # ── Save heatmaps ─────────────────────────
    print("[7/9] Generating heatmaps...")
    hm_a = heatmap_gen.render_team_heatmap("Team A")
    hm_b = heatmap_gen.render_team_heatmap("Team B")
    cv2.imwrite(OUTPUT_HM_A, hm_a)
    cv2.imwrite(OUTPUT_HM_B, hm_b)
    print(f"  Saved: {OUTPUT_HM_A}")
    print(f"  Saved: {OUTPUT_HM_B}")

    # ── Save stats JSON ───────────────────────
    print("[8/9] Saving statistics...")
    stats_out = {
        "per_player": stats_trk.get_summary(),
        "per_team": stats_trk.get_team_summary(),
        "possession_seconds": {
            "Team A": round(poss_s_a, 2),
            "Team B": round(poss_s_b, 2),
        },
        "total_passes": {
            team: ts.get("passes_made", 0)
            for team, ts in stats_trk.get_team_summary().items()
        },
        "total_events": len(possession.events),
    }

    # Convert int keys to str for JSON
    stats_out["per_player"] = {str(k): v for k, v in stats_out["per_player"].items()}

    with open(OUTPUT_STATS, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"  Saved: {OUTPUT_STATS}")

    # ── Print summary ─────────────────────────
    print("\n[9/9] ── MATCH SUMMARY ──────────────────────")
    ts = stats_trk.get_team_summary()
    total_poss = poss_s_a + poss_s_b
    for team, data in ts.items():
        poss_pct = (data["possession_s"] / total_poss * 100) if total_poss > 0 else 0
        print(f"  {team}:")
        print(f"    Possession : {poss_pct:.1f}%")
        print(f"    Distance   : {data['total_distance_m']:.0f} m")
        print(f"    Passes     : {data['passes_made']}")
        print(f"    Intercepts : {data['interceptions']}")
    print(f"\n  Total events detected: {len(possession.events)}")
    print("\nPipeline complete! ✅")


if __name__ == "__main__":
    run_full_pipeline()
