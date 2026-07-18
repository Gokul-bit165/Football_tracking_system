"""
Heatmap Generator.
Creates per-player or per-team positional heatmaps on a standard 2D football pitch.
"""
import numpy as np
import cv2
from collections import defaultdict


# Standard pitch dimensions in meters
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


def generate_pitch_canvas(scale: float = 8.0) -> np.ndarray:
    """
    Creates a blank green football pitch canvas with lines.

    Args:
        scale: Pixels per meter.

    Returns:
        BGR pitch image (numpy array).
    """
    W = int(PITCH_LENGTH * scale)
    H = int(PITCH_WIDTH * scale)

    # Dark green grass background
    canvas = np.full((H, W, 3), (34, 85, 34), dtype=np.uint8)

    # Alternate stripe pattern
    stripe_width = int(10 * scale)
    for x in range(0, W, stripe_width * 2):
        canvas[:, x:min(x + stripe_width, W)] = (40, 95, 40)

    lc = (255, 255, 255)  # Line color
    lw = 2                 # Line width

    def m2px(mx, my):
        return (int(mx * scale), int(my * scale))

    # Outer boundary
    cv2.rectangle(canvas, m2px(0, 0), m2px(PITCH_LENGTH, PITCH_WIDTH), lc, lw)

    # Halfway line
    cv2.line(canvas, m2px(PITCH_LENGTH / 2, 0), m2px(PITCH_LENGTH / 2, PITCH_WIDTH), lc, lw)

    # Centre circle (radius ~9.15m)
    cx, cy = m2px(PITCH_LENGTH / 2, PITCH_WIDTH / 2)
    cv2.circle(canvas, (cx, cy), int(9.15 * scale), lc, lw)
    cv2.circle(canvas, (cx, cy), 3, lc, -1)

    # Penalty areas (left and right)
    for side_x in [0, PITCH_LENGTH - 16.5]:
        cv2.rectangle(canvas, m2px(side_x, (PITCH_WIDTH - 40.32) / 2),
                      m2px(side_x + 16.5, (PITCH_WIDTH + 40.32) / 2), lc, lw)

    # Goal areas (6-yard boxes)
    for side_x in [0, PITCH_LENGTH - 5.5]:
        cv2.rectangle(canvas, m2px(side_x, (PITCH_WIDTH - 18.32) / 2),
                      m2px(side_x + 5.5, (PITCH_WIDTH + 18.32) / 2), lc, lw)

    # Penalty spots
    cv2.circle(canvas, m2px(11.0, PITCH_WIDTH / 2), 3, lc, -1)
    cv2.circle(canvas, m2px(PITCH_LENGTH - 11.0, PITCH_WIDTH / 2), 3, lc, -1)

    return canvas


class HeatmapGenerator:
    """
    Accumulates player pitch positions and renders Gaussian heatmaps on a 2D pitch canvas.
    """

    def __init__(self, scale: float = 8.0, sigma: float = 1.5):
        """
        Args:
            scale: Pixels per meter.
            sigma: Gaussian spread in meters.
        """
        self.scale = scale
        self.sigma = sigma
        self.W = int(PITCH_LENGTH * scale)
        self.H = int(PITCH_WIDTH * scale)

        # Accumulation maps: team -> float density map
        self.team_density = defaultdict(lambda: np.zeros((self.H, self.W), dtype=np.float32))
        # Per player density maps
        self.player_density = defaultdict(lambda: np.zeros((self.H, self.W), dtype=np.float32))

    def add_position(self, player_id: int, team: str, pitch_pos):
        """
        Accumulate one Gaussian blob at (pitch_pos) for this player/team.

        Args:
            player_id: Unique tracking ID.
            team: Team name string.
            pitch_pos: (x, y) in meters on the pitch.
        """
        if pitch_pos is None:
            return

        px = int(pitch_pos[0] * self.scale)
        py = int(pitch_pos[1] * self.scale)

        if not (0 <= px < self.W and 0 <= py < self.H):
            return

        # Draw a Gaussian blob into both maps
        sigma_px = int(self.sigma * self.scale)

        # Create a temporary single-point blob
        blob = np.zeros((self.H, self.W), dtype=np.float32)
        blob[py, px] = 1.0

        if sigma_px > 0:
            blob = cv2.GaussianBlur(blob, (0, 0), sigma_px)

        if team not in ("Referee", "Unknown"):
            self.team_density[team] += blob

        self.player_density[player_id] += blob

    def render_team_heatmap(self, team: str, alpha: float = 0.6) -> np.ndarray:
        """
        Render the heatmap for a specific team overlaid on a pitch canvas.

        Args:
            team: Team name.
            alpha: Blend factor for heatmap overlay.

        Returns:
            BGR image with heatmap overlay.
        """
        canvas = generate_pitch_canvas(self.scale)
        density = self.team_density.get(team, np.zeros((self.H, self.W), dtype=np.float32))

        if density.max() > 0:
            norm = (density / density.max() * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            # Blend only non-zero areas
            mask = (norm > 0).astype(np.float32)
            for c in range(3):
                canvas[:, :, c] = np.clip(
                    canvas[:, :, c] * (1 - alpha * mask) + heatmap_color[:, :, c] * alpha * mask,
                    0, 255
                ).astype(np.uint8)

        # Title
        label_color = (255, 255, 255) if team == "Team A" else (80, 80, 255)
        cv2.putText(canvas, f"{team} Heatmap", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2, cv2.LINE_AA)
        return canvas

    def render_combined_heatmap(self) -> np.ndarray:
        """
        Render a side-by-side comparison heatmap for Team A and Team B.

        Returns:
            Combined BGR image.
        """
        hm_a = self.render_team_heatmap("Team A")
        hm_b = self.render_team_heatmap("Team B")
        return np.hstack([hm_a, hm_b])

    def render_player_heatmap(self, player_id: int, team: str = "") -> np.ndarray:
        """
        Render positional heatmap for a specific player.

        Args:
            player_id: Player tracking ID.
            team: Team name (used for label only).

        Returns:
            BGR image with heatmap overlay.
        """
        canvas = generate_pitch_canvas(self.scale)
        density = self.player_density.get(player_id, np.zeros((self.H, self.W), dtype=np.float32))

        if density.max() > 0:
            norm = (density / density.max() * 255).astype(np.uint8)
            heatmap_color = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)
            mask = (norm > 0).astype(np.float32)
            alpha = 0.65
            for c in range(3):
                canvas[:, :, c] = np.clip(
                    canvas[:, :, c] * (1 - alpha * mask) + heatmap_color[:, :, c] * alpha * mask,
                    0, 255
                ).astype(np.uint8)

        cv2.putText(canvas, f"Player {player_id} ({team})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas
