"""
Mini-map / Bird's-Eye View Renderer.
Draws a real-time 2D top-down pitch with player dots and ball position.
"""
import cv2
import numpy as np
from src.viz.heatmap import generate_pitch_canvas, PITCH_LENGTH, PITCH_WIDTH


# Team display colors (BGR)
TEAM_COLORS = {
    "Team A": (255, 255, 255),    # White
    "Team B": (50, 50, 255),      # Red
    "Goalkeeper": (0, 255, 255),  # Cyan
    "Referee": (0, 200, 0),       # Green
    "Unknown": (150, 150, 150),
}


class MinimapRenderer:
    """
    Renders a real-time 2D top-down pitch minimap with player dots,
    ball position, and possession indicator.
    """

    def __init__(self, scale: float = 5.0):
        """
        Args:
            scale: Pixels per meter.
        """
        self.scale = scale
        self.W = int(PITCH_LENGTH * scale)
        self.H = int(PITCH_WIDTH * scale)

    def m2px(self, mx: float, my: float):
        return (int(np.clip(mx * self.scale, 0, self.W - 1)),
                int(np.clip(my * self.scale, 0, self.H - 1)))

    def render(
        self,
        player_positions: dict,
        player_teams: dict,
        ball_pos_m=None,
        possession_id: int = None,
        frame_num: int = 0
    ) -> np.ndarray:
        """
        Render the minimap for a single frame.

        Args:
            player_positions: {player_id: (x_m, y_m)} in meters.
            player_teams: {player_id: team_name}.
            ball_pos_m: (x_m, y_m) of the ball in meters, or None.
            possession_id: ID of player currently in possession.
            frame_num: Current frame number (for display).

        Returns:
            BGR minimap image.
        """
        canvas = generate_pitch_canvas(self.scale)

        # Draw players
        for pid, pos in player_positions.items():
            if pos is None:
                continue
            team = player_teams.get(pid, "Unknown")
            color = TEAM_COLORS.get(team, TEAM_COLORS["Unknown"])
            px, py = self.m2px(pos[0], pos[1])

            # Highlight player with possession
            if pid == possession_id:
                cv2.circle(canvas, (px, py), 9, (0, 215, 255), -1)  # Gold ring
            cv2.circle(canvas, (px, py), 5, color, -1)

            # Player ID label
            cv2.putText(canvas, str(pid), (px + 5, py - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw ball
        if ball_pos_m is not None:
            bx, by = self.m2px(ball_pos_m[0], ball_pos_m[1])
            cv2.circle(canvas, (bx, by), 5, (0, 120, 255), -1)
            cv2.circle(canvas, (bx, by), 7, (0, 180, 255), 1)

        # Frame counter
        cv2.putText(canvas, f"Frame {frame_num}", (5, self.H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        return canvas
