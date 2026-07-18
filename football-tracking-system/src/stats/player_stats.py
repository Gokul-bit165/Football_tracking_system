"""
Player Statistics Tracker.
Tracks per-player: distance covered, top speed, average speed, possession time, passes made/received.
"""
import numpy as np
from collections import defaultdict


class PlayerStatsTracker:
    """
    Accumulates per-player statistics across an entire video.
    """

    def __init__(self, fps: float = 25.0):
        self.fps = fps
        self.dt = 1.0 / fps

        # Per-player data
        self.trajectories = defaultdict(list)      # player_id -> list of (x, y) in meters
        self.distances = defaultdict(float)         # player_id -> total distance (m)
        self.speeds = defaultdict(list)             # player_id -> list of instantaneous speeds (m/s)
        self.possession_frames = defaultdict(int)   # player_id -> frames in possession
        self.passes_made = defaultdict(int)         # player_id -> count
        self.passes_received = defaultdict(int)     # player_id -> count
        self.interceptions = defaultdict(int)       # player_id -> count
        self.teams = {}                             # player_id -> team name

    def update_position(self, player_id: int, pitch_pos, team: str):
        """
        Register a player's pitch position for this frame.

        Args:
            player_id: Unique tracking ID.
            pitch_pos: (x, y) in meters on the pitch, or None if not visible.
            team: Team label string.
        """
        self.teams[player_id] = team
        prev_positions = self.trajectories[player_id]

        if pitch_pos is not None:
            if len(prev_positions) > 0 and prev_positions[-1] is not None:
                dx = pitch_pos[0] - prev_positions[-1][0]
                dy = pitch_pos[1] - prev_positions[-1][1]
                dist = float(np.sqrt(dx**2 + dy**2))
                # Clamp to avoid GPS-like jumps (>12 m/s * dt is unrealistic for a frame)
                if dist / self.dt < 12.0:
                    self.distances[player_id] += dist
                    self.speeds[player_id].append(dist / self.dt)

        self.trajectories[player_id].append(pitch_pos)

    def update_possession(self, player_id: int):
        """Mark that player_id has the ball this frame."""
        if player_id is not None:
            self.possession_frames[player_id] += 1

    def update_pass_event(self, event: dict):
        """
        Register a pass/interception event.

        Args:
            event: Dict with keys: from, to, team_from, team_to, status
        """
        from_id = event.get("from")
        to_id = event.get("to")
        status = event.get("status", "success")

        if from_id is not None:
            self.passes_made[from_id] += 1
        if to_id is not None:
            if status == "intercepted":
                self.interceptions[to_id] += 1
            else:
                self.passes_received[to_id] += 1

    def get_summary(self) -> dict:
        """
        Return a dict summarizing all players' stats.

        Returns:
            dict: player_id -> stats dict
        """
        summary = {}
        for pid in self.trajectories:
            speeds = self.speeds.get(pid, [0.0])
            top_speed = float(max(speeds)) if speeds else 0.0
            avg_speed = float(np.mean(speeds)) if speeds else 0.0
            possession_s = self.possession_frames[pid] / self.fps

            summary[pid] = {
                "team": self.teams.get(pid, "Unknown"),
                "distance_m": round(self.distances[pid], 2),
                "top_speed_ms": round(top_speed, 2),
                "avg_speed_ms": round(avg_speed, 2),
                "possession_s": round(possession_s, 2),
                "passes_made": self.passes_made[pid],
                "passes_received": self.passes_received[pid],
                "interceptions": self.interceptions[pid],
            }
        return summary

    def get_team_summary(self) -> dict:
        """
        Aggregate stats per team.

        Returns:
            dict: team_name -> aggregated stats
        """
        team_stats = defaultdict(lambda: {
            "players": 0,
            "total_distance_m": 0.0,
            "possession_s": 0.0,
            "passes_made": 0,
            "passes_received": 0,
            "interceptions": 0,
        })

        for pid, stats in self.get_summary().items():
            team = stats["team"]
            if team in ("Referee", "Unknown"):
                continue
            ts = team_stats[team]
            ts["players"] += 1
            ts["total_distance_m"] += stats["distance_m"]
            ts["possession_s"] += stats["possession_s"]
            ts["passes_made"] += stats["passes_made"]
            ts["passes_received"] += stats["passes_received"]
            ts["interceptions"] += stats["interceptions"]

        # Round values
        for team in team_stats:
            team_stats[team]["total_distance_m"] = round(team_stats[team]["total_distance_m"], 2)
            team_stats[team]["possession_s"] = round(team_stats[team]["possession_s"], 2)

        return dict(team_stats)
