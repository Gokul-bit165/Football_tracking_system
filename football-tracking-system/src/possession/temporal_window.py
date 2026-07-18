import numpy as np
from collections import deque

class TemporalWindowPossession:
    """
    Temporal window-based ball possession estimator.
    Tracks player closest to the ball and filters out rapid toggling.
    Detects pass and interception events using a state transition machine.
    """
    def __init__(self, proximity_threshold: float = 1.5, window_frames: int = 10):
        """
        Args:
            proximity_threshold (float): Max distance (in meters or pixels) to consider a player in possession.
            window_frames (int): Number of consecutive frames needed to assign possession.
        """
        self.proximity_threshold = proximity_threshold
        self.window_frames = window_frames
        
        # Sliding window buffer for closest player IDs
        self.window = deque(maxlen=window_frames)
        
        # Possession tracking states
        self.current_possession = None
        self.pass_candidate = None  # Tuple: (player_id, team_name, start_frame)
        self.frame_count = 0
        
        # Global list of events detected
        self.events = []

    def get_closest_player(self, player_positions: dict, ball_position: tuple) -> tuple:
        """
        Calculate the closest player to the ball.
        
        Args:
            player_positions (dict): Map of player_id -> (x, y)
            ball_position (tuple): (x, y) coordinates of the ball
            
        Returns:
            tuple: (closest_player_id, min_distance)
        """
        if ball_position is None or not player_positions:
            return None, float("inf")
            
        bx, by = ball_position
        min_dist = float("inf")
        closest_id = None
        
        for pid, pos in player_positions.items():
            px, py = pos
            dist = np.sqrt((px - bx)**2 + (py - by)**2)
            if dist < min_dist:
                min_dist = dist
                closest_id = pid
                
        return closest_id, min_dist

    def update(self, player_positions: dict, ball_position: tuple, player_teams: dict) -> tuple:
        """
        Updates possession state with current positions and detects events.
        
        Args:
            player_positions (dict): Map of player_id -> (x, y)
            ball_position (tuple): (x, y) for the ball
            player_teams (dict): Map of player_id -> team_name (e.g. 'Team A', 'Team B', 'Referee')
            
        Returns:
            tuple: (current_possession_player_id, list_of_new_events_in_this_frame)
        """
        new_events = []
        closest_id, min_dist = self.get_closest_player(player_positions, ball_position)
        
        # 1. Add current frame candidate to the sliding window
        if closest_id is not None and min_dist <= self.proximity_threshold:
            self.window.append(closest_id)
        else:
            self.window.append(None)
            
        # 2. Determine smoothed possession using majority voting
        if len(self.window) < self.window_frames:
            # Buffer not full yet, return None
            self.frame_count += 1
            return None, new_events
            
        # Count votes
        votes = {}
        for pid in self.window:
            votes[pid] = votes.get(pid, 0) + 1
            
        # Find candidate with highest votes
        best_candidate = max(votes, key=votes.get)
        vote_count = votes[best_candidate]
        
        # Majority threshold (at least 50% of the window)
        majority_thresh = self.window_frames // 2
        
        smoothed_possession = self.current_possession
        if vote_count >= majority_thresh:
            smoothed_possession = best_candidate
            
        # 3. Possession transition state machine
        if smoothed_possession != self.current_possession:
            # Case A: Possession lost (ball kicked/released)
            if self.current_possession is not None and smoothed_possession is None:
                pid = self.current_possession
                team = player_teams.get(pid, "Unknown")
                # Exclude Referees from possession events
                if team != "Referee":
                    self.pass_candidate = (pid, team, self.frame_count)
                    
            # Case B: Possession established (ball received)
            elif smoothed_possession is not None:
                new_owner = smoothed_possession
                team_to = player_teams.get(new_owner, "Unknown")
                
                if team_to != "Referee":
                    # Check if we were tracking an active pass candidate
                    if self.pass_candidate is not None:
                        from_player, team_from, frame_start = self.pass_candidate
                        
                        if from_player != new_owner:
                            # A pass event completed!
                            status = "success" if team_from == team_to else "intercepted"
                            event = {
                                "type": "pass",
                                "from": from_player,
                                "to": new_owner,
                                "team_from": team_from,
                                "team_to": team_to,
                                "status": status,
                                "frame_start": frame_start,
                                "frame_end": self.frame_count
                            }
                            self.events.append(event)
                            new_events.append(event)
                            
                        self.pass_candidate = None
                        
                    # Handle direct transfer (e.g. tackle, handoff without intermediate None frame)
                    elif self.current_possession is not None:
                        from_player = self.current_possession
                        team_from = player_teams.get(from_player, "Unknown")
                        
                        if from_player != new_owner and team_from != "Referee":
                            status = "success" if team_from == team_to else "intercepted"
                            event = {
                                "type": "pass",
                                "from": from_player,
                                "to": new_owner,
                                "team_from": team_from,
                                "team_to": team_to,
                                "status": status,
                                "frame_start": self.frame_count - 1,
                                "frame_end": self.frame_count
                            }
                            self.events.append(event)
                            new_events.append(event)
                            
            self.current_possession = smoothed_possession
            
        self.frame_count += 1
        return self.current_possession, new_events

