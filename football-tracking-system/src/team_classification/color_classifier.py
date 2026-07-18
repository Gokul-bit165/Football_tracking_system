import cv2
import numpy as np

class ColorTeamClassifier:
    """
    K-Means and HSV based player jersey color classifier.
    Extremely fast and lightweight.
    """
    def __init__(self, team_colors: dict = None):
        """
        Initialize the color classifier with reference team colors.
        
        Args:
            team_colors (dict): Dictionary mapping team labels to their reference HSV values.
                                Example:
                                {
                                    "Team A": [10, 200, 200],   # Red (approx)
                                    "Team B": [120, 200, 200],  # Blue (approx)
                                    "Referee": [30, 200, 200]   # Yellow/Green (approx)
                                }
        """
        # Set default colors if none provided (e.g. Red vs Blue vs Yellow Referee)
        if team_colors is None:
            self.team_colors = {
                "Team A": np.array([0, 255, 255], dtype=np.float32),    # Red
                "Team B": np.array([120, 255, 255], dtype=np.float32),  # Blue
                "Referee": np.array([30, 255, 255], dtype=np.float32)   # Yellow
            }
        else:
            self.team_colors = {k: np.array(v, dtype=np.float32) for k, v in team_colors.items()}

    def get_dominant_hsv(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract the dominant HSV color from the upper 45% (jersey region) of a player crop,
        filtering out background green grass.
        
        Args:
            crop (np.ndarray): BGR image patch of the player.
            
        Returns:
            np.ndarray: Dominant HSV color vector [H, S, V].
        """
        if crop is None or crop.size == 0:
            return np.array([0, 0, 0], dtype=np.float32)
            
        h, w, _ = crop.shape
        # 1. Slice upper 45% (jersey region)
        jersey_crop = crop[:max(1, int(h * 0.45)), :]
        
        # 2. Convert to HSV
        hsv_jersey = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)
        
        # 3. Mask out background grass (green HSV range: [35, 40, 40] to [85, 255, 255])
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv_jersey, lower_green, upper_green)
        non_green_mask = cv2.bitwise_not(green_mask)
        
        # Extract non-green pixels
        pixels = hsv_jersey[non_green_mask > 0]
        
        # Fall back to all jersey pixels if grass filtering removed almost everything
        if len(pixels) < 15:
            pixels = hsv_jersey.reshape(-1, 3)
            
        # Convert to float32 for cv2.kmeans
        pixels = np.float32(pixels)
        
        # 4. Perform K-Means clustering (K=2 is best to separate jersey from shadows/numbers)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, flags)
        
        # Find the center that has the most pixel assignments
        counts = np.bincount(labels.flatten())
        dominant_center = centers[np.argmax(counts)]
        
        return dominant_center

    @staticmethod
    def hsv_distance(hsv1: np.ndarray, hsv2: np.ndarray) -> float:
        """
        Compute Euclidean distance in HSV space, accounting for Hue's circular wrap-around at 180.
        """
        # OpenCV Hue ranges from 0 to 180
        dh = min(abs(hsv1[0] - hsv2[0]), 180 - abs(hsv1[0] - hsv2[0])) / 180.0
        ds = (hsv1[1] - hsv2[1]) / 255.0
        dv = (hsv1[2] - hsv2[2]) / 255.0
        
        # Apply weights: Hue is highly critical, Saturation/Value are weighted less due to shadows/shines
        return float(np.sqrt(2.5 * (dh ** 2) + 0.5 * (ds ** 2) + 0.5 * (dv ** 2)))

    def classify(self, crop: np.ndarray) -> str:
        """
        Classify the player crop into Team A, Team B, or Referee.
        
        Args:
            crop (np.ndarray): BGR player crop.
            
        Returns:
            str: Assigned team name label.
        """
        dominant_hsv = self.get_dominant_hsv(crop)
        
        # Find closest reference color using HSV distance
        min_dist = float("inf")
        assigned_team = "Unknown"
        
        for team, ref_color in self.team_colors.items():
            dist = self.hsv_distance(dominant_hsv, ref_color)
            if dist < min_dist:
                min_dist = dist
                assigned_team = team
                
        return assigned_team
