import numpy as np

class PitchProjector:
    """
    Projector from image pixel coords to bird's-eye view coordinates.
    """
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        """
        Args:
            pitch_length (float): Standard length of the football pitch in meters (default: 105.0).
            pitch_width (float): Standard width of the football pitch in meters (default: 68.0).
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def project_point(self, pt: tuple, H: np.ndarray) -> tuple:
        """
        Project pixel coordinates (foot-point) to pitch ground-plane coordinates.
        
        Args:
            pt (tuple): (x, y) pixels.
            H (numpy.ndarray): 3x3 homography matrix.
            
        Returns:
            tuple: (x_meters, y_meters) coordinates.
        """
        if pt is None or H is None:
            return None
            
        # Convert to homogeneous coordinate [u, v, 1]^T
        p_pixel = np.array([pt[0], pt[1], 1.0], dtype=np.float32)
        
        # Multiply by homography matrix
        p_meters = np.dot(H, p_pixel)
        
        # Normalize by homogeneous scale coordinate
        if p_meters[2] != 0.0:
            mx = float(p_meters[0] / p_meters[2])
            my = float(p_meters[1] / p_meters[2])
            
            # Clip projected coordinate to stay roughly near field boundaries
            mx = np.clip(mx, -5.0, self.pitch_length + 5.0)
            my = np.clip(my, -5.0, self.pitch_width + 5.0)
            return (mx, my)
            
        return (0.0, 0.0)

    def compute_distance_covered(self, trajectory: list) -> float:
        """
        Computes distance covered in meters by summing deltas.
        
        Args:
            trajectory (list): List of (x, y) coordinates in meters.
            
        Returns:
            float: Total distance in meters.
        """
        total_dist = 0.0
        # Filter out None coordinates
        valid_pts = [pt for pt in trajectory if pt is not None]
        
        for i in range(1, len(valid_pts)):
            p1 = valid_pts[i-1]
            p2 = valid_pts[i]
            total_dist += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
        return float(total_dist)

    def estimate_speed(self, trajectory: list, fps: float) -> list:
        """
        Computes speed (m/s) using temporal differences, applying SMA smoothing.
        
        Args:
            trajectory (list): List of (x, y) coordinates in meters.
            fps (float): Frame rate of the video.
            
        Returns:
            list: Smoothed speeds (m/s) for each step (length matches trajectory).
        """
        if not trajectory or len(trajectory) < 2:
            return [0.0] * len(trajectory)
            
        dt = 1.0 / fps
        raw_speeds = [0.0]
        
        # Calculate raw speed at each step
        for i in range(1, len(trajectory)):
            p1 = trajectory[i-1]
            p2 = trajectory[i]
            
            if p1 is None or p2 is None:
                raw_speeds.append(0.0)
            else:
                dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                raw_speeds.append(dist / dt)
                
        # Smooth speeds using Simple Moving Average (window size = 5)
        smoothed_speeds = []
        window_size = 5
        for i in range(len(raw_speeds)):
            start_idx = max(0, i - window_size + 1)
            sub_window = raw_speeds[start_idx : i + 1]
            smoothed_speeds.append(float(np.mean(sub_window)))
            
        return smoothed_speeds

