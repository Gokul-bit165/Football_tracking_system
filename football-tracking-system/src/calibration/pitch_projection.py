"""
Pitch Projection and Spatial Analytics.

Applies the homography matrix H specifically to the foot-points of bounding boxes
(the contact point on the ground plane, z=0) to project players and ball into top-down
2D pitch coordinates (105m x 68m). Computes metrics like heatmaps, distance covered, and speed.
"""

class PitchProjector:
    """
    Projector from image pixel coords to bird's-eye view coordinates.
    """
    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    def project_point(self, pt, H):
        """
        Project pixel coordinates (foot-point) to pitch ground-plane coordinates.
        
        Args:
            pt (tuple): (x, y) pixels.
            H (numpy.ndarray): 3x3 homography matrix.
            
        Returns:
            tuple: (x_meters, y_meters) coordinates.
        """
        pass

    def compute_distance_covered(self, trajectory):
        """
        Computes distance covered in meters by summing deltas.
        
        Args:
            trajectory (list): List of (x, y) coordinates in meters.
            
        Returns:
            float: Total distance.
        """
        pass

    def estimate_speed(self, trajectory, fps):
        """
        Computes speed (m/s) using temporal differences, applying smoothing.
        """
        pass
