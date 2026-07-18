"""
Homography Estimator and Smoother.

Applies RANSAC algorithms (e.g. cv2.findHomography) to filter out outlier keypoint
estimations and compute the 3x3 projective transformation matrix H mapping pixel space to
physical field coordinates. Smooths H temporally to avoid high-frequency jitter.
"""

import cv2
import numpy as np

class HomographyEstimator:
    """
    RANSAC keypoint matching and temporal smoothing for pitch calibration.
    """
    def __init__(self, reproj_threshold: float = 5.0, smooth_alpha: float = 0.8):
        """
        Args:
            reproj_threshold (float): RANSAC outlier distance threshold.
            smooth_alpha (float): Smoothing weight for exponential moving average of H.
        """
        self.reproj_threshold = reproj_threshold
        self.smooth_alpha = smooth_alpha
        self.last_H = None

    def estimate(self, detected_pts, ground_truth_pts) -> np.ndarray:
        """
        Computes homography matrix H.
        
        Args:
            detected_pts (numpy.ndarray): Detected keypoints in frame (shape: [N, 2]).
            ground_truth_pts (numpy.ndarray): Actual 2D pitch coordinates in meters (shape: [N, 2]).
            
        Returns:
            numpy.ndarray: Projective matrix H (3x3), or None if estimation failed.
        """
        src_pts = np.array(detected_pts, dtype=np.float32)
        dst_pts = np.array(ground_truth_pts, dtype=np.float32)
        
        if len(src_pts) < 4 or len(dst_pts) < 4:
            return self.last_H
            
        try:
            H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.reproj_threshold)
            if H is not None:
                if H[2, 2] != 0.0:
                    H = H / H[2, 2]
                return H
        except Exception:
            pass
            
        return self.last_H

    def smooth(self, current_H: np.ndarray) -> np.ndarray:
        """
        Smooths H over time to avoid visual jitter.
        
        Args:
            current_H (numpy.ndarray): Current calculated 3x3 homography matrix.
            
        Returns:
            numpy.ndarray: Smoothed 3x3 homography matrix.
        """
        if current_H is None:
            return self.last_H
            
        if current_H[2, 2] != 0.0:
            current_H = current_H / current_H[2, 2]
            
        if self.last_H is None:
            self.last_H = current_H.copy()
            return self.last_H
            
        smoothed_H = self.smooth_alpha * current_H + (1.0 - self.smooth_alpha) * self.last_H
        
        if smoothed_H[2, 2] != 0.0:
            smoothed_H = smoothed_H / smoothed_H[2, 2]
            
        self.last_H = smoothed_H.copy()
        return self.last_H

