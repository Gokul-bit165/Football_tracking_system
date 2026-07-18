import os
import yaml
import numpy as np
import ultralytics
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.trackers.bot_sort import BOTSORT
from src.tracking.bytetrack_wrapper import DetectionsAdapter

class BoTSORTWrapper:
    """
    BoT-SORT tracker wrapper class.
    """
    def __init__(self, config_path: str = None, frame_rate: int = 30):
        """
        Initialize BoT-SORT parameters.
        
        Args:
            config_path (str): Optional path to YAML tracker parameters.
            frame_rate (int): Video frame rate.
        """
        # Load default botsort settings from the installed ultralytics package
        ultralytics_path = os.path.dirname(ultralytics.__file__)
        default_config_path = os.path.join(ultralytics_path, "cfg", "trackers", "botsort.yaml")
        
        if config_path and os.path.exists(config_path):
            active_config = config_path
        else:
            active_config = default_config_path
            
        with open(active_config, "r") as f:
            tracker_args = yaml.safe_load(f)
            
        # Parse arguments into IterableSimpleNamespace as required by BOTSORT
        self.args = IterableSimpleNamespace(**tracker_args)
        self.frame_rate = frame_rate
        self.tracker = BOTSORT(self.args)

    def update(self, detections, frame):
        """
        Update tracker with new frame detections and estimate camera motion.
        
        Args:
            detections: Either an ultralytics.engine.results.Boxes instance, 
                        or a numpy array / list of shape [N, 6] (each row [x1, y1, x2, y2, conf, cls]).
            frame (np.ndarray): Raw image frame (required for GMC).
            
        Returns:
            np.ndarray: Tracked boxes of shape [M, 7] or [M, 8], where each row contains 
                        [x1, y1, x2, y2, track_id, conf, cls, original_index].
        """
        # If it's a list or numpy array, wrap it in DetectionsAdapter
        if not hasattr(detections, "conf"):
            detections = DetectionsAdapter(np.array(detections, dtype=np.float32))
            
        if len(detections) == 0:
            return np.empty((0, 7), dtype=np.float32)
            
        # BOTSORT.update returns np.ndarray of shape [M, 7] or [M, 8]
        # x.result format is [x1, y1, x2, y2, track_id, score, cls, idx]
        tracks = self.tracker.update(detections, frame)
        return tracks

