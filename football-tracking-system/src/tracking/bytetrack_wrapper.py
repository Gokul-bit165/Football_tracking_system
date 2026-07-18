import os
import yaml
import numpy as np
import ultralytics
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.trackers.byte_tracker import BYTETracker

class DetectionsAdapter:
    """
    Adapts custom or raw numpy detection matrices into the object format expected 
    by Ultralytics' internal tracker implementations.
    """
    def __init__(self, data: np.ndarray):
        # data format: [N, 6] where columns are [x1, y1, x2, y2, conf, cls_id]
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = np.expand_dims(data, axis=0)
        self.data = data

    @property
    def xyxy(self):
        return self.data[:, :4] if len(self.data) > 0 else np.empty((0, 4), dtype=np.float32)

    @property
    def xywh(self):
        if len(self.data) == 0:
            return np.empty((0, 4), dtype=np.float32)
        x1, y1, x2, y2 = self.data[:, 0], self.data[:, 1], self.data[:, 2], self.data[:, 3]
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        return np.stack([xc, yc, w, h], axis=-1)

    @property
    def conf(self):
        return self.data[:, 4] if len(self.data) > 0 else np.empty((0,), dtype=np.float32)

    @property
    def cls(self):
        return self.data[:, 5] if len(self.data) > 0 else np.empty((0,), dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return DetectionsAdapter(self.data[index])

class ByteTrackWrapper:
    """
    ByteTrack tracker wrapper class.
    """
    def __init__(self, config_path: str = None, frame_rate: int = 30):
        """
        Initialize ByteTrack with parameters.
        
        Args:
            config_path (str): Optional path to YAML containing tracking parameters.
            frame_rate (int): Video frame rate.
        """
        # Load default bytetrack settings from the installed ultralytics package
        ultralytics_path = os.path.dirname(ultralytics.__file__)
        default_config_path = os.path.join(ultralytics_path, "cfg", "trackers", "bytetrack.yaml")
        
        if config_path and os.path.exists(config_path):
            active_config = config_path
        else:
            active_config = default_config_path
            
        with open(active_config, "r") as f:
            tracker_args = yaml.safe_load(f)
            
        # Parse arguments into IterableSimpleNamespace as required by BYTETracker
        self.args = IterableSimpleNamespace(**tracker_args)
        self.frame_rate = frame_rate
        self.tracker = BYTETracker(self.args)

    def update(self, detections, frame=None):
        """
        Update tracker with new frame detections.
        
        Args:
            detections: Either an ultralytics.engine.results.Boxes instance, 
                        or a numpy array / list of shape [N, 6] (each row [x1, y1, x2, y2, conf, cls]).
            frame (np.ndarray): Raw image frame (optional for ByteTrack).
            
        Returns:
            np.ndarray: Tracked boxes of shape [M, 7] or [M, 8], where each row contains 
                        [x1, y1, x2, y2, track_id, conf, cls, original_index].
        """
        # If it's a list or numpy array, wrap it in DetectionsAdapter
        if not hasattr(detections, "conf"):
            detections = DetectionsAdapter(np.array(detections, dtype=np.float32))
            
        if len(detections) == 0:
            return np.empty((0, 7), dtype=np.float32)
            
        # ByteTracker.update returns np.ndarray of shape [M, 7] or [M, 8]
        # x.result format is [x1, y1, x2, y2, track_id, score, cls, idx]
        tracks = self.tracker.update(detections, frame)
        return tracks

