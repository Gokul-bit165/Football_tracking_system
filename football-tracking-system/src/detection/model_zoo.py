"""
Model Zoo Wrapper for Football Tracking System.

Provides a unified interface to load, initialize, and run inference using 
various benchmark models including:
- YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO12
- RT-DETR
- DINO / Grounding DINO (for offline pseudo-labeling)
"""

class ModelZoo:
    """
    Registry and loader for all object detection architectures.
    """
    def __init__(self, model_type: str, checkpoint_path: str = None):
        """
        Initialize the model architecture.
        
        Args:
            model_type (str): Key of the model to load (e.g., 'yolo11n', 'rt-detr')
            checkpoint_path (str): Optional path to pre-trained weights
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path

    def load_model(self):
        """
        Loads model architecture and weights.
        """
        pass

    def predict(self, frame):
        """
        Runs object detection inference on a single frame.
        
        Args:
            frame: Input image/frame.
            
        Returns:
            list: Bounding boxes with format [x1, y1, x2, y2, confidence, class_id]
        """
        pass
