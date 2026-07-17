"""
Dataset exporter module.
Converts clean, validated, and augmented images and COCO annotations to standard YOLO format.
Creates physical image files and matching normalized label .txt files, and writes the data.yaml configuration.
"""

import os
import cv2
import yaml
from loguru import logger
from typing import List, Dict, Any, Tuple

class YOLOExporter:
    """
    Exports a processed dataset split to standard YOLO v8/11 format.
    """
    def __init__(self, output_dir: str, class_names: List[str]):
        """
        Args:
            output_dir (str): Path to data/processed/<version_tag>/cleaned_dataset/
            class_names (List[str]): List of canonical class names indexed by canonical IDs.
        """
        self.output_dir = output_dir
        self.class_names = class_names
        
        # Track statistics
        self.split_counts: Dict[str, int] = {}
        self.split_box_counts: Dict[str, int] = {}

    def prepare_directories(self, splits: List[str]):
        """
        Creates YOLO images and labels directories for each split.
        """
        for split in splits:
            os.makedirs(os.path.join(self.output_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "labels", split), exist_ok=True)
        logger.info(f"YOLO directory structure initialized under: {self.output_dir}")

    def export_image_and_labels(
        self,
        img: Any,
        bboxes_coco: List[List[float]],
        class_labels: List[int],
        original_filename: str,
        split_name: str,
        replica_id: int = 0
    ) -> Tuple[str, int]:
        """
        Saves a physical image and its normalized label .txt file in YOLO format.
        
        Args:
            img: OpenCV image (BGR).
            bboxes_coco: List of COCO bounding boxes [[x_min, y_min, w, h]].
            class_labels: List of canonical class IDs.
            original_filename: Raw image filename (e.g. 'frame_0.jpg').
            split_name: Split directory ('train', 'val', or 'test').
            replica_id: Suffix ID if this is an oversampled duplicate.
            
        Returns:
            Tuple[str, int]: (saved_image_path, num_boxes_exported)
        """
        img_height, img_width = img.shape[:2]
        
        # Build unique filename for replicas
        base_name, ext = os.path.splitext(original_filename)
        if replica_id > 0:
            target_filename = f"{base_name}_rep{replica_id}{ext}"
        else:
            target_filename = original_filename
            
        img_dest_path = os.path.join(self.output_dir, "images", split_name, target_filename)
        label_dest_path = os.path.join(
            self.output_dir, "labels", split_name, f"{os.path.splitext(target_filename)[0]}.txt"
        )
        
        # Write the physical image file
        cv2.imwrite(img_dest_path, img)
        
        # Write the YOLO label file
        lines = []
        box_count = 0
        for bbox, cid in zip(bboxes_coco, class_labels):
            x_min, y_min, w, h = bbox
            
            # Compute YOLO normalized coordinates
            x_center = (x_min + w / 2.0) / img_width
            y_center = (y_min + h / 2.0) / img_height
            norm_w = w / img_width
            norm_h = h / img_height
            
            # Enforce limits [0.0, 1.0]
            x_center = max(0.0, min(x_center, 1.0))
            y_center = max(0.0, min(y_center, 1.0))
            norm_w = max(0.0001, min(norm_w, 1.0))
            norm_h = max(0.0001, min(norm_h, 1.0))
            
            lines.append(f"{int(round(cid))} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            box_count += 1
            
        with open(label_dest_path, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
            
        return img_dest_path, box_count

    def generate_data_yaml(self):
        """
        Generates the required data.yaml file for training with Ultralytics YOLO.
        """
        yaml_path = os.path.join(self.output_dir, "data.yaml")
        
        data_dict = {
            "path": os.path.abspath(self.output_dir),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: name for i, name in enumerate(self.class_names)}
        }
        
        with open(yaml_path, "w") as f:
            yaml.dump(data_dict, f, default_flow_style=False)
            
        logger.info(f"YOLO dataset config file written successfully to: {yaml_path}")
        
    def log_split_statistics(self, split_name: str, img_count: int, box_count: int):
        """
        Registers split statistics.
        """
        self.split_counts[split_name] = img_count
        self.split_box_counts[split_name] = box_count
        logger.info(f"Split '{split_name}' finalized: {img_count} images, {box_count} label boxes.")
