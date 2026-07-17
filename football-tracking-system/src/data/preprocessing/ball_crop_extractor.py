"""
Ball crop extraction module.
Extracts individual crops of the football class from the cleaned train split
and saves them to an interim folder along with sidecar JSON files for copy-paste augmentation.
"""

import os
import cv2
import json
import shutil
from loguru import logger
from typing import List, Dict, Any

class BallCropExtractor:
    """
    Extracts crops of 'ball' class instances from cleaned train frames.
    """
    def __init__(self, interim_dir: str, ball_class_id: int = 0):
        """
        Args:
            interim_dir (str): Base interim directory.
            ball_class_id (int): Canonical ID for the ball class (default: 0).
        """
        self.crop_dir = os.path.join(interim_dir, "ball_crops")
        self.ball_class_id = ball_class_id
        
        # Track statistics
        self.extracted_count = 0

    def prepare_directory(self):
        """
        Ensures the crop directory exists and is empty.
        """
        if os.path.exists(self.crop_dir):
            logger.info(f"Clearing existing ball crops directory: {self.crop_dir}")
            shutil.rmtree(self.crop_dir)
        os.makedirs(self.crop_dir, exist_ok=True)
        logger.info(f"Prepared directory for ball crops: {self.crop_dir}")

    def extract_crops(
        self,
        image_records: List[Dict[str, Any]],
        img_dir: str,
        annotations_by_image: Dict[int, List[Dict[str, Any]]],
        category_id_mapping: Dict[int, int]
    ) -> int:
        """
        Extracts cropped ball bounding boxes from images and saves them.
        
        Args:
            image_records: Cleaned train split image records.
            img_dir: Path to raw/cleaned image directory.
            annotations_by_image: Map of image_id -> list of COCO annotations.
            category_id_mapping: Map from raw COCO category ID -> canonical ID.
            
        Returns:
            int: Number of crops successfully extracted.
        """
        self.prepare_directory()
        logger.info(f"Extracting ball crops from {len(image_records)} cleaned train images...")
        
        self.extracted_count = 0
        
        for rec in image_records:
            image_id = rec["id"]
            file_name = rec["file_name"]
            img_path = os.path.join(img_dir, file_name)
            
            # Get annotations for this image
            annotations = annotations_by_image.get(image_id, [])
            ball_annotations = []
            
            for ann in annotations:
                raw_cat_id = ann.get("category_id")
                canonical_id = category_id_mapping.get(raw_cat_id)
                
                if canonical_id == self.ball_class_id:
                    ball_annotations.append(ann)
                    
            if not ball_annotations:
                continue
                
            # Load the image
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Could not load image {img_path} for crop extraction.")
                continue
                
            img_height, img_width = img.shape[:2]
            
            for ann in ball_annotations:
                bbox = ann.get("bbox")  # [x_min, y_min, width, height]
                if not bbox or len(bbox) != 4:
                    continue
                    
                x_min, y_min, w, h = bbox
                
                # Convert to integer pixel coordinates
                x1 = int(round(x_min))
                y1 = int(round(y_min))
                x2 = int(round(x_min + w))
                y2 = int(round(y_min + h))
                
                # Ensure coordinates are within image boundaries
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(x1 + 1, min(x2, img_width))
                y2 = max(y1 + 1, min(y2, img_height))
                
                # Crop
                crop = img[y1:y2, x1:x2]
                
                if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                    continue
                    
                # Save crop image
                crop_name = f"ball_crop_{self.extracted_count:05d}.jpg"
                crop_path = os.path.join(self.crop_dir, crop_name)
                cv2.imwrite(crop_path, crop)
                
                # Save sidecar metadata JSON file for provenance
                meta_name = f"ball_crop_{self.extracted_count:05d}.json"
                meta_path = os.path.join(self.crop_dir, meta_name)
                
                metadata = {
                    "source_image": file_name,
                    "original_bbox": [float(x_min), float(y_min), float(w), float(h)]
                }
                
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                    
                self.extracted_count += 1
                
        logger.info(f"Successfully extracted {self.extracted_count} ball crops with sidecar JSON files to {self.crop_dir}")
        return self.extracted_count
