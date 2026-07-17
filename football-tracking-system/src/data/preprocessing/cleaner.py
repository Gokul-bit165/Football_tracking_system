"""
Cleaning module for removing corrupted, duplicate, blurry, and static/replay transition frames.
Handles within-split cleanups and writes safe copies of cleaned data.
"""

import os
import cv2
import re
import numpy as np
import imagehash
from PIL import Image
from loguru import logger
from typing import Dict, List, Any, Set, Tuple

class DatasetCleaner:
    """
    Cleans a split of a football player/ball detection dataset.
    """
    def __init__(
        self,
        blur_threshold: float = 100.0,
        duplicate_threshold: int = 5,
        logo_pixel_diff_threshold: float = 1.5,
        remove_blurry: bool = True,
        remove_duplicates: bool = True,
        remove_empty: bool = True,
        remove_logo_replay: bool = True
    ):
        self.blur_threshold = blur_threshold
        self.duplicate_threshold = duplicate_threshold
        self.logo_pixel_diff_threshold = logo_pixel_diff_threshold
        
        self.remove_blurry = remove_blurry
        self.remove_duplicates = remove_duplicates
        self.remove_empty = remove_empty
        self.remove_logo_replay = remove_logo_replay
        
        # Track statistics
        self.stats = {
            "corrupted": 0,
            "blurry": 0,
            "duplicate": 0,
            "empty": 0,
            "logo_replay": 0,
            "original_count": 0,
            "cleaned_count": 0
        }

    def parse_filename(self, filename: str) -> Tuple[str, int]:
        """
        Parses sequence information from Roboflow filenames.
        Example: '08fd33_0_1_png.rf.15fee0064ecbc442fabe68c6406dc72e.jpg'
        prefix: '08fd33_0', frame_idx: 1
        """
        # Strip extension
        base = os.path.splitext(filename)[0]
        # Match pattern like: text_number_number_png
        match = re.match(r"^([a-zA-Z0-9]+_\d+)_(\d+)_png", base)
        if match:
            prefix = match.group(1)
            frame_idx = int(match.group(2))
            return prefix, frame_idx
        # Fallback if pattern doesn't match
        return "unknown_clip", 0

    def compute_blur_score(self, img_gray: np.ndarray) -> float:
        """
        Computes Laplacian variance as a proxy for image sharpness.
        """
        return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())

    def compute_hash(self, img_pil: Image.Image) -> imagehash.ImageHash:
        """
        Computes Perceptual Hash (pHash) for duplicate detection.
        """
        return imagehash.phash(img_pil)

    def clean_split(
        self,
        image_records: List[Dict[str, Any]],
        img_dir: str,
        annotations_by_image: Dict[int, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Processes images in a split, filters out bad files, and returns cleaned image records.
        
        Args:
            image_records: List of COCO image records (dicts).
            img_dir: Path to directory containing images.
            annotations_by_image: Map of image_id -> list of COCO annotations.
            
        Returns:
            List[Dict[str, Any]]: Cleaned image records.
        """
        self.stats["original_count"] = len(image_records)
        logger.info(f"Starting dataset cleaning. Original images count: {len(image_records)}")
        
        # Sort images by clip prefix and frame index to reconstruct temporal sequence for diffing
        sorted_records = []
        for rec in image_records:
            prefix, idx = self.parse_filename(rec["file_name"])
            sorted_records.append((prefix, idx, rec))
        sorted_records.sort(key=lambda x: (x[0], x[1]))
        
        cleaned_records = []
        hash_registry: Dict[imagehash.ImageHash, str] = {} # hash -> file_name of the kept image
        
        # Keep track of previous frame in the sequence for static/replay frame detection
        prev_clip_prefix = None
        prev_img_data = None
        
        for i, (prefix, idx, rec) in enumerate(sorted_records):
            file_name = rec["file_name"]
            img_path = os.path.join(img_dir, file_name)
            img_id = rec["id"]
            
            # Check if file exists
            if not os.path.exists(img_path):
                logger.warning(f"File not found: {img_path}. Skipping.")
                self.stats["corrupted"] += 1
                continue
                
            # 1. Corrupted Image Check (try loading it)
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Failed to load image: {img_path}. Marking as corrupted.")
                self.stats["corrupted"] += 1
                prev_clip_prefix = None
                prev_img_data = None
                continue
                
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # 2. Blurry Frame Check
            blur_score = self.compute_blur_score(img_gray)
            if self.remove_blurry and blur_score < self.blur_threshold:
                logger.debug(f"Image {file_name} is blurry (score={blur_score:.2f} < {self.blur_threshold}). Discarding.")
                self.stats["blurry"] += 1
                prev_clip_prefix = None
                prev_img_data = None
                continue
                
            # 3. Duplicate Frame Check (within split)
            img_hash = self.compute_hash(img_pil)
            is_duplicate = False
            for registered_hash, registered_name in hash_registry.items():
                if (img_hash - registered_hash) <= self.duplicate_threshold:
                    logger.debug(f"Image {file_name} is duplicate of {registered_name} (hash diff={img_hash - registered_hash}). Discarding.")
                    self.stats["duplicate"] += 1
                    is_duplicate = True
                    break
            if self.remove_duplicates and is_duplicate:
                continue
                
            # Register hash for keeping it
            hash_registry[img_hash] = file_name
            
            # 4. Empty Label Check
            labels = annotations_by_image.get(img_id, [])
            if self.remove_empty and not labels:
                logger.debug(f"Image {file_name} contains zero annotations. Discarding.")
                self.stats["empty"] += 1
                prev_clip_prefix = None
                prev_img_data = None
                continue
                
            # 5. Logo Transition / Replay Frame Check (Low pixel difference + no labels)
            if self.remove_logo_replay and (not labels):
                if prev_clip_prefix == prefix and prev_img_data is not None:
                    # Calculate mean absolute error between current and previous frame
                    pixel_diff = np.mean(np.abs(img_gray.astype(np.float32) - prev_img_data.astype(np.float32)))
                    if pixel_diff < self.logo_pixel_diff_threshold:
                        logger.debug(f"Image {file_name} is a static replay/logo transition (frame diff={pixel_diff:.4f} < {self.logo_pixel_diff_threshold} and zero annotations). Discarding.")
                        self.stats["logo_replay"] += 1
                        # Do not update prev_img_data so that next frames are compared to the last moving frame
                        continue
            
            # Save state for the next comparison
            prev_clip_prefix = prefix
            prev_img_data = img_gray
            
            cleaned_records.append(rec)
            
        self.stats["cleaned_count"] = len(cleaned_records)
        logger.info(
            f"Finished dataset cleaning.\n"
            f"Original: {self.stats['original_count']} -> Cleaned: {self.stats['cleaned_count']}\n"
            f"Filtered statistics: Corrupted={self.stats['corrupted']}, Blurry={self.stats['blurry']}, "
            f"Duplicates={self.stats['duplicate']}, Empty={self.stats['empty']}, Logo/Replays={self.stats['logo_replay']}"
        )
        return cleaned_records
