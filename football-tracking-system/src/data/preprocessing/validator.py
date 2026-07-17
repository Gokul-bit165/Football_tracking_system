"""
Validation module for dataset labels.
Maps annotations to canonical classes, checks box constraints, auto-fixes minor issues,
and keeps an explicit change log of modifications made.
"""

from loguru import logger
from typing import Dict, List, Any, Tuple, Set

class LabelValidator:
    """
    Validates and canonicalizes COCO annotations for a dataset split.
    """
    def __init__(self, class_mapping: Dict[str, int]):
        """
        Args:
            class_mapping (Dict[str, int]): Map of expected category names to canonical IDs.
        """
        self.class_mapping = class_mapping
        # Track statistics
        self.total_boxes_checked = 0
        self.total_boxes_clipped = 0
        self.total_boxes_dropped = 0
        self.change_log: List[Dict[str, Any]] = []

    def validate_categories(self, categories: List[Dict[str, Any]], used_category_ids: Set[int]) -> Dict[int, int]:
        """
        Validates categories listed in COCO JSON.
        Maps raw COCO category IDs to canonical category IDs.
        Fails loudly if any raw category name doesn't match the canonical expected names,
        ignoring unused categories.
        
        Args:
            categories: List of COCO category dicts.
            used_category_ids: Set of category IDs actually present in annotations.
            
        Returns:
            Dict[int, int]: Mapping from raw COCO category ID -> canonical ID (0, 1, 2, 3)
        """
        id_mapping = {}
        for cat in categories:
            cat_name = cat.get("name", "").strip().lower()
            cat_id = cat.get("id")
            
            # Skip validation if category is completely unused in the dataset split
            if cat_id not in used_category_ids:
                logger.info(f"Unused category '{cat_name}' (ID: {cat_id}) skipped.")
                continue
                
            if cat_name not in self.class_mapping:
                error_msg = (
                    f"[CRITICAL] Unexpected category '{cat_name}' (ID: {cat_id}) found in COCO labels. "
                    f"Expected categories: {list(self.class_mapping.keys())}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            canonical_id = self.class_mapping[cat_name]
            id_mapping[cat_id] = canonical_id
            logger.info(f"Mapped category '{cat_name}' (raw ID: {cat_id}) -> canonical ID: {canonical_id}")
            
        return id_mapping

    def validate_and_fix_annotation(
        self, 
        bbox: List[float], 
        img_width: int, 
        img_height: int, 
        image_name: str,
        ann_id: int
    ) -> Tuple[bool, List[float], str]:
        """
        Checks bounding box constraints and auto-fixes out-of-bound errors.
        COCO bbox format: [x_min, y_min, width, height]
        
        Args:
            bbox: [x_min, y_min, w, h] in absolute pixels
            img_width: Width of image in pixels
            img_height: Height of image in pixels
            image_name: Filename for tracking logs
            ann_id: ID of annotation for validation logging
            
        Returns:
            Tuple[bool, List[float], str]: (is_valid, corrected_bbox, description_of_fix)
        """
        self.total_boxes_checked += 1
        
        if len(bbox) != 4:
            self.total_boxes_dropped += 1
            return False, [], "Dropped: bounding box does not contain exactly 4 coordinates."
            
        x_min, y_min, w, h = bbox
        
        # Check for absolute degenerate values (NaNs, Infinite)
        import math
        if any(math.isnan(v) or math.isinf(v) for v in [x_min, y_min, w, h]):
            self.total_boxes_dropped += 1
            return False, [], "Dropped: coordinates contain NaN or Inf values."
            
        clipped = False
        desc_parts = []
        
        # Validate coordinates clipping boundaries
        if x_min < 0:
            desc_parts.append(f"clipped x_min from {x_min} to 0")
            w = w + x_min # reduce width by how much we shifted x_min left
            x_min = 0.0
            clipped = True
            
        if y_min < 0:
            desc_parts.append(f"clipped y_min from {y_min} to 0")
            h = h + y_min # reduce height by how much we shifted y_min up
            y_min = 0.0
            clipped = True
            
        if x_min >= img_width:
            self.total_boxes_dropped += 1
            return False, [], f"Dropped: x_min ({x_min}) outside image width ({img_width})."
            
        if y_min >= img_height:
            self.total_boxes_dropped += 1
            return False, [], f"Dropped: y_min ({y_min}) outside image height ({img_height})."
            
        # Clip max width/height
        if x_min + w > img_width:
            desc_parts.append(f"clipped width from {w} to {img_width - x_min}")
            w = img_width - x_min
            clipped = True
            
        if y_min + h > img_height:
            desc_parts.append(f"clipped height from {h} to {img_height - y_min}")
            h = img_height - y_min
            clipped = True
            
        # Validate area
        if w <= 0 or h <= 0:
            self.total_boxes_dropped += 1
            return False, [], f"Dropped: degenerate size after check (width={w}, height={h})."
            
        if clipped:
            self.total_boxes_clipped += 1
            desc = ", ".join(desc_parts)
            fix_log = {
                "image_name": image_name,
                "annotation_id": ann_id,
                "original_bbox": bbox,
                "corrected_bbox": [x_min, y_min, w, h],
                "fix_applied": desc
            }
            self.change_log.append(fix_log)
            logger.debug(f"Auto-fixed bbox in {image_name}: {desc}")
            return True, [x_min, y_min, w, h], desc
            
        return True, [x_min, y_min, w, h], ""

    def get_summary_report(self) -> Dict[str, Any]:
        """
        Returns a dictionary summary of validation corrections made.
        """
        return {
            "total_boxes_checked": self.total_boxes_checked,
            "total_boxes_clipped": self.total_boxes_clipped,
            "total_boxes_dropped": self.total_boxes_dropped,
            "fixes_applied": len(self.change_log),
            "change_log": self.change_log
        }
