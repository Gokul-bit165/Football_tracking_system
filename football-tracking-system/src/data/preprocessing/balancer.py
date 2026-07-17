"""
Balancing module for handling class imbalance.
Calculates class distributions, computes inverse-frequency class weights,
and oversamples train images containing the ball class.
"""

import os
import json
from loguru import logger
from typing import List, Dict, Any, Tuple

class DatasetBalancer:
    """
    Computes class statistics and oversamples ball-containing frames in the train split.
    """
    def __init__(
        self,
        oversample_ball: bool = True,
        oversample_factor: int = 3,
        ball_class_id: int = 0
    ):
        self.oversample_ball = oversample_ball
        self.oversample_factor = oversample_factor
        self.ball_class_id = ball_class_id

    def calculate_distribution(
        self,
        image_records: List[Dict[str, Any]],
        annotations_by_image: Dict[int, List[Dict[str, Any]]],
        category_id_mapping: Dict[int, int]
    ) -> Tuple[Dict[int, int], Dict[int, float]]:
        """
        Calculates category frequencies and computes inverse-frequency class weights.
        """
        counts = {0: 0, 1: 0, 2: 0, 3: 0} # Canonical IDs
        
        for rec in image_records:
            image_id = rec["id"]
            for ann in annotations_by_image.get(image_id, []):
                raw_cat_id = ann.get("category_id")
                canonical_id = category_id_mapping.get(raw_cat_id)
                if canonical_id in counts:
                    counts[canonical_id] += 1
                    
        total_instances = sum(counts.values())
        weights = {}
        
        if total_instances > 0:
            num_classes = len(counts)
            for cid, count in counts.items():
                if count > 0:
                    # Inverse frequency formula normalized so average weight is 1.0
                    raw_w = total_instances / count
                    weights[cid] = raw_w
                else:
                    weights[cid] = 1.0
                    
            # Normalize so that weights sum to the number of classes
            sum_w = sum(weights.values())
            if sum_w > 0:
                for cid in weights:
                    weights[cid] = (weights[cid] / sum_w) * num_classes
        else:
            weights = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
            
        return counts, weights

    def save_class_weights(
        self,
        counts: Dict[int, int],
        weights: Dict[int, float],
        output_path: str
    ):
        """
        Saves computed counts and weights as a class_weights.json file.
        """
        # Convert keys to string for JSON serialization compatibility
        data = {
            "counts": {str(k): v for k, v in counts.items()},
            "weights": {str(k): round(v, 4) for k, v in weights.items()}
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved class weights and counts to {output_path}")

    def balance_train_split(
        self,
        train_image_records: List[Dict[str, Any]],
        annotations_by_image: Dict[int, List[Dict[str, Any]]],
        category_id_mapping: Dict[int, int],
        split_name: str
    ) -> List[Dict[str, Any]]:
        """
        Oversamples images containing balls in the train split.
        Asserts split is strictly 'train'.
        """
        assert split_name == "train", f"[ERROR] Balancer was called on split '{split_name}'. Balancing/Oversampling must ONLY run on train split!"
        
        if not self.oversample_ball or self.oversample_factor <= 1:
            logger.info("Oversampling disabled or factor <= 1. Returning original split list.")
            return train_image_records
            
        logger.info(f"Starting oversampling for train split. Factor: {self.oversample_factor}x")
        
        oversampled_records = []
        ball_containing_count = 0
        
        for rec in train_image_records:
            image_id = rec["id"]
            has_ball = False
            
            # Check if this image has a ball
            for ann in annotations_by_image.get(image_id, []):
                raw_cat_id = ann.get("category_id")
                canonical_id = category_id_mapping.get(raw_cat_id)
                if canonical_id == self.ball_class_id:
                    has_ball = True
                    break
                    
            # Add the original record
            oversampled_records.append(rec)
            
            if has_ball:
                ball_containing_count += 1
                # Duplicate the record reference (factor - 1) times
                # Add copy metadata indicating it's an oversampled replica
                for c_idx in range(1, self.oversample_factor):
                    replica = rec.copy()
                    # We store annotation copy info or replica suffix in the dict
                    replica["__replica_id__"] = c_idx
                    oversampled_records.append(replica)
                    
        total_final = len(oversampled_records)
        added_count = total_final - len(train_image_records)
        
        logger.info(
            f"Oversampling complete.\n"
            f"Train split images: {len(train_image_records)} -> {total_final} (Added {added_count} ball-containing copies).\n"
            f"Unique ball-containing frames in train: {ball_containing_count}."
        )
        
        return oversampled_records
