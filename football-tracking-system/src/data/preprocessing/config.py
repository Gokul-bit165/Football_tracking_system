"""
Configuration module for the football detection preprocessing pipeline.
Loads and validates settings from a YAML configuration file.
"""

import os
import yaml
from typing import List, Dict, Any, Optional

class PreprocessingConfig:
    """
    Holds configuration values for the preprocessing pipeline.
    """
    def __init__(self, config_dict: Dict[str, Any], project_root: str):
        self.project_root = project_root
        
        # Version and Seed
        self.version_tag: str = config_dict.get("version_tag", "v1")
        self.seed: int = config_dict.get("seed", 42)
        
        # Paths
        coco_sources_raw: List[str] = config_dict.get("coco_sources", [])
        self.coco_sources: List[str] = [
            os.path.abspath(os.path.join(project_root, p)) for p in coco_sources_raw
        ]
        
        self.output_dir: str = os.path.abspath(
            os.path.join(project_root, config_dict.get("output_dir", "data/processed"))
        )
        self.interim_dir: str = os.path.abspath(
            os.path.join(project_root, config_dict.get("interim_dir", "data/interim"))
        )
        
        # Canonical Class Mapping
        self.class_mapping: Dict[str, int] = config_dict.get("class_mapping", {
            "ball": 0,
            "goalkeeper": 1,
            "player": 2,
            "referee": 3
        })
        
        # Cleaning Options
        cleaning_dict = config_dict.get("cleaning", {})
        self.remove_blurry: bool = cleaning_dict.get("remove_blurry", True)
        self.blur_threshold: float = float(cleaning_dict.get("blur_threshold", 100.0))
        self.remove_duplicates: bool = cleaning_dict.get("remove_duplicates", True)
        self.duplicate_threshold: int = int(cleaning_dict.get("duplicate_threshold", 5))
        self.remove_empty: bool = cleaning_dict.get("remove_empty", True)
        self.remove_logo_replay: bool = cleaning_dict.get("remove_logo_replay", True)
        self.logo_pixel_diff_threshold: float = float(cleaning_dict.get("logo_pixel_diff_threshold", 1.5))
        
        # Balancing Options
        balancing_dict = config_dict.get("balancing", {})
        self.oversample_ball: bool = balancing_dict.get("oversample_ball", True)
        self.oversample_factor: int = int(balancing_dict.get("oversample_factor", 3))
        
        # Augmentations Options
        aug_dict = config_dict.get("augmentations", {})
        self.augmentations_enabled: bool = aug_dict.get("enabled", True)
        self.image_size: int = int(aug_dict.get("image_size", 1280))
        
        # Extract per-augmentation configs
        self.aug_hsv: Dict[str, Any] = aug_dict.get("hsv", {"enabled": False})
        self.aug_motion_blur: Dict[str, Any] = aug_dict.get("motion_blur", {"enabled": False})
        self.aug_perspective: Dict[str, Any] = aug_dict.get("perspective", {"enabled": False})
        self.aug_horizontal_flip: Dict[str, Any] = aug_dict.get("horizontal_flip", {"enabled": False})
        self.aug_ball_copy_paste: Dict[str, Any] = aug_dict.get("ball_copy_paste", {"enabled": False})
        
        # Advanced augmentation stubs
        self.aug_rain: Dict[str, Any] = aug_dict.get("rain", {"enabled": False})
        self.aug_fog: Dict[str, Any] = aug_dict.get("fog", {"enabled": False})
        self.aug_night_sim: Dict[str, Any] = aug_dict.get("night_sim", {"enabled": False})
        self.aug_camera_shake: Dict[str, Any] = aug_dict.get("camera_shake", {"enabled": False})
        self.aug_compression: Dict[str, Any] = aug_dict.get("compression", {"enabled": False})
        self.aug_mixup: Dict[str, Any] = aug_dict.get("mixup", {"enabled": False})
        self.aug_random_crop: Dict[str, Any] = aug_dict.get("random_crop", {"enabled": False})

    @classmethod
    def load_from_yaml(cls, yaml_path: str, project_root: str) -> "PreprocessingConfig":
        """
        Loads YAML file and returns a configuration object.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
            
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
            
        return cls(config_dict, project_root)
