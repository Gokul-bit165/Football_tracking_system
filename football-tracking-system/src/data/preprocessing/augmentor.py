"""
Augmentation module for dataset expansion and domain gap reduction.
Applies Albumentations and custom copy-paste operations to the train split.
Ensures deterministic, parallel-safe executions using stable per-image hashing.
"""

import os
import cv2
import random
import hashlib
import numpy as np
import albumentations as A
from loguru import logger
from typing import List, Dict, Any, Tuple

def get_image_seed(filename: str, global_seed: int) -> int:
    """
    Computes a stable, deterministic seed for an image using MD5.
    Guarantees reproducibility even under multiprocessing and parallel executions.
    """
    hash_object = hashlib.md5(filename.encode("utf-8"))
    hash_hex = hash_object.hexdigest()
    # Convert first 8 characters of hex to int and add global_seed
    stable_hash = int(hash_hex[:8], 16)
    return (stable_hash + global_seed) % (2**31 - 1)

class DatasetAugmentor:
    """
    Applies Albumentations and custom augmentations (e.g. ball copy-paste) to training frames.
    """
    def __init__(self, config: Any):
        """
        Args:
            config: PreprocessingConfig object.
        """
        self.config = config
        self.global_seed = config.seed
        self.interim_dir = config.interim_dir
        self.ball_crops_dir = os.path.join(self.interim_dir, "ball_crops")
        
        # Load ball crop filenames if copy-paste is enabled
        self.ball_crops: List[str] = []
        if self.config.aug_ball_copy_paste.get("enabled", True):
            if os.path.exists(self.ball_crops_dir):
                self.ball_crops = [
                    f for f in os.listdir(self.ball_crops_dir) 
                    if f.endswith(".jpg") or f.endswith(".png")
                ]
                logger.info(f"Loaded {len(self.ball_crops)} ball crops for Copy-Paste augmentation.")
            else:
                logger.warning(f"Ball crops directory not found at {self.ball_crops_dir}. Copy-Paste will be skipped.")

    def build_albumentations_pipeline(self, image_seed: int) -> A.Compose:
        """
        Builds the Albumentations pipeline based on config toggles.
        """
        transform_list = []
        
        # HSV shift
        if self.config.aug_hsv.get("enabled", True):
            hsv_conf = self.config.aug_hsv
            transform_list.append(
                A.HueSaturationValue(
                    hue_shift_limit=int(hsv_conf.get("hgain", 0.015) * 180),
                    sat_shift_limit=int(hsv_conf.get("sgain", 0.7) * 255),
                    val_shift_limit=int(hsv_conf.get("vgain", 0.4) * 255),
                    p=hsv_conf.get("p", 0.5)
                )
            )
            
        # Motion Blur
        if self.config.aug_motion_blur.get("enabled", True):
            blur_conf = self.config.aug_motion_blur
            limit = blur_conf.get("blur_limit", [3, 7])
            transform_list.append(
                A.MotionBlur(
                    blur_limit=limit,
                    p=blur_conf.get("p", 0.5)
                )
            )
            
        # Perspective / Affine
        if self.config.aug_perspective.get("enabled", True):
            pers_conf = self.config.aug_perspective
            scale = pers_conf.get("scale", [0.05, 0.1])
            # Albumentations Perspective scale maps to translation/scaling
            transform_list.append(
                A.Perspective(
                    scale=scale,
                    p=pers_conf.get("p", 0.5)
                )
            )
            
        # Horizontal Flip
        if self.config.aug_horizontal_flip.get("enabled", True):
            flip_conf = self.config.aug_horizontal_flip
            transform_list.append(
                A.HorizontalFlip(
                    p=flip_conf.get("p", 0.5)
                )
            )
            
        # --- Advanced Augmentation Stubs (Toggled off by default) ---
        if self.config.aug_rain.get("enabled", False):
            # Placeholder: Simulates rain lines
            transform_list.append(A.RandomRain(p=self.config.aug_rain.get("p", 0.2)))
            
        if self.config.aug_fog.get("enabled", False):
            # Placeholder: Simulates fog
            transform_list.append(A.RandomFog(p=self.config.aug_fog.get("p", 0.2)))
            
        if self.config.aug_night_sim.get("enabled", False):
            # Night simulation: Reduce gamma & modify contrast
            gamma_range = self.config.aug_night_sim.get("gamma_range", [0.4, 0.7])
            transform_list.append(
                A.RandomGamma(
                    gamma_limit=(int(gamma_range[0] * 100), int(gamma_range[1] * 100)),
                    p=self.config.aug_night_sim.get("p", 0.5)
                )
            )
            
        if self.config.aug_camera_shake.get("enabled", False):
            # Camera shake using small random shifts
            transform_list.append(
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=10,
                    p=self.config.aug_camera_shake.get("p", 0.3)
                )
            )
            
        if self.config.aug_compression.get("enabled", False):
            # Video compression noise
            transform_list.append(
                A.ImageCompression(
                    quality_range=(30, 80),
                    p=self.config.aug_compression.get("p", 0.3)
                )
            )
            
        # MixUp and RandomCrop will be executed inline/manually if enabled, 
        # but we add a Crop transformation here as a placeholder for RandomCrop
        if self.config.aug_random_crop.get("enabled", False):
            crop_scale = self.config.aug_random_crop.get("crop_scale", [0.8, 1.0])
            # Placeholder: Albumentations RandomSizedBBoxSafeCrop
            transform_list.append(
                A.RandomSizedBBoxSafeCrop(
                    width=self.config.image_size,
                    height=self.config.image_size,
                    erosion_rate=0.0,
                    p=self.config.aug_random_crop.get("p", 0.5)
                )
            )
            
        # Set up pipeline. We use BboxParams to handle box coordinate changes.
        # coco format: [x_min, y_min, width, height] (absolute scale handled if label_fields specified)
        # We pass coordinates as min/max ratios (albumentations expects normalized coordinates: x_min, y_min, x_max, y_max)
        bbox_params = A.BboxParams(
            format="albumentations", 
            label_fields=["class_labels"]
        )
        
        # Seed Albumentations operations using the per-image seed
        random.seed(image_seed)
        np.random.seed(image_seed)
        
        return A.Compose(transform_list, bbox_params=bbox_params)

    def apply_ball_copy_paste(
        self,
        img: np.ndarray,
        bboxes: List[List[float]],
        class_labels: List[int],
        image_seed: int
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Pastes ball crops on the image at random locations and updates bounding boxes.
        
        Args:
            img: OpenCV BGR image.
            bboxes: Bounding boxes in normalized [x_min, y_min, x_max, y_max] format.
            class_labels: List of canonical class labels.
            image_seed: The deterministic seed for this image.
            
        Returns:
            Tuple[np.ndarray, List[List[float]], List[int]]: Modified image, boxes, labels.
        """
        if not self.ball_crops:
            return img, bboxes, class_labels
            
        # Seed local RNG deterministically for this image
        local_rng = random.Random(image_seed)
        
        # Determine if we execute paste
        p = self.config.aug_ball_copy_paste.get("p", 0.5)
        if local_rng.random() > p:
            return img, bboxes, class_labels
            
        max_paste = self.config.aug_ball_copy_paste.get("max_balls_to_paste", 3)
        num_pastes = local_rng.randint(1, max_paste)
        
        img_height, img_width = img.shape[:2]
        
        new_img = img.copy()
        new_bboxes = list(bboxes)
        new_labels = list(class_labels)
        
        for _ in range(num_pastes):
            crop_name = local_rng.choice(self.ball_crops)
            crop_path = os.path.join(self.ball_crops_dir, crop_name)
            
            crop = cv2.imread(crop_path)
            if crop is None:
                continue
                
            ch, cw = crop.shape[:2]
            if ch <= 0 or cw <= 0:
                continue
                
            # Random scale crop slightly (0.8x to 1.3x)
            scale = local_rng.uniform(0.8, 1.3)
            nw, nh = int(round(cw * scale)), int(round(ch * scale))
            if nw > 0 and nh > 0:
                crop = cv2.resize(crop, (nw, nh))
                ch, cw = nh, nw
                
            # Apply slight rotation or HSV shift to crop itself deterministically
            # (simple flip or HSV tweak)
            if local_rng.random() > 0.5:
                crop = cv2.flip(crop, 1)
                
            # Choose a random position inside image boundaries
            max_x = img_width - cw
            max_y = img_height - ch
            if max_x <= 0 or max_y <= 0:
                continue
                
            px = local_rng.randint(0, max_x)
            py = local_rng.randint(0, max_y)
            
            # Simple overwrite pasting (no advanced alpha blending needed for tiny balls,
            # but we can do a minor edge feathering if desired. Overwriting works well.)
            new_img[py:py+ch, px:px+cw] = crop
            
            # Add new normalized bbox: [x_min, y_min, x_max, y_max]
            x_min_norm = px / img_width
            y_min_norm = py / img_height
            x_max_norm = (px + cw) / img_width
            y_max_norm = (py + ch) / img_height
            
            new_bboxes.append([x_min_norm, y_min_norm, x_max_norm, y_max_norm])
            new_labels.append(0) # 0 is ball
            
        return new_img, new_bboxes, new_labels

    def augment_image(
        self,
        img: np.ndarray,
        bboxes_coco: List[List[float]],
        class_labels: List[int],
        filename: str,
        split_name: str,
        replica_id: int = 0
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Main entry point for augmenting a single image.
        Ensures validations, deterministic seeding, Albumentations, and copy-paste.
        
        Args:
            img: Input image (BGR).
            bboxes_coco: List of COCO bounding boxes [[x_min, y_min, w, h]].
            class_labels: List of canonical class labels.
            filename: Image filename.
            split_name: Split name (must be 'train' to apply augmentations).
            replica_id: ID for replica copy (used in oversampling).
            
        Returns:
            Tuple[np.ndarray, List[List[float]], List[int]]: Augmented (image, bboxes in COCO format, labels).
        """
        img_height, img_width = img.shape[:2]
        
        # Enforce train-only scope
        assert split_name == "train", f"[ERROR] Augmentor was called on split '{split_name}'. Augmentations must ONLY run on train split!"
        
        # Derive stable per-image seed.
        # Include replica_id to ensure duplicates of the same frame get different augmentations.
        seed_key = f"{filename}_rep{replica_id}"
        image_seed = get_image_seed(seed_key, self.global_seed)
        
        # Convert COCO absolute [x, y, w, h] to normalized [x_min, y_min, x_max, y_max]
        norm_bboxes = []
        for box in bboxes_coco:
            x, y, w, h = box
            x_min = x / img_width
            y_min = y / img_height
            x_max = (x + w) / img_width
            y_max = (y + h) / img_height
            
            # Clip bounds to avoid minor float rounding exceedances
            x_min = max(0.0, min(x_min, 1.0))
            y_min = max(0.0, min(y_min, 1.0))
            x_max = max(x_min + 0.0001, min(x_max, 1.0))
            y_max = max(y_min + 0.0001, min(y_max, 1.0))
            
            norm_bboxes.append([x_min, y_min, x_max, y_max])
            
        # 1. Custom Ball Copy-Paste Augmentation (applied first, so Albumentations affects the pasted ball too)
        if self.config.aug_ball_copy_paste.get("enabled", True):
            img, norm_bboxes, class_labels = self.apply_ball_copy_paste(
                img, norm_bboxes, class_labels, image_seed
            )
            
        # 2. Albumentations transforms
        if self.config.augmentations_enabled:
            pipeline = self.build_albumentations_pipeline(image_seed)
            try:
                transformed = pipeline(
                    image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    bboxes=norm_bboxes,
                    class_labels=class_labels
                )
                img = cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR)
                norm_bboxes = transformed["bboxes"]
                class_labels = transformed["class_labels"]
            except Exception as e:
                logger.error(f"Albumentations failed for {filename} with seed {image_seed}: {e}")
                # Fallback to unaugmented copy
                pass
                
        # 3. MixUp / RandomCrop inline stubs
        # (MixUp requires blending with another frame, which is handled at train time.
        # If MixUp config was enabled: print a debug stub log)
        if self.config.aug_mixup.get("enabled", False):
            logger.debug(f"MixUp placeholder activated for {filename} (train-time mixup recommended).")
            
        # Convert norm bboxes back to absolute COCO [x, y, w, h] format
        final_bboxes_coco = []
        for box in norm_bboxes:
            x_min, y_min, x_max, y_max = box
            px_min = x_min * img_width
            py_min = y_min * img_height
            pw = (x_max - x_min) * img_width
            ph = (y_max - y_min) * img_height
            final_bboxes_coco.append([px_min, py_min, pw, ph])
            
        return img, final_bboxes_coco, class_labels
