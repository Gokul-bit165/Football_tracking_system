"""
Orchestration script for the football tracker dataset preprocessing pipeline.
Sequentially runs: config load -> label validation -> dataset cleaning -> cross-split leak check ->
ball crop extraction -> class balancing -> train-only augmentations -> YOLO export -> statistical reporting.
"""

import os
import argparse
import json
import shutil
import cv2
from loguru import logger
from tqdm import tqdm
from typing import Dict, List, Any

# Import modules from preprocessing package
from src.data.preprocessing.config import PreprocessingConfig
from src.data.preprocessing.validator import LabelValidator
from src.data.preprocessing.cleaner import DatasetCleaner
from src.data.preprocessing.ball_crop_extractor import BallCropExtractor
from src.data.preprocessing.balancer import DatasetBalancer
from src.data.preprocessing.augmentor import DatasetAugmentor
from src.data.preprocessing.exporter import YOLOExporter
from src.data.preprocessing.analyzer import DatasetAnalyzer

def run_preprocessing_pipeline(config_path: str):
    """
    Executes the entire end-to-end preprocessing pipeline.
    """
    logger.info("Initializing preprocessing pipeline orchestrator...")
    
    # 1. Load Configuration
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    config = PreprocessingConfig.load_from_yaml(config_path, project_root)
    logger.info(f"Loaded config from {config_path}. Version: '{config.version_tag}', Seed: {config.seed}")
    
    # Define and prepare processed folder
    processed_dir = os.path.join(config.output_dir, config.version_tag)
    cleaned_dataset_dir = os.path.join(processed_dir, "cleaned_dataset")
    os.makedirs(processed_dir, exist_ok=True)
    
    # Initialize components
    validator = LabelValidator(config.class_mapping)
    cleaner = DatasetCleaner(
        blur_threshold=config.blur_threshold,
        duplicate_threshold=config.duplicate_threshold,
        logo_pixel_diff_threshold=config.logo_pixel_diff_threshold,
        remove_blurry=config.remove_blurry,
        remove_duplicates=config.remove_duplicates,
        remove_empty=config.remove_empty,
        remove_logo_replay=config.remove_logo_replay
    )
    balancer = DatasetBalancer(
        oversample_ball=config.oversample_ball,
        oversample_factor=config.oversample_factor,
        ball_class_id=0
    )
    exporter = YOLOExporter(
        output_dir=cleaned_dataset_dir,
        class_names=list(config.class_mapping.keys())
    )
    analyzer = DatasetAnalyzer(
        class_mapping=config.class_mapping,
        duplicate_threshold=config.duplicate_threshold
    )
    
    exporter.prepare_directories(["train", "val", "test"])
    
    # We will process each COCO source (currently 1 source)
    if not config.coco_sources:
        logger.error("No COCO dataset sources specified in configuration!")
        return
        
    raw_source = config.coco_sources[0]
    logger.info(f"Processing COCO raw dataset source: {raw_source}")
    
    # Map raw splits (train, valid, test) to YOLO splits (train, val, test)
    split_mapping = {
        "train": "train",
        "valid": "val",
        "test": "test"
    }
    
    # Store intermediate cleaned data for cross-split duplicate checks and reports
    cleaned_splits_data = {}
    original_split_counts = {}
    
    # We also keep stats dictionaries to log before vs after
    before_preprocessing_stats = {}
    after_preprocessing_stats = {}
    
    # Track counts for final summary
    final_counts = {}
    
    # Execute processing split by split
    for raw_split, yolo_split in split_mapping.items():
        split_dir = os.path.join(raw_source, raw_split)
        coco_json_path = os.path.join(split_dir, "_annotations.coco.json")
        
        if not os.path.exists(coco_json_path):
            logger.warning(f"COCO annotations not found at {coco_json_path}. Skipping split '{raw_split}'.")
            continue
            
        logger.info(f"--- Processing Split: {raw_split} -> YOLO: {yolo_split} ---")
        
        # Load COCO json
        with open(coco_json_path, "r") as f:
            coco_data = json.load(f)
            
        images = coco_data.get("images", [])
        original_split_counts[yolo_split] = len(images)
        annotations = coco_data.get("annotations", [])
        categories = coco_data.get("categories", [])
        
        # A. Validate categories (canonical class mapping)
        used_category_ids = {ann["category_id"] for ann in annotations}
        category_id_mapping = validator.validate_categories(categories, used_category_ids)
        
        # Index annotations by image_id
        annotations_by_image = {}
        for ann in annotations:
            img_id = ann["image_id"]
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
            
        # B. Validate & Auto-fix annotations box constraints
        valid_annotations_by_image = {}
        for img_id, anns in annotations_by_image.items():
            img_rec = next((im for im in images if im["id"] == img_id), None)
            if not img_rec:
                continue
            w_img, h_img = img_rec["width"], img_rec["height"]
            img_name = img_rec["file_name"]
            
            valid_anns = []
            for ann in anns:
                bbox = ann.get("bbox")
                ann_id = ann.get("id")
                
                is_valid, corrected_bbox, desc = validator.validate_and_fix_annotation(
                    bbox, w_img, h_img, img_name, ann_id
                )
                
                if is_valid:
                    ann["bbox"] = corrected_bbox
                    valid_annotations_by_image[img_id] = valid_annotations_by_image.get(img_id, []) + [ann]
                    
        # C. Clean Dataset (corrupted, blurry, duplicates, static frames)
        cleaned_images = cleaner.clean_split(images, split_dir, valid_annotations_by_image)
        cleaned_splits_data[yolo_split] = (cleaned_images, split_dir)
        
        # Compile stats before balancing/augmentation for train
        if yolo_split == "train":
            before_preprocessing_stats = analyzer.compute_split_stats(
                cleaned_images, split_dir, valid_annotations_by_image, category_id_mapping
            )
            
        # D. Train Split specific operations (Ball crops, Balancing, Augmentation)
        if yolo_split == "train":
            # 1. Extract ball crops from clean train split
            logger.info("Extracting ball crops from cleaned train split...")
            extractor = BallCropExtractor(config.interim_dir)
            extractor.extract_crops(cleaned_images, split_dir, valid_annotations_by_image, category_id_mapping)
            
            # 2. Oversample ball class & calculate class weights
            logger.info("Balancing class instances in train split...")
            train_counts, train_weights = balancer.calculate_distribution(
                cleaned_images, valid_annotations_by_image, category_id_mapping
            )
            balancer.save_class_weights(
                train_counts, train_weights, os.path.join(processed_dir, "class_weights.json")
            )
            
            balanced_train_images = balancer.balance_train_split(
                cleaned_images, valid_annotations_by_image, category_id_mapping, yolo_split
            )
            
            # 3. Apply deterministic Smart Augmentations & Export
            logger.info("Applying Albumentations & Copy-Paste augmentations and exporting...")
            augmentor = DatasetAugmentor(config)
            
            exported_images_count = 0
            exported_boxes_count = 0
            
            for rec in tqdm(balanced_train_images, desc="Augmenting & Exporting Train split"):
                file_name = rec["file_name"]
                img_path = os.path.join(split_dir, file_name)
                
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                image_id = rec["id"]
                anns = valid_annotations_by_image.get(image_id, [])
                
                bboxes_coco = [ann["bbox"] for ann in anns]
                class_labels = [category_id_mapping[ann["category_id"]] for ann in anns]
                
                # Fetch replica index if oversampled copy
                replica_id = rec.get("__replica_id__", 0)
                
                # Run train augmentations
                aug_img, aug_bboxes, aug_labels = augmentor.augment_image(
                    img, bboxes_coco, class_labels, file_name, yolo_split, replica_id
                )
                
                # Export to YOLO format
                _, boxes_written = exporter.export_image_and_labels(
                    aug_img, aug_bboxes, aug_labels, file_name, yolo_split, replica_id
                )
                
                exported_images_count += 1
                exported_boxes_count += boxes_written
                
            exporter.log_split_statistics(yolo_split, exported_images_count, exported_boxes_count)
            final_counts[yolo_split] = {"images": exported_images_count, "labels": exported_boxes_count}
            
        else:
            # E. Val & Test splits pass through cleaned-but-otherwise-untouched
            logger.info(f"Exporting clean split '{yolo_split}' (no augmentations/balancing)...")
            exported_images_count = 0
            exported_boxes_count = 0
            
            for rec in tqdm(cleaned_images, desc=f"Exporting Clean {yolo_split}"):
                file_name = rec["file_name"]
                img_path = os.path.join(split_dir, file_name)
                
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                image_id = rec["id"]
                anns = valid_annotations_by_image.get(image_id, [])
                
                bboxes_coco = [ann["bbox"] for ann in anns]
                class_labels = [category_id_mapping[ann["category_id"]] for ann in anns]
                
                # Export clean copy
                _, boxes_written = exporter.export_image_and_labels(
                    img, bboxes_coco, class_labels, file_name, yolo_split
                )
                
                exported_images_count += 1
                exported_boxes_count += boxes_written
                
            exporter.log_split_statistics(yolo_split, exported_images_count, exported_boxes_count)
            final_counts[yolo_split] = {"images": exported_images_count, "labels": exported_boxes_count}

    # 4. Generate data.yaml config for YOLO
    exporter.generate_data_yaml()

    # 5. Run Cross-Split Data Leakage Check
    leaks = analyzer.check_cross_split_leakage(cleaned_splits_data)
    
    # 6. Analyze Final Processed Train split and Save Plots
    # Parse final processed dataset train split for final statistics comparison
    logger.info("Computing final processed train statistics...")
    final_processed_train_images = []
    final_train_labels_dir = os.path.join(cleaned_dataset_dir, "labels", "train")
    final_train_images_dir = os.path.join(cleaned_dataset_dir, "images", "train")
    
    if os.path.exists(final_train_images_dir):
        for fname in os.listdir(final_train_images_dir):
            if fname.endswith(".jpg") or fname.endswith(".png"):
                final_processed_train_images.append({
                    "id": len(final_processed_train_images) + 1,
                    "file_name": fname
                })
                
    # Read final YOLO label files directly to compile final counts
    final_annotations_by_image = {}
    final_cat_mapping = {0: 0, 1: 1, 2: 2, 3: 3} # Already canonicalized
    
    for rec in final_processed_train_images:
        img_id = rec["id"]
        fname = rec["file_name"]
        lbl_name = f"{os.path.splitext(fname)[0]}.txt"
        lbl_path = os.path.join(final_train_labels_dir, lbl_name)
        
        final_annotations_by_image[img_id] = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cid = int(float(parts[0]))
                        # Map normalized coordinates back to absolute fake bbox for analyzer compatibility
                        x_c, y_c, nw, nh = map(float, parts[1:])
                        # fake absolute values
                        w_img, h_img = 1920, 1080
                        w = nw * w_img
                        h = nh * h_img
                        x = (x_c * w_img) - w/2
                        y = (y_c * h_img) - h/2
                        final_annotations_by_image[img_id].append({
                            "category_id": cid,
                            "bbox": [x, y, w, h]
                        })
                        
    after_preprocessing_stats = analyzer.compute_split_stats(
        final_processed_train_images,
        final_train_images_dir,
        final_annotations_by_image,
        final_cat_mapping
    )
    
    # Save visual comparison charts
    logger.info("Generating and saving analysis plots...")
    plots_dir = os.path.join(processed_dir, "statistics")
    analyzer.generate_and_save_plots(before_preprocessing_stats, after_preprocessing_stats, plots_dir)
    
    # Compile final JSON/CSV reports
    reports_dir = os.path.join(processed_dir, "reports")
    report_data = {
        "version_tag": config.version_tag,
        "seed": config.seed,
        "class_mapping": config.class_mapping,
        "leaked_image_pairs_count": len(leaks),
        "leaks": leaks,
        "cleaner_exclusions": {
            "corrupted": cleaner.stats["corrupted"],
            "blurry": cleaner.stats["blurry"],
            "duplicate": cleaner.stats["duplicate"],
            "empty": cleaner.stats["empty"],
            "logo_replay": cleaner.stats["logo_replay"],
            "original_train": original_split_counts.get("train", 0),
            "original_val": original_split_counts.get("val", 0),
            "original_test": original_split_counts.get("test", 0)
        },
        "validator_fixes": validator.get_summary_report(),
        "train_split_before_stats": before_preprocessing_stats,
        "train_split_after_stats": after_preprocessing_stats,
        "final_counts": final_counts
    }
    # Strip detailed change log from raw reports JSON file to avoid bloated report exports, keeping summary counts
    report_data_stripped = report_data.copy()
    report_data_stripped["validator_fixes"] = validator.get_summary_report().copy()
    if "change_log" in report_data_stripped["validator_fixes"]:
        del report_data_stripped["validator_fixes"]["change_log"]
        
    analyzer.export_reports(report_data_stripped, reports_dir)
    
    # 7. Generate final preprocessing_summary.md file
    generate_preprocessing_summary_md(
        report_data=report_data,
        processed_dir=processed_dir,
        version_tag=config.version_tag,
        class_names=list(config.class_mapping.keys())
    )
    
    logger.info("==================================================================")
    logger.info(f"Preprocessing Pipeline Completed Successfully for Version '{config.version_tag}'!")
    logger.info(f"Final cleaned YOLO dataset saved to: {cleaned_dataset_dir}")
    logger.info(f"Review the summary report at: {os.path.join(processed_dir, 'preprocessing_summary.md')}")
    logger.info("==================================================================")

def generate_preprocessing_summary_md(
    report_data: Dict[str, Any],
    processed_dir: str,
    version_tag: str,
    class_names: List[str]
):
    """
    Writes a preprocessing_summary.md file in the processed directory detailing statistics.
    """
    summary_path = os.path.join(processed_dir, "preprocessing_summary.md")
    
    # Compile class weights
    weights_path = os.path.join(processed_dir, "class_weights.json")
    class_weights_text = "N/A"
    if os.path.exists(weights_path):
        with open(weights_path, "r") as wf:
            class_weights_text = "```json\n" + wf.read() + "\n```"
            
    # Compile leaks list
    leaks = report_data.get("leaks", [])
    leak_text = ""
    if leaks:
        leak_text = "> [!WARNING]\n> **Potential Data Leakage Detected!** Near-duplicates exist across splits:\n"
        for leak in leaks:
            leak_text += f"> - `{leak['split_a']}/{leak['file_a']}` and `{leak['split_b']}/{leak['file_b']}` (Hamming Distance: {leak['hamming_distance']})\n"
    else:
        leak_text = "*No cross-split data leakage detected.*"
        
    # Compile validator log
    val_report = report_data.get("validator_fixes", {})
    detailed_fixes = val_report.get("change_log", [])
    fixes_list_text = ""
    if detailed_fixes:
        fixes_list_text = "### Detailed Box Fixes Change Log\n| Image Name | Box ID | Original [x,y,w,h] | Fixed [x,y,w,h] | Action Taken |\n| --- | --- | --- | --- | --- |\n"
        # Limit to first 25 fixes to avoid summary document bloat
        for fix in detailed_fixes[:25]:
            fixes_list_text += f"| `{fix['image_name']}` | `{fix['annotation_id']}` | `{fix['original_bbox']}` | `{fix['corrected_bbox']}` | {fix['fix_applied']} |\n"
        if len(detailed_fixes) > 25:
            fixes_list_text += f"\n*(Showing first 25 of {len(detailed_fixes)} total fixes. See reports/dataset_analysis_report.json for complete log)*\n"
    else:
        fixes_list_text = "*No label boxes required corrections.*"

    content = f"""# Preprocessing Summary Report - Version {version_tag}

This report outlines the data quality analytics, cleaning filters, balancing oversamples, and final counts compiled during the execution of the dataset preprocessing pipeline.

## Dataset Split Final Counts
| Split | Original Count | Post-Cleaning Count | Post-Balancing Count (Physical YOLO Files) | Total Label Boxes Exported |
| --- | --- | --- | --- | --- |
| **Train** | {report_data['cleaner_exclusions']['original_train']} | {report_data['train_split_before_stats']['total_images']} | {report_data['final_counts'].get('train', {}).get('images', 0)} | {report_data['final_counts'].get('train', {}).get('labels', 0)} |
| **Validation** | {report_data['cleaner_exclusions']['original_val']} | N/A | {report_data['final_counts'].get('val', {}).get('images', 0)} | {report_data['final_counts'].get('val', {}).get('labels', 0)} |
| **Test** | {report_data['cleaner_exclusions']['original_test']} | N/A | {report_data['final_counts'].get('test', {}).get('images', 0)} | {report_data['final_counts'].get('test', {}).get('labels', 0)} |

## Data Cleaning & Filtration Metrics
- **Corrupted Images Removed**: {report_data['cleaner_exclusions']['corrupted']}
- **Blurry Images Removed**: {report_data['cleaner_exclusions']['blurry']} (Laplacian variance < {report_data['train_split_before_stats'].get('avg_blur_score', 0):.2f})
- **Duplicates Removed (Within-split)**: {report_data['cleaner_exclusions']['duplicate']}
- **Empty Label Images Removed**: {report_data['cleaner_exclusions']['empty']}
- **Logo Transitions/Replays Discarded**: {report_data['cleaner_exclusions']['logo_replay']}

## Label Validation and Auto-fixes
- **Total Boxes Checked**: {val_report.get('total_boxes_checked', 0)}
- **Boxes Clipped to Image Boundaries**: {val_report.get('total_boxes_clipped', 0)}
- **Zero-Area / Degenerate Boxes Dropped**: {val_report.get('total_boxes_dropped', 0)}

{fixes_list_text}

## Class Imbalance and Oversampling
Prior to balancing, the train split class instances were highly imbalanced:
- **Ball Instances**: {report_data['train_split_before_stats']['class_distribution'].get('ball', 0)} (Average pixel area: {report_data['train_split_before_stats'].get('avg_ball_area_pixels', 0):.2f} px)
- **Player Instances**: {report_data['train_split_before_stats']['class_distribution'].get('player', 0)}
- **Goalkeeper Instances**: {report_data['train_split_before_stats']['class_distribution'].get('goalkeeper', 0)}
- **Referee Instances**: {report_data['train_split_before_stats']['class_distribution'].get('referee', 0)}

After oversampling the ball class (factor: {report_data['final_counts'].get('train', {}).get('images', 0) - report_data['train_split_before_stats']['total_images']} replica copies added with distinct deterministic augmentations), the final train class distribution is:
- **Ball Instances**: {report_data['train_split_after_stats']['class_distribution'].get('ball', 0)}
- **Player Instances**: {report_data['train_split_after_stats']['class_distribution'].get('player', 0)}
- **Goalkeeper Instances**: {report_data['train_split_after_stats']['class_distribution'].get('goalkeeper', 0)}
- **Referee Instances**: {report_data['train_split_after_stats']['class_distribution'].get('referee', 0)}

### Computed Canonical Inverse-Frequency Class Weights
These weights have been exported for optional training-time loss-weighting if oversampling is disabled:
{class_weights_text}

## Cross-Split Data Leakage Verification
{leak_text}
"""
    with open(summary_path, "w") as f:
        f.write(content)
    logger.info(f"Saved markdown summary to: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end Preprocessing Pipeline for Football player and ball detection")
    parser.add_argument("--config", type=str, default="configs/preprocessing_config.yaml", help="Path to preprocessing configuration YAML file")
    args = parser.parse_args()
    
    # Run the orchestrator
    run_preprocessing_pipeline(args.config)
