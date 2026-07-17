"""
Training Module for Football Player/Ball Detectors.

This script wraps training logic for YOLO-family models (YOLO11, YOLO11m)
and RT-DETR. It handles hyperparameter loading, data preparation, transfer learning,
EMA checkpoints, and logs metrics to MLflow.
"""

import os
import argparse
import yaml
import torch
from loguru import logger
from ultralytics import YOLO

def train_model(config_path: str, epochs_override: int = None, batch_override: int = None, device_override: str = None):
    """
    Main training loop wrapper.
    
    Args:
        config_path (str): Path to the training configuration YAML file.
        epochs_override (int, optional): Optional epoch limit override for dry-runs.
        batch_override (int, optional): Optional batch size override.
        device_override (str, optional): Optional device override.
    """
    logger.info(f"Loading training config from: {config_path}")
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        train_config = yaml.safe_load(f)

    # 1. Apply CLI Overrides
    if epochs_override is not None:
        logger.info(f"Overriding epochs: {train_config.get('epochs')} -> {epochs_override}")
        train_config["epochs"] = epochs_override
    if batch_override is not None:
        logger.info(f"Overriding batch size: {train_config.get('batch')} -> {batch_override}")
        train_config["batch"] = batch_override
    if device_override is not None:
        logger.info(f"Overriding device: {train_config.get('device')} -> {device_override}")
        train_config["device"] = device_override

    # 2. Setup Environment Variables for MLflow
    mlflow_db_dir = "experiments/mlflow"
    os.makedirs(mlflow_db_dir, exist_ok=True)
    tracking_uri = f"sqlite:///{os.path.abspath(mlflow_db_dir)}/mlflow.db"
    
    logger.info(f"Configuring MLflow Tracking URI to: {tracking_uri}")
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Football-Player-Ball-Detection"
    
    # Generate a run name based on the model and config parameters
    model_name = os.path.basename(train_config.get("model", "yolo")).split('.')[0]
    run_name = f"{model_name}_baseline_epochs{train_config.get('epochs')}"
    os.environ["MLFLOW_RUN"] = run_name
    logger.info(f"MLflow Run Name configured: {run_name}")

    # 3. Dynamic Device Selection
    device = train_config.get("device", "")
    if device == "" or device is None:
        if torch.cuda.is_available():
            device = "0"
            logger.info("CUDA GPU detected. Setting training device to GPU 0.")
        else:
            device = "cpu"
            logger.info("No CUDA GPU detected. Falling back to CPU for training.")
    else:
        logger.info(f"Using pre-configured training device: {device}")
    train_config["device"] = device

    # 4. Load Model
    model_path = train_config.get("model")
    logger.info(f"Initializing YOLO model from checkpoint: {model_path}")
    
    # Ensure parent directory for local model caching exists
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        
    model = YOLO(model_path)

    # 5. Extract hyperparameters for training
    # Ultralytics train accept parameters directly as kwargs
    train_args = {
        "data": train_config.get("data"),
        "epochs": train_config.get("epochs", 50),
        "imgsz": train_config.get("imgsz", 640),
        "batch": train_config.get("batch", 16),
        "device": train_config.get("device"),
        "optimizer": train_config.get("optimizer", "AdamW"),
        "lr0": train_config.get("lr0", 0.01),
        "lrf": train_config.get("lrf", 0.01),
        "weight_decay": train_config.get("weight_decay", 0.0005),
        "seed": train_config.get("seed", 42),
        "project": train_config.get("project", "runs/detect"),
        "name": train_config.get("name", "yolo11m_baseline"),
        "save": train_config.get("save", True),
        "val": train_config.get("val", True),
        "cache": train_config.get("cache", False),
        "plots": train_config.get("plots", True),
        "workers": train_config.get("workers", 8),
        # Loss weights
        "box": train_config.get("box", 7.5),
        "cls": train_config.get("cls", 0.5),
        "dfl": train_config.get("dfl", 1.5),
        # Augmentation parameters
        "mosaic": train_config.get("mosaic", 1.0),
        "close_mosaic": train_config.get("close_mosaic", 10),
        "copy_paste": train_config.get("copy_paste", 0.0),
        "hsv_h": train_config.get("hsv_h", 0.015),
        "hsv_s": train_config.get("hsv_s", 0.7),
        "hsv_v": train_config.get("hsv_v", 0.4),
        "fliplr": train_config.get("fliplr", 0.5),
        "scale": train_config.get("scale", 0.5),
        "translate": train_config.get("translate", 0.1),
        "momentum": train_config.get("momentum", 0.937),
    }

    logger.info("Starting YOLO training pipeline with parameters:")
    for k, v in train_args.items():
        logger.info(f"  - {k}: {v}")

    # 6. Execute Training Loop
    try:
        results = model.train(**train_args)
        logger.info("Training pipeline completed successfully!")
        
        # Log location of best weights
        save_dir = getattr(results, "save_dir", None) or os.path.join(train_args["project"], train_args["name"])
        logger.info(f"Best model weights and charts saved under: {save_dir}")
        
    except Exception as e:
        logger.exception(f"An error occurred during model training: {e}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Football Detection Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs for verification run")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu or index)")
    
    args = parser.parse_args()
    train_model(
        config_path=args.config,
        epochs_override=args.epochs,
        batch_override=args.batch,
        device_override=args.device
    )
