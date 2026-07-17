"""
Download dataset from Roboflow and save it to the data/raw/ directory.
"""

import os
import shutil
from roboflow import Roboflow

def download():
    # Target directory in raw data folder
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "football-players-detection"))
    
    # Initialize Roboflow
    print("Connecting to Roboflow...")
    rf = Roboflow(api_key="w6csk9iwbhiLrVptjv4U")
    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    version = project.version(1)
    
    # Download dataset in COCO JSON format
    print("Downloading dataset in COCO format...")
    dataset = version.download("coco")
    
    # Check if download was successful and locate the downloaded folder
    downloaded_path = dataset.location
    print(f"Dataset downloaded to temporary path: {downloaded_path}")
    
    if os.path.exists(downloaded_path):
        # Ensure raw folder exists
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        
        # If the destination already exists, remove it first
        if os.path.exists(target_dir):
            print(f"Removing existing directory: {target_dir}")
            shutil.rmtree(target_dir)
            
        # Move the downloaded folder to data/raw/football-players-detection
        print(f"Moving dataset to final location: {target_dir}")
        shutil.move(downloaded_path, target_dir)
        print("Dataset download and setup complete!")
    else:
        print("[ERROR] Download directory not found.")

if __name__ == "__main__":
    download()
