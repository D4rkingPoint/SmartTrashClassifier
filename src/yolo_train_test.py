from ultralytics import YOLO
import os
from multiprocessing import freeze_support # Good practice to include

# You can define functions or classes here, outside the main block.

def main():
    """Contains the main logic of your script."""
    # --- (Your existing dataset and training code goes here) ---
    HOME = os.getcwd()
    DATASET_DIR = os.path.join(HOME, "dataset")
    DATASET_NAME = "dataset-para-proyecto-vision-yolo11"
    DATASET_PATH = os.path.join(DATASET_DIR, DATASET_NAME)
    data_yaml_path = os.path.join(DATASET_PATH, 'data.yaml')

    # ... (Your logic for downloading the dataset if needed) ...

    # 1. Load the model
    model = YOLO('yolo11s.pt')

    # 2. Train the model
    # The line that was causing the crash
    results = model.train(
        data=data_yaml_path,
        epochs=30,
        imgsz=640,
        device=0,
        batch=32,
        plots=True,
        project="runs/Entrenamiento_yolov11_new",
    )

# This block ensures the code is only run when the script is executed directly
if __name__ == '__main__':
    # On Windows, freeze_support() is needed for creating executables,
    # and it's a good practice to include it for multiprocessing.
    freeze_support()
    
    # Run your main function
    main()