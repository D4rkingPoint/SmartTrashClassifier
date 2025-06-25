from ultralytics import YOLO
import os

# Crear carpeta datasets
HOME = os.getcwd()
DATASET_DIR = f"{HOME}/dataset"
DATASET_NAME = "dataset-para-proyecto-vision-yolo8"
DATASET_PATH = os.path.join(DATASET_DIR, DATASET_NAME)

os.makedirs(DATASET_DIR, exist_ok=True)
os.chdir(DATASET_DIR)

# Verificar si ya existe el dataset
if not os.path.exists(DATASET_PATH):
    from roboflow import Roboflow
    rf = Roboflow(api_key="RdgAUhTbcOD8jUWNIy9A")
    project = rf.workspace("proyectos-qu6sq").project("dataset-para-proyecto-vision")
    version = project.version(2)
    dataset = version.download("yolov8", location=DATASET_PATH)                             
    dataset_path = dataset.location  # <-- Definimos la ruta cuando se descarga
else:
    print(f"El dataset '{DATASET_NAME}' ya está disponible, no se descarga nuevamente.")
    dataset_path = DATASET_PATH  # <-- Definimos la ruta manualmente

# Volver al directorio principal
os.chdir(HOME)

train_path = "runs"
# Entrenamiento usando subprocess y CLI

import subprocess

subprocess.run([
    "yolo",
    "task=detect",
    "mode=train",
    "model=yolov8s.pt",
    f"data={dataset_path}/data.yaml",
    "epochs=30",
    "imgsz=640",
    "device=0",
    "plots=True",
    f"project={train_path}/Entrenamiento_yolov8",  # Cambia esto por el nombre deseado
])
