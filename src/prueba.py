import os
from roboflow import Roboflow
import cv2
import pytorch_lightning as pl
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset
import numpy as np
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from pytorch_lightning import Trainer
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader


HOME = os.getcwd()
DATASET_DIR = os.path.join(HOME, "dataset")
DATASET_NAME = "dataset-para-proyecto-vision-Efficientdet"
DATASET_PATH = os.path.join(DATASET_DIR, DATASET_NAME)

os.makedirs(DATASET_DIR, exist_ok=True)
os.chdir(DATASET_DIR)


if not os.path.exists(DATASET_PATH):
    rf = Roboflow(api_key="RdgAUhTbcOD8jUWNIy9A")
    project = rf.workspace("proyectos-qu6sq").project("clasificacion-de-resuidos")
    version = project.version(4)
    dataset = version.download("coco", location=DATASET_PATH)
    dataset_path = dataset.location
else:
    print(f"Dataset '{DATASET_NAME}' ya está disponible, no se descarga nuevamente.")
    dataset_path = DATASET_PATH

os.chdir(HOME)