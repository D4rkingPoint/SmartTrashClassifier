# evaluate_fasterrcnn.py
import os
import json
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==== CONFIGURACION ====
CLASSES = ['Compostable',  'cardboard', 'glass', 'metal', 'paper', 'plastic']

NUM_CLASSES = len(CLASSES) + 1  # +1 para background
IMG_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_PATH = "dataset/dataset-para-proyecto-vision-faster-rcnn/valid"
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, "_annotations.coco.json")
IMAGES_DIR = DATASET_PATH
WEIGHTS_PATH = "model_fasterrcnn/fasterrcnn_weights.pth"
SAVE_DIR = "evaluate_fasterrcnn"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== TRANSFORMACION ====
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ==== CARGAR MODELO ====
def load_model():
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# ==== GENERAR PREDICCIONES COCO ====
def evaluate():
    model = load_model()
    coco = COCO(ANNOTATIONS_PATH)
    results = []

    for img_id in tqdm(coco.getImgIds(), desc="Evaluando Faster R-CNN"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(IMAGES_DIR, img_info['file_name'])
        image = cv2.imread(img_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]

        input_tensor = transform(image=rgb)['image'].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)[0]

        boxes = outputs['boxes'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score < 0.2:
                continue
            x1, y1, x2, y2 = box
            coco_box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            results.append({
                "image_id": img_id,
                "category_id": int(label),
                "bbox": coco_box,
                "score": float(score)
            })

    pred_path = os.path.join(SAVE_DIR, "fasterrcnn_predictions.json")
    with open(pred_path, "w") as f:
        json.dump(results, f)

    coco_dt = coco.loadRes(pred_path)
    coco_eval = COCOeval(coco, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Extraer métricas principales
    ap50 = round(coco_eval.stats[1] * 100, 2)  # mAP@0.5
    recall = round(coco_eval.stats[8] * 100, 2)  # AR@100
    precision = round(ap50, 2)  # Aproximamos precision como mAP@0.5
    f1 = round(2 * precision * recall / (precision + recall + 1e-8), 2)

    df = pd.DataFrame([
        ["Faster R-CNN", ap50, precision, recall, f1]
    ], columns=["Modelo", "mAP@0.5 (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"])
    df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"), index=False)
    print("\nResultados guardados en:", SAVE_DIR)
    print(df)

# ==== MAIN ====
if __name__ == "__main__":
    evaluate()
