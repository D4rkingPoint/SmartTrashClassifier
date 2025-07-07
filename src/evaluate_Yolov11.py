import os
import json
import cv2
import torch
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

# Configuración
DATASET_PATH = "dataset/dataset-para-proyecto-vision-Efficientdet/valid"  # Ajusta a tu path
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, "_annotations.coco.json")
IMAGES_DIR = DATASET_PATH
WEIGHTS_PATH = "runs/Entrenamiento_yolov11_new/train13/weights/best.pt"  # Ajusta aquí

SAVE_DIR = "evaluate_yolov11"
os.makedirs(SAVE_DIR, exist_ok=True)

# Clases (asegúrate que coinciden con tus IDs COCO)
CLASSES = ['metal', 'paper', 'plastic', 'glass', 'cardboard', 'compostable']

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(WEIGHTS_PATH)

# Cargar COCO ground truth
coco = COCO(ANNOTATIONS_PATH)

results = []

for img_id in tqdm(coco.getImgIds(), desc="Inferencia YOLOv11"):
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(IMAGES_DIR, img_info['file_name'])
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: no se pudo leer {img_path}")
        continue
    
    # Inferencia (ajusta conf y iou si quieres)
    preds = model(img)[0]
    
    boxes = preds.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
    scores = preds.boxes.conf.cpu().numpy()
    labels = preds.boxes.cls.cpu().numpy().astype(int)
    
    for box, score, label in zip(boxes, scores, labels):
        if score < 0.2:
            continue
        x1, y1, x2, y2 = box
        bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]  # COCO bbox format
        results.append({
            "image_id": img_id,
            "category_id": int(label) + 1,  # IDs COCO empiezan en 1
            "bbox": bbox,
            "score": float(score)
        })

# Guardar resultados JSON para COCOeval
pred_path = os.path.join(SAVE_DIR, "yolov11_predictions.json")
with open(pred_path, "w") as f:
    json.dump(results, f)

# Evaluación COCO
coco_dt = coco.loadRes(pred_path)
coco_eval = COCOeval(coco, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

ap50 = round(coco_eval.stats[1] * 100, 2)  # mAP@0.5
recall = round(coco_eval.stats[8] * 100, 2)  # AR@100
precision = ap50  # Aproximamos precisión como mAP@0.5
f1 = round(2 * precision * recall / (precision + recall + 1e-8), 2)

df = pd.DataFrame([["YOLOv11", ap50, precision, recall, f1]],
                  columns=["Modelo", "mAP@0.5 (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"])
csv_path = os.path.join(SAVE_DIR, "metrics.csv")
df.to_csv(csv_path, index=False)

print(f"Métricas guardadas en {csv_path}")
print(df)
