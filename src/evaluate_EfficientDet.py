# evaluate_efficientdet.py
import os
import json
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==== CONFIGURACION ====
CLASSES = ['Trash-JOLd', 'Compostable', 'Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic']
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_PATH = "dataset/dataset-para-proyecto-vision-Efficientdet/valid"
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, "_annotations.coco.json")
IMAGES_DIR = DATASET_PATH
WEIGHTS_PATH = "model_efficientDet/efficientdet_weights.pth"
SAVE_DIR = "evaluate_efficientdet"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==== TRANSFORMACION ====
transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ==== CARGAR MODELO ====
def load_model():
    config = get_efficientdet_config('tf_efficientdet_d0')
    config.num_classes = NUM_CLASSES
    config.image_size = (IMG_SIZE, IMG_SIZE)
    net = EfficientDet(config, pretrained_backbone=False)
    net.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model = DetBenchPredict(net)
    model.eval().to(DEVICE)
    return model

# ==== GENERAR PREDICCIONES COCO ====
def evaluate():
    model = load_model()
    coco = COCO(ANNOTATIONS_PATH)
    results = []

    for img_id in tqdm(coco.getImgIds(), desc="Evaluando EfficientDet"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(IMAGES_DIR, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: no se pudo leer {img_path}")
            continue
        
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]

        input_tensor = transform(image=rgb)['image'].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            preds = model(input_tensor)[0]

        preds = preds.detach().cpu().numpy()

        # En EfficientDet preds: [x1, y1, x2, y2, score, label]
        if preds.shape[0] == 0:
            continue
        
        # Escalar coordenadas a tamaño original
        scale_x = orig_w / IMG_SIZE
        scale_y = orig_h / IMG_SIZE
        preds[:, 0] *= scale_x  # x1
        preds[:, 2] *= scale_x  # x2
        preds[:, 1] *= scale_y  # y1
        preds[:, 3] *= scale_y  # y2

        for pred in preds:
            x1, y1, x2, y2, score, label = pred
            if score < 0.2:
                continue
            if int(label) == 0:  # ignorar clase basura
                continue
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            results.append({
                "image_id": img_id,
                "category_id": int(label),
                "bbox": bbox,
                "score": float(score)
            })


    pred_path = os.path.join(SAVE_DIR, "efficientdet_predictions.json")
    with open(pred_path, "w") as f:
        json.dump(results, f)

    coco_dt = coco.loadRes(pred_path)
    coco_eval = COCOeval(coco, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    ap50 = round(coco_eval.stats[1] * 100, 2)  # mAP@0.5
    recall = round(coco_eval.stats[8] * 100, 2)  # AR@100
    precision = ap50  # aproximamos precisión como mAP@0.5
    f1 = round(2 * precision * recall / (precision + recall + 1e-8), 2)

    df = pd.DataFrame([
        ["EfficientDet", ap50, precision, recall, f1]
    ], columns=["Modelo", "mAP@0.5 (%)", "Precision (%)", "Recall (%)", "F1 Score (%)"])
    df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"), index=False)
    print("\nResultados guardados en:", SAVE_DIR)
    print(df)

# ==== MAIN ====
if __name__ == "__main__":
    evaluate()
