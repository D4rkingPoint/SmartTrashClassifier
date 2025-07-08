import os
import json
import cv2
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO
import matplotlib.pyplot as plt # <--- Importar
import seaborn as sns           # <--- Importar

# --- Configuración (sin cambios) ---
DATASET_PATH = "dataset/dataset-para-proyecto-vision-Efficientdet/valid"
ANNOTATIONS_PATH = os.path.join(DATASET_PATH, "_annotations.coco.json")
IMAGES_DIR = DATASET_PATH
WEIGHTS_PATH = "runs/Entrenamiento_yolov11_new/train14/weights/best.pt"

SAVE_DIR = "evaluate_yolov11_final"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(WEIGHTS_PATH)

# --- Cargar datos y realizar inferencia (sin cambios) ---
coco = COCO(ANNOTATIONS_PATH)
results = []
pred_path = os.path.join(SAVE_DIR, "yolov11_predictions_dataset_final.json")

if not os.path.exists(pred_path):
    for img_id in tqdm(coco.getImgIds(), desc="Inferencia YOLOv11"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(IMAGES_DIR, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: no se pudo leer {img_path}")
            continue
        
        preds = model(img, conf=0.2, iou=0.5)[0]
        
        boxes = preds.boxes.xyxy.cpu().numpy()
        scores = preds.boxes.conf.cpu().numpy()
        labels = preds.boxes.cls.cpu().numpy().astype(int)
        
        model_names = model.names
        coco_cat_ids = {cat_info['name']: cid for cid, cat_info in coco.cats.items()}
        
        for box, score, label_id in zip(boxes, scores, labels):
            class_name = model_names[label_id]
            if class_name in coco_cat_ids:
                coco_id = coco_cat_ids[class_name]
            else:
                continue
            
            x1, y1, x2, y2 = box
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            results.append({
                "image_id": img_id,
                "category_id": coco_id,
                "bbox": bbox,
                "score": float(score)
            })

    with open(pred_path, "w") as f:
        json.dump(results, f)
else:
    print(f"Cargando predicciones existentes desde {pred_path}")


# --- Evaluación COCO (sin cambios) ---
coco_dt = coco.loadRes(pred_path)
coco_eval = COCOeval(coco, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()

print("\n--- RESUMEN GENERAL COCO ---")
coco_eval.summarize()

# --- Cálculo de Métricas por Clase (sin cambios) ---
print("\n--- MÉTRICAS POR CLASE (AP @ IoU=0.50) ---")

cat_ids = coco.getCatIds()
cat_id_to_name = {cat['id']: cat['name'] for cat in coco.dataset['categories']}
per_class_metrics = []

for cat_id in cat_ids:
    try:
        cat_index = coco_eval.params.catIds.index(cat_id)
    except ValueError:
        continue

    iou_idx = np.where(coco_eval.params.iouThrs == 0.5)[0][0]
    
    precision_scores = coco_eval.eval['precision'][iou_idx, :, cat_index, 0, -1]
    ap = np.mean(precision_scores[precision_scores > -1])
    ap = ap if not np.isnan(ap) else 0.0

    recall_scores = coco_eval.eval['recall'][iou_idx, cat_index, 0, -1]
    ar = recall_scores if recall_scores > -1 else 0.0
    
    f1 = 2 * (ap * ar) / (ap + ar + 1e-8)

    per_class_metrics.append({
        "Clase": cat_id_to_name[cat_id],
        "AP@0.50 (%)": round(ap * 100, 2),
        "Recall@0.50 (%)": round(ar * 100, 2),
        "F1-Score@0.50 (%)": round(f1 * 100, 2)
    })

per_class_metrics = [metric for metric in per_class_metrics if metric['Clase'] != 'Trash-JOLd']

# Crear y guardar el DataFrame con las métricas por clase
df_per_class = pd.DataFrame(per_class_metrics)
csv_path_per_class = os.path.join(SAVE_DIR, "metrics_per_class.csv")
df_per_class.to_csv(csv_path_per_class, index=False)

print(f"\nMétricas por clase guardadas en {csv_path_per_class}")
print(df_per_class.to_string())

# ### NUEVO: GENERAR Y GUARDAR GRÁFICO ###
print("\n--- GENERANDO GRÁFICO DE BARRAS ---")

# Ordenar el dataframe de forma decreciente por AP
df_sorted = df_per_class.sort_values(by="AP@0.50 (%)", ascending=False)

# Crear el gráfico
plt.figure(figsize=(12, 8))
ax = sns.barplot(x="Clase", y="AP@0.50 (%)", data=df_sorted, palette="viridis")

# Añadir título y etiquetas
ax.set_title("Rendimiento por Clase (AP @ 0.50 IoU)", fontsize=16, pad=20)
ax.set_xlabel("Clase", fontsize=12)
ax.set_ylabel("Average Precision (%)", fontsize=12)

# Rotar etiquetas del eje X para mejor legibilidad
plt.xticks(rotation=45, ha='right')

# Añadir el valor exacto sobre cada barra
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points')

# Ajustar layout y guardar la imagen
plt.tight_layout()
chart_path = os.path.join(SAVE_DIR, "metrics_per_class_chart.png")
plt.savefig(chart_path)

print(f"\nGráfico guardado exitosamente en: {chart_path}")

# Opcional: mostrar el gráfico en pantalla
# plt.show()