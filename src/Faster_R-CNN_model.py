import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ================================
# CONFIGURACIÓN
# ================================
CLASSES = [ 'Metal', 'Plastic', 'Glass', 'Cardboard', 'Paper', 'Compostable']
num_classes = len(CLASSES)
image_size = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# TRANSFORMACIONES
# ================================
transform = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def preprocess(image):
    transformed = transform(image=image)
    return transformed["image"].unsqueeze(0).to(device)  # Shape (1, 3, H, W)

# ================================
# CARGA DEL MODELO
# ================================
def load_model(weights_path='model_fasterrcnn/fasterrcnn_weights.pth'):
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)  # +1 por background

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model

# ================================
# DIBUJAR PREDICCIONES
# ================================
def draw_predictions(frame, outputs, threshold=0.5):
    boxes = outputs['boxes']
    scores = outputs['scores']
    labels = outputs['labels']

    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold and 1 <= label <= len(CLASSES):
            x1, y1, x2, y2 = map(int, box.tolist())
            class_name = CLASSES[label - 1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# ================================
# LOOP PRINCIPAL
# ================================
# ================================
# LOOP PRINCIPAL (CORREGIDO)
# ================================
def run_camera(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo acceder a la cámara.")
        return

    print("✅ Cámara iniciada. Presiona 'q' para salir.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ No se pudo leer el frame.")
            break

        # Guardamos las dimensiones originales del frame
        orig_h, orig_w, _ = frame.shape
        
        # Preprocesamos la imagen para el modelo
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(rgb_frame)

        with torch.no_grad():
            outputs = model(input_tensor)[0]

        # --- AQUÍ ESTÁ LA CORRECCIÓN ---
        # Escalamos las coordenadas de las cajas al tamaño original del frame
        
        # Calculamos los factores de escala
        x_scale = orig_w / image_size
        y_scale = orig_h / image_size

        # Obtenemos las cajas y las escalamos
        boxes = outputs['boxes']
        boxes[:, [0, 2]] *= x_scale  # Escala las coordenadas X (x1, x2)
        boxes[:, [1, 3]] *= y_scale  # Escala las coordenadas Y (y1, y2)
        
        # Dibujamos las predicciones en el frame original usando las cajas ya escaladas
        result_frame = draw_predictions(frame, outputs, threshold=0.5)

        cv2.imshow("Faster R-CNN - Webcam", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ================================
# EJECUCIÓN
# ================================
if __name__ == "__main__":
    print("🔄 Cargando modelo Faster R-CNN...")
    model = load_model()
    print("✅ Modelo cargado correctamente.")
    run_camera(model)
