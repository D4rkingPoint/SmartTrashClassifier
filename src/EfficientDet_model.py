import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet

# ================================
# CONFIGURACIÓN GENERAL
# ================================
CLASSES = ['Trash', 'Metal', 'Plastic', 'Glass', 'Cardboard', 'Paper', 'Compostable']
num_classes = len(CLASSES)
image_size = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================
# PREPROCESAMIENTO DE LA IMAGEN
# ================================
def preprocess(image):
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    transformed = transform(image=image)
    return transformed["image"].unsqueeze(0)  # batch size 1

# ================================
# CARGA DEL MODELO ENTRENADO
# ================================
def load_model(weights_path='model_efficientDet/efficientdet_weights.pth'):
    config = get_efficientdet_config('tf_efficientdet_d0')
    config.num_classes = num_classes
    config.image_size = (image_size, image_size)

    # 1. Creamos la arquitectura base del modelo
    net = EfficientDet(config, pretrained_backbone=False)

    # 2. Cargamos los pesos directamente en la arquitectura base 'net'
    #    Estos pesos ya incluyen la 'class_net' entrenada.
    net.load_state_dict(torch.load(weights_path, map_location=device))
    
    # 3. ¡ELIMINAMOS ESTA LÍNEA! No reinicies la cabeza de clasificación.
    #net.class_net = HeadNet(config, num_outputs=num_classes)  #<-- ESTA LÍNEA SE VA
    
    # 4. Envolvemos el 'net' ya cargado con la envoltura de predicción
    model = DetBenchPredict(net)
    model.eval().to(device)
    
    return model

# ================================
# DIBUJAR CAJAS Y ETIQUETAS
# ================================
def draw_predictions(frame, boxes, scores, labels, threshold=0.3):
    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold and 0 <= label < len(CLASSES):  # ✅ validación segura
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            text = f"{CLASSES[label]}: {score:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,0), 2)
    return frame


# ================================
# MAIN LOOP DE LA CÁMARA
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

        orig = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(rgb).to(device)

        with torch.no_grad():
            preds = model(input_tensor)[0]

            preds = preds.detach().cpu().numpy()

            if preds.shape[0] > 0:
                # Escalar a la resolución original
                h_orig, w_orig = orig.shape[:2]
                scale_x = w_orig / image_size
                scale_y = h_orig / image_size
                preds[:, 0] *= scale_x  # x1
                preds[:, 2] *= scale_x  # x2
                preds[:, 1] *= scale_y  # y1
                preds[:, 3] *= scale_y  # y2

                boxes = preds[:, :4]
                scores = preds[:, 4]
                labels = preds[:, 5].astype(int)
                #print("⚠️ Labels detectados:", labels)

            else:
                boxes = np.array([])
                scores = np.array([])
                labels = np.array([])


        result_frame = draw_predictions(orig, boxes, scores, labels)

        cv2.imshow("EfficientDet - Webcam", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ================================
# EJECUCIÓN PRINCIPAL
# ================================
if __name__ == "__main__":
    print("🔄 Cargando modelo EfficientDet...")
    model = load_model()
    print("✅ Modelo cargado correctamente.")
    run_camera(model)
