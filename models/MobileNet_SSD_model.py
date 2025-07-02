import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Configuración (debe coincidir con el entrenamiento)
HOME = os.getcwd()
MODEL_DIR = os.path.join(HOME, "model_mobilenet")
MODEL_PATH = os.path.join(MODEL_DIR, "ssd_mobilenet_final.keras")  # o "ssd_mobilenet_final.keras"
CLASSES = ['metal', 'paper', 'plastic', 'glass', 'cardboard', 'trash', 'compostable']
IMG_SIZE = 300
CONFIDENCE_THRESHOLD = 0.95 # Umbral de confianza para mostrar detecciones

# Cargar el modelo entrenado
print("Cargando modelo...")
model = load_model(MODEL_PATH, compile=False)

def preprocess_image(image_path):
    """Preprocesa la imagen para la red"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    
    orig_img = img.copy()
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalización
    return np.expand_dims(img, axis=0), orig_img

def decode_predictions(preds, confidence_thresh=CONFIDENCE_THRESHOLD):
    """Decodifica las predicciones del modelo SSD"""
    # preds shape: (1, TOTAL_PRIORS, 4 + num_classes + 1)
    boxes = preds[0, :, :4]  # Coordenadas normalizadas [0,1]
    class_probs = preds[0, :, 4:]  # Probabilidades de clase
    
    # Convertir a coordenadas absolutas
    boxes = boxes * IMG_SIZE
    
    # Obtener clase con mayor probabilidad para cada prior box
    class_ids = np.argmax(class_probs, axis=-1)
    confidences = np.max(class_probs, axis=-1)
    
    # Filtrar por confianza y rango válido de clases
    valid_classes_mask = (class_ids >= 0) & (class_ids < len(CLASSES))
    mask = (confidences > confidence_thresh) & valid_classes_mask
    
    boxes = boxes[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]
    
    # Aplicar NMS (Non-Maximum Suppression)
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), 
            confidences.tolist(), 
            confidence_thresh, 
            0.4  # Umbral NMS
        )
        
        if len(indices) > 0:
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()
            else:  # Para OpenCV < 4.5.2
                indices = [i[0] for i in indices]
            
            return boxes[indices], class_ids[indices], confidences[indices]
    return np.zeros((0, 4)), np.array([], dtype=int), np.array([])

def draw_detections(image, boxes, class_ids, confidences):
    """Dibuja las detecciones en la imagen"""
    for box, class_id, confidence in zip(boxes, class_ids, confidences):
        xmin, ymin, xmax, ymax = box.astype(int)
        
        # Color y etiqueta
        color = (0, 255, 0)  # Verde
        label = f"{CLASSES[class_id]}: {confidence:.2f}"
        
        # Dibujar bounding box y etiqueta
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(
            image, label, (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

def test_image(image_path):
    """Procesa una imagen y muestra los resultados"""
    try:
        # Preprocesar
        input_img, orig_img = preprocess_image(image_path)
        
        # Inferencia
        preds = model.predict(input_img)
        
        # Decodificar predicciones
        boxes, class_ids, confidences = decode_predictions(preds)
        
        # Redimensionar boxes a tamaño original
        h, w = orig_img.shape[:2]
        scale_x, scale_y = w / IMG_SIZE, h / IMG_SIZE
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y
        
        # Dibujar detecciones
        draw_detections(orig_img, boxes, class_ids, confidences)
        
        # Mostrar resultados
        cv2.imshow("Detections", orig_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Guardar imagen con detecciones
        output_path = os.path.join(HOME, "detection_result.jpg")
        cv2.imwrite(output_path, orig_img)
        print(f"Resultado guardado en: {output_path}")
        
    except Exception as e:
        print(f"Error procesando imagen: {e}")

def test_webcam():
    """Prueba el modelo con la webcam en tiempo real"""
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocesar frame
        input_img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        input_img = input_img / 255.0
        input_img = np.expand_dims(input_img, axis=0)
        
        # Inferencia
        preds = model.predict(input_img)
        
        # Decodificar predicciones
        boxes, class_ids, confidences = decode_predictions(preds)
        
        # Escalar boxes al tamaño del frame
        h, w = frame.shape[:2]
        scale_x, scale_y = w / IMG_SIZE, h / IMG_SIZE
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y
        
        # Dibujar detecciones
        draw_detections(frame, boxes, class_ids, confidences)
        
        # Mostrar frame
        cv2.imshow("Webcam - SSD MobileNetV2", frame)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ejemplo de uso con una imagen
    #test_image_path = os.path.join(HOME, "test_image.jpg")  # Cambia por tu imagen de prueba
    #if os.path.exists(test_image_path):
    #    test_image(test_image_path)
    #else:
        #print(f"No se encontró imagen de prueba en: {test_image_path}")
        #print("Probando con webcam...")
        #test_webcam()
    print("Probando con webcam...")
    test_webcam()