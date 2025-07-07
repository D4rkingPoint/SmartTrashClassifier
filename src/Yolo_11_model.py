import cv2
from ultralytics import YOLO
import numpy as np
from PIL import ImageFont, ImageDraw, Image

CLASSES = ['metal', 'paper', 'plastic', 'glass', 'cardboard', 'compostable']

# Mapeo de color (BGR OpenCV) y nombre del contenedor
CLASS_COLORS = {
    'paper': ((255, 0, 0), "Azul"),
    'cardboard': ((255, 0, 0), "Azul"),
    'plastic': ((0, 255, 255), "Amarillo"),
    'metal': ((192, 192, 192), "Gris"),
    'glass': ((0, 255, 0), "Verde"),
    'trash': ((0, 0, 0), "Negro"),
    'compostable': ((42, 42, 165), "Café")
}

CLASS_INFO = {
    'paper': "Reciclar papel ahorra hasta 50 litros por kilo.",
    'cardboard': "Reciclar cartón evita talar árboles.",
    'plastic': "Cada kilo de plástico reciclado ahorra 100 litros de agua.",
    'metal': "Una lata tarda 200 años en degradarse.",
    'glass': "El vidrio puede reciclarse infinitamente.",
    'trash': "Este residuo no es reciclable.",
    'compostable': "Este residuo es biodegradable."
}

# Fuente personalizada (puedes cambiar por otra si tienes una en .ttf)
FONT_PATH = "arial.ttf"  # Asegúrate que esta fuente exista, o usa otra
FONT_SIZE = 18

model = YOLO('../runs/Entrenamiento_yolov11_new/train12/weights/best.pt') # ../runs/Entrenamiento_yolov11_new/train12/weights/best.pt
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detected_classes = set()

    for result in results.boxes:
        cls_id = int(result.cls[0])
        
        # --- Obtener la confianza ---
        conf = float(result.conf[0])

        # --- Obtener las coordenadas y encoger la caja (como antes) ---
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        shrink_factor = 0.30
        width = x2 - x1
        height = y2 - y1
        new_x1 = int(x1 + width * shrink_factor / 2)
        new_y1 = int(y1 + height * shrink_factor / 2)
        new_x2 = int(x2 - width * shrink_factor / 2)
        new_y2 = int(y2 - height * shrink_factor / 2)

        # --- Crear la etiqueta con la clase y la confianza ---
        label = CLASSES[cls_id]
        label_with_conf = f"{label} {conf:.2f}"  # Formato: "plastic 0.87"
        
        # --- Dibujar en el frame ---
        color_bgr, _ = CLASS_COLORS.get(label, ((255, 255, 255), ""))
        cv2.rectangle(frame, (new_x1, new_y1), (new_x2, new_y2), color_bgr, 2)
        
        # Usar la nueva etiqueta que incluye la confianza
        cv2.putText(frame, label_with_conf, (new_x1, new_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        detected_classes.add(label)

        # Convertir frame a PIL para dibujar texto
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil, mode="RGBA")
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    # Posiciones iniciales
    x_text, y_text = 10, 10
    line_height = 30

    # Calcular todas las líneas de texto
    text_lines = []
    for label in sorted(detected_classes):
        info = CLASS_INFO.get(label, "")
        _, container_color = CLASS_COLORS.get(label, ((255, 255, 255), "Desconocido"))
        text_lines.append(f"{label.capitalize()}: {info}")
        text_lines.append(f"Este material va en el contenedor: {container_color}")

    # Calcular ancho máximo del texto
    max_width = 0
    for line in text_lines:
        bbox = font.getbbox(line)
        text_width = bbox[2] - bbox[0]
        if text_width > max_width:
            max_width = text_width

    # Dibujar fondo dinámico
    padding = 10
    total_height = line_height * len(text_lines) + padding
    draw.rectangle(
        [x_text - padding, y_text - padding,
         x_text + max_width + padding, y_text + total_height - padding],
        fill=(0, 0, 0, 160)
    )

    # Dibujar los textos
    for line in text_lines:
        draw.text((x_text, y_text), line, font=font, fill=(255, 255, 255, 255))
        y_text += line_height

    # Convertir de nuevo a formato OpenCV
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    cv2.imshow("YOLOv11 Recycling Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
