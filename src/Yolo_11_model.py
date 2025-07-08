#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ultralytics import YOLO
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

class YOLOTester:
    def __init__(self, model_path):
        """
        Inicializa el probador de YOLO con el modelo entrenado y los recursos necesarios.
        """
        self.model = YOLO(model_path)
        print(f"Modelo cargado desde: {model_path}")

        # MAPEADO DE RECOMENDACIONES Y COLORES
        self.recommendations = {
            "paper": "Deposita el papel limpio y seco en el contenedor azul. Evita reciclar papel sucio o con grasa.",
            "cardboard": "Dobla el cartón para ahorrar espacio y colócalo en el contenedor azul. No debe tener restos de comida.",
            "plastic": "Enjuaga los envases plásticos y colócalos en el contenedor amarillo. No mezcles con residuos orgánicos.",
            "metal": "Latas y envases metálicos deben ir limpios al contenedor gris. Evita desechar objetos con restos de alimentos.",
            "glass": "Bota botellas y frascos de vidrio en el contenedor verde. No incluyas vidrios rotos, espejos o cerámica.",
            "Compostable": "Deposita residuos orgánicos como restos de comida o cáscaras en el contenedor café."
        }

        # Colores BGR para las Bbox
        self.bin_colors = {
            "paper": (255, 128, 0),      # Azul
            "cardboard": (255, 128, 0),  # Azul
            "plastic": (0, 255, 255),    # Amarillo
            "metal": (128, 128, 128),    # Gris
            "glass": (0, 128, 0),        # Verde
            "Compostable": (42, 42, 165) # Café
        }

        # CARGA DE RUTAS DE IMÁGENES DE TACHOS
        self.bin_images_paths = {
            "paper": "img/tachos_azul.png",
            "cardboard": "img/tachos_azul.png",
            "plastic": "img/tachos_amarillo.png",
            "metal": "img/tachos_gris.png",
            "glass": "img/tachos_verde.png",
            "Compostable": "img/tachos_cafe.png",
            "default": "img/tachos_negro.png" # Imagen por defecto
        }
        
        self.loaded_bin_images = {}
        for cls, path in self.bin_images_paths.items():
            if os.path.exists(path):
                self.loaded_bin_images.setdefault(cls, cv2.imread(path))
            else:
                print(f"Advertencia: No se encontró la imagen del tacho en '{path}'.")
                placeholder = np.full((100, 100, 3), (200, 200, 200), dtype=np.uint8)
                cv2.putText(placeholder, "?", (35, 65), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)
                self.loaded_bin_images.setdefault(cls, placeholder)

        self.default_recommendation = "Apunta la cámara a un objeto para identificarlo y recibir una recomendación de reciclaje."

        # Cargar fuentes que soporten tildes
        try:
            # Fuente para el título (más grande y en negrita si es posible)
            self.font_title = ImageFont.truetype("arialbd.ttf", 28)
        except IOError:
            print("Advertencia: Fuente 'arialbd.ttf' no encontrada para el título. Usando Arial normal.")
            try:
                self.font_title = ImageFont.truetype("arial.ttf", 28)
            except IOError:
                self.font_title = ImageFont.load_default()
        
        try:
            # Fuente para el texto de recomendación
            self.font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            print("Advertencia: Fuente 'arial.ttf' no encontrada. Usando fuente por defecto.")
            self.font = ImageFont.load_default()

    def _draw_text_with_pil(self, image, text, position, font, fill):
        """Dibuja texto en una imagen de OpenCV usando Pillow para soportar UTF-8."""
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text(position, text, font=font, fill=fill)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _draw_custom_frame(self, frame, results):
        """
        Dibuja las detecciones, el panel de recomendación y la bbox personalizada en un frame.
        """
        panel_width = 400
        height, width, _ = frame.shape
        panel = np.full((height, panel_width, 3), 255, dtype=np.uint8)
        annotated_frame = frame.copy()
        found_detection = False
        
        boxes = results and len(results) > 0 and results[0].boxes
        if boxes:
            found_detection = True
            top_detection = None
            highest_conf = 0.0

            for box in boxes:
                conf = box.conf[0].cpu().numpy()
                if conf > highest_conf:
                    highest_conf = conf
                    top_detection = box
            
            for box in boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[cls]
                color = self.bin_colors.get(class_name, (0, 0, 255))
                cv2.rectangle(annotated_frame, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)
                label = f"{class_name}: {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, (coords[0], coords[1] - h - 10), (coords[0] + w, coords[1]), color, -1)
                cv2.putText(annotated_frame, label, (coords[0], coords[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if top_detection is not None:
                cls = int(top_detection.cls[0].cpu().numpy())
                class_name = self.model.names[cls]
                
                # --- DIBUJAR TÍTULO ---
                title_text = class_name.capitalize()
                title_width = self.font_title.getlength(title_text)
                title_x = (panel_width - title_width) // 2
                title_y = 40
                panel = self._draw_text_with_pil(panel, title_text, (title_x, title_y), self.font_title, (0, 0, 0))
                
                image_y_start = title_y + self.font_title.getbbox(title_text)[3] + 20

                bin_image = self.loaded_bin_images.get(class_name)
                if bin_image is not None:
                    # --- CÁLCULO DE TAMAÑO DE IMAGEN CON RESTRICCIÓN DE ALTURA ---
                    bin_h, bin_w, _ = bin_image.shape
                    scale = (panel_width - 40) / bin_w
                    new_w, new_h = int(bin_w * scale), int(bin_h * scale)

                    # **LA CORRECCIÓN ESTÁ AQUÍ**
                    max_image_height = height - image_y_start - 100 # Espacio para texto debajo
                    if new_h > max_image_height:
                        scale = max_image_height / bin_h
                        new_w, new_h = int(bin_w * scale), int(bin_h * scale)
                    # **FIN DE LA CORRECCIÓN**

                    resized_bin = cv2.resize(bin_image, (new_w, new_h))
                    x_offset = (panel_width - new_w) // 2
                    panel[image_y_start:image_y_start + new_h, x_offset:x_offset + new_w] = resized_bin
                    last_y = image_y_start + new_h + 20
                else:
                    last_y = image_y_start
                
                recommendation_text = self.recommendations.get(class_name, "No hay recomendación disponible.")
                y = last_y
                words = recommendation_text.split(' ')
                line = ''
                for word in words:
                    test_line = f"{line} {word}".strip()
                    line_width = self.font.getlength(test_line)
                    if line_width > panel_width - 20:
                        panel = self._draw_text_with_pil(panel, line, (10, y), self.font, (0, 0, 0))
                        y += self.font.getbbox(" ")[3] + 10
                        line = word
                    else:
                        line = test_line
                panel = self._draw_text_with_pil(panel, line, (10, y), self.font, (0, 0, 0))

        if not found_detection:
            title_text = "Esperando Detección"
            title_width = self.font_title.getlength(title_text)
            title_x = (panel_width - title_width) // 2
            title_y = 40
            panel = self._draw_text_with_pil(panel, title_text, (title_x, title_y), self.font_title, (100, 100, 100))
            
            image_y_start = title_y + self.font_title.getbbox(title_text)[3] + 20

            default_bin_image = self.loaded_bin_images.get("default")
            if default_bin_image is not None:
                 # --- CÁLCULO DE TAMAÑO DE IMAGEN CON RESTRICCIÓN DE ALTURA (CASO POR DEFECTO) ---
                bin_h, bin_w, _ = default_bin_image.shape
                scale = (panel_width - 40) / bin_w
                new_w, new_h = int(bin_w * scale), int(bin_h * scale)
                
                # **LA CORRECCIÓN ESTÁ AQUÍ TAMBIÉN**
                max_image_height = height - image_y_start - 100 
                if new_h > max_image_height:
                    scale = max_image_height / bin_h
                    new_w, new_h = int(bin_w * scale), int(bin_h * scale)
                # **FIN DE LA CORRECCIÓN**

                resized_bin = cv2.resize(default_bin_image, (new_w, new_h))
                x_offset = (panel_width - new_w) // 2
                panel[image_y_start:image_y_start + new_h, x_offset:x_offset + new_w] = resized_bin
                last_y = image_y_start + new_h + 20
            else:
                 last_y = image_y_start
            
            y = last_y
            words = self.default_recommendation.split(' ')
            line = ''
            for word in words:
                test_line = f"{line} {word}".strip()
                line_width = self.font.getlength(test_line)
                if line_width > panel_width - 20:
                    panel = self._draw_text_with_pil(panel, line, (10, y), self.font, (0, 0, 0))
                    y += self.font.getbbox(" ")[3] + 10
                    line = word
                else:
                    line = test_line
            panel = self._draw_text_with_pil(panel, line, (10, y), self.font, (0, 0, 0))

        final_frame = cv2.hconcat([panel, annotated_frame])
        return final_frame

    def test_webcam(self, conf_threshold=0.5):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: No se pudo abrir la webcam")
            return
        print("Probando con webcam. Presiona 'q' para salir.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame, conf=conf_threshold, verbose=False)
            final_frame = self._draw_custom_frame(frame, results)
            cv2.imshow('YOLO Webcam con Recomendaciones', final_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# Ejemplo de uso
if __name__ == "__main__":
    def find_model():
        possible_model_paths = [
            '../runs/Entrenamiento_yolov11_new/train13/weights/best.pt',
            'runs/Entrenamiento_yolov11_new/train13/weights/best.pt',
            'best.pt'
        ]
        for path in possible_model_paths:
            if os.path.exists(path):
                print(f"Modelo encontrado en: {os.path.abspath(path)}")
                return path
        print("Error: No se encontró el modelo 'best.pt' en las rutas comunes.")
        return None

    MODEL_PATH = find_model()
    if MODEL_PATH:
        tester = YOLOTester(MODEL_PATH)
        tester.test_webcam(conf_threshold=0.4)
    else:
        print("\nEjecución detenida. Por favor, asegúrate de que la ruta al modelo (.pt) es correcta.")