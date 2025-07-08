from ultralytics import YOLO
import cv2
import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt

class YOLOTester:
    def __init__(self, model_path):
        """
        Inicializa el probador de YOLO con el modelo entrenado
        
        Args:
            model_path (str): Ruta al modelo entrenado (.pt)
        """
        self.model_path = model_path
        self.model = YOLO(model_path)
        print(f"Modelo cargado desde: {model_path}")
        
    def test_single_image(self, image_path, save_path=None, show_conf=True, conf_threshold=0.4):
        """
        Prueba el modelo en una sola imagen
        
        Args:
            image_path (str): Ruta a la imagen
            save_path (str): Ruta donde guardar la imagen con predicciones
            show_conf (bool): Mostrar confianza en las predicciones
            conf_threshold (float): Umbral de confianza mínimo
        """
        # Verificar que la imagen existe
        if not os.path.exists(image_path):
            print(f"Error: La imagen {image_path} no existe")
            print(f"Ruta absoluta buscada: {os.path.abspath(image_path)}")
            return None
        
        print(f"Procesando imagen: {image_path}")
        
        # Realizar predicción
        results = self.model(image_path, conf=conf_threshold)
        
        # Mostrar resultados
        for result in results:
            # Obtener información de las detecciones
            boxes = result.boxes
            if boxes is not None:
                print(f"\n--- Resultados para {image_path} ---")
                print(f"Detecciones encontradas: {len(boxes)}")
                
                for i, box in enumerate(boxes):
                    # Obtener coordenadas, confianza y clase
                    coords = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[cls]
                    
                    print(f"Detección {i+1}:")
                    print(f"  Clase: {class_name}")
                    print(f"  Confianza: {conf:.2f}")
                    print(f"  Coordenadas: x1={coords[0]:.1f}, y1={coords[1]:.1f}, x2={coords[2]:.1f}, y2={coords[3]:.1f}")
            
            # Guardar imagen con predicciones
            if save_path:
                result.save(save_path)
                print(f"Imagen guardada en: {save_path}")
            
            # Mostrar imagen
            result.show()
        
        return results
    
    def test_multiple_images(self, images_folder, output_folder=None, conf_threshold=0.5):
        """
        Prueba el modelo en múltiples imágenes
        
        Args:
            images_folder (str): Carpeta con imágenes
            output_folder (str): Carpeta para guardar resultados
            conf_threshold (float): Umbral de confianza
        """
        # Verificar que la carpeta existe
        if not os.path.exists(images_folder):
            print(f"Error: La carpeta {images_folder} no existe")
            print(f"Ruta absoluta buscada: {os.path.abspath(images_folder)}")
            return None
        
        # Extensiones de imagen soportadas
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(images_folder, ext)))
            image_paths.extend(glob.glob(os.path.join(images_folder, ext.upper())))
        
        print(f"Buscando en: {os.path.abspath(images_folder)}")
        print(f"Archivos encontrados: {os.listdir(images_folder) if os.path.exists(images_folder) else 'Carpeta no existe'}")
        
        if not image_paths:
            print(f"No se encontraron imágenes en {images_folder}")
            print("Extensiones buscadas:", image_extensions)
            return None
        
        print(f"Encontradas {len(image_paths)} imágenes")
        
        # Crear carpeta de salida si se especifica
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
        
        results_summary = []
        
        for i, image_path in enumerate(image_paths):
            print(f"\nProcesando imagen {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            # Realizar predicción
            results = self.model(image_path, conf=conf_threshold)
            
            for result in results:
                boxes = result.boxes
                detections_count = len(boxes) if boxes is not None else 0
                
                results_summary.append({
                    'image': os.path.basename(image_path),
                    'detections': detections_count
                })
                
                # Guardar resultado si se especifica carpeta de salida
                if output_folder:
                    save_path = os.path.join(output_folder, f"pred_{os.path.basename(image_path)}")
                    result.save(save_path)
        
        # Mostrar resumen
        print("\n--- RESUMEN DE RESULTADOS ---")
        total_detections = sum(item['detections'] for item in results_summary)
        print(f"Total de imágenes procesadas: {len(results_summary)}")
        print(f"Total de detecciones: {total_detections}")
        print(f"Promedio de detecciones por imagen: {total_detections/len(results_summary):.2f}")
        
        return results_summary
    
    def test_video(self, video_path, output_path=None, conf_threshold=0.5):
        """
        Prueba el modelo en un video
        
        Args:
            video_path (str): Ruta al video
            output_path (str): Ruta para guardar video con predicciones
            conf_threshold (float): Umbral de confianza
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: No se pudo abrir el video {video_path}")
            return
        
        # Obtener propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {video_path}")
        print(f"FPS: {fps}, Resolución: {width}x{height}, Frames: {total_frames}")
        
        # Configurar escritor de video si se especifica salida
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detections_per_frame = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Realizar predicción
            results = self.model(frame, conf=conf_threshold)
            
            # Procesar resultados
            for result in results:
                boxes = result.boxes
                detections_count = len(boxes) if boxes is not None else 0
                detections_per_frame.append(detections_count)
                
                # Dibujar predicciones en el frame
                annotated_frame = result.plot()
                
                # Guardar frame si se especifica salida
                if output_path:
                    out.write(annotated_frame)
                
                # Mostrar frame (opcional - comentar si no quieres ver en tiempo real)
                # cv2.imshow('YOLO Detection', annotated_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            
            # Mostrar progreso
            if frame_count % 30 == 0:
                print(f"Procesados {frame_count}/{total_frames} frames")
        
        # Limpiar recursos
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        # Mostrar estadísticas
        print(f"\n--- ESTADÍSTICAS DEL VIDEO ---")
        print(f"Frames procesados: {frame_count}")
        print(f"Total detecciones: {sum(detections_per_frame)}")
        print(f"Promedio detecciones por frame: {np.mean(detections_per_frame):.2f}")
        if output_path:
            print(f"Video guardado en: {output_path}")
        
        return detections_per_frame
    
    def test_webcam(self, conf_threshold=0.5):
        """
        Prueba el modelo en tiempo real con la webcam
        
        Args:
            conf_threshold (float): Umbral de confianza
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: No se pudo abrir la webcam")
            return
        
        print("Probando con webcam. Presiona 'q' para salir.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Realizar predicción
            results = self.model(frame, conf=conf_threshold)
            
            # Mostrar resultados
            for result in results:
                annotated_frame = result.plot()
                cv2.imshow('YOLO Webcam Detection', annotated_frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def evaluate_model(self, test_dataset_path):
        """
        Evalúa el modelo en un dataset de prueba
        
        Args:
            test_dataset_path (str): Ruta al dataset de prueba
        """
        print("Evaluando modelo...")
        metrics = self.model.val(data=test_dataset_path)
        
        print(f"\n--- MÉTRICAS DE EVALUACIÓN ---")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        
        return metrics

# Ejemplo de uso
if __name__ == "__main__":
    # Función para encontrar el modelo automáticamente
    
    def find_model():
        possible_model_paths = [
            '../runs/Entrenamiento_yolov11_new/train14/weights/best.pt',
            '../../runs/Entrenamiento_yolov11_new/train14/weights/best.pt',
            './runs/Entrenamiento_yolov11_new/train14/weights/best.pt',
            'runs/Entrenamiento_yolov11_new/train14/weights/best.pt',
            '../../../runs/Entrenamiento_yolov11_new/train14/weights/best.pt',
            r'C:\Users\alvar\Desktop\Computer_vision_proyect\runs\Entrenamiento_yolov11_new\train14\weights\best.pt'
        ]
        
        for path in possible_model_paths:
            if os.path.exists(path):
                print(f"Modelo encontrado en: {path}")
                return path
        
        print("No se encontró el modelo en las rutas comunes.")
        print("Rutas buscadas:")
        for path in possible_model_paths:
            print(f"  - {path} -> {os.path.abspath(path)}")
        
        return None
    
    # Buscar el modelo
    MODEL_PATH = find_model()
    
    if MODEL_PATH is None:
        print("\nPor favor, especifica la ruta correcta del modelo manualmente.")
        print("Ejemplo: MODEL_PATH = r'C:\\ruta\\completa\\al\\modelo\\best.pt'")
        exit(1)
    
    # Crear instancia del probador
    tester = YOLOTester(MODEL_PATH)
    
    '''
    # Función para encontrar imágenes disponibles
    def find_available_images():
        possible_folders = ['../image_test', './image_test', 'image_test', '../images', './images']
        for folder in possible_folders:
            if os.path.exists(folder):
                files = os.listdir(folder)
                image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
                if image_files:
                    print(f"Encontradas imágenes en: {folder}")
                    print(f"Archivos: {image_files}")
                    return folder, image_files
        return None, []
    
    # Buscar imágenes disponibles
    image_folder, available_images = find_available_images()
    
    if image_folder and available_images:
        print(f"\nUsando carpeta: {image_folder}")
        print(f"Primera imagen: {available_images[0]}")
        
        # 1. Probar con la primera imagen encontrada
        first_image_path = os.path.join(image_folder, available_images[0])
        print(f"\nProbando con: {first_image_path}")
        tester.test_single_image(first_image_path, save_path='resultado.jpg')
        
        # 2. Probar con todas las imágenes de la carpeta
        print(f"\nProbando con todas las imágenes de: {image_folder}")
        tester.test_multiple_images(image_folder, output_folder='resultados')
    else:
        print("No se encontraron imágenes en las carpetas comunes.")
        print("\nPuedes:")
        print("1. Crear una carpeta 'image_test' en el directorio actual")
        print("2. Poner algunas imágenes allí")
        print("3. O especificar la ruta manualmente en el código")
        print("\nEjemplo de uso manual:")
        print("tester.test_single_image(r'C:\\ruta\\completa\\a\\imagen.jpg', save_path='resultado.jpg')")
    '''
    # Otras opciones de prueba (comentadas):
    
    # 3. Probar con video
    # tester.test_video('ruta/a/tu/video.mp4', output_path='video_resultado.mp4')
    
    # 4. Probar con webcam
    tester.test_webcam()
    
    # 5. Evaluar modelo (necesitas la ruta al data.yaml de tu dataset)
    # tester.evaluate_model('ruta/al/dataset/data.yaml')
    
    print("\nProbador de YOLO listo.")