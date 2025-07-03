import os
import torch
import cv2
import numpy as np
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# ================================
# CONFIGURACIÓN
# ================================
CLASSES = ['Trash', 'Metal', 'Plastic', 'Glass', 'Cardboard', 'Paper', 'Compostable']
num_classes = len(CLASSES)
image_size = 512
model_dir = 'model_efficientDet'

# Colores para cada clase (BGR para OpenCV)
COLORS = {
    'Trash': (128, 128, 128),      # Gris
    'Metal': (192, 192, 192),      # Plata
    'Plastic': (0, 255, 255),      # Amarillo/Cyan
    'Glass': (255, 255, 0),        # Cyan
    'Cardboard': (0, 165, 255),    # Naranja
    'Paper': (255, 255, 255),      # Blanco
    'Compostable': (0, 255, 0)     # Verde
}

# ================================
# TRANSFORMACIONES PARA INFERENCIA
# ================================
transform = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ================================
# CARGAR MODELO
# ================================
def load_model(model_path, device):
    """Carga el modelo entrenado"""
    print("Cargando modelo...")
    
    # Configurar el modelo igual que en el entrenamiento
    config = get_efficientdet_config('tf_efficientdet_d0')
    config.num_classes = num_classes
    config.image_size = (image_size, image_size)
    
    # Crear la red igual que en entrenamiento
    net = EfficientDet(config, pretrained_backbone=False)
    net.class_net = HeadNet(config, num_outputs=num_classes)
    
    # Cargar los pesos
    checkpoint = torch.load(model_path, map_location=device)
    
    # Manejar diferentes formatos de checkpoint
    if 'state_dict' in checkpoint:
        # Es un checkpoint de Lightning
        state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            # Remover prefijos de Lightning
            if key.startswith('model.model.'):
                new_key = key.replace('model.model.', '')
            elif key.startswith('model.'):
                new_key = key.replace('model.', '')
            else:
                new_key = key
            state_dict[new_key] = value
        net.load_state_dict(state_dict)
    else:
        # Es un state_dict directo
        net.load_state_dict(checkpoint)
    
    # Crear DetBenchPredict después de cargar los pesos
    model = DetBenchPredict(net)
    
    model.to(device)
    model.eval()
    print("Modelo cargado exitosamente!")
    return model

# ================================
# FUNCIONES DE INFERENCIA
# ================================
def preprocess_image(image_path):
    """Preprocesa la imagen para inferencia"""
    # Leer imagen
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    
    # Convertir a RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image_rgb.shape[:2]
    
    # Aplicar transformaciones
    transformed = transform(image=image_rgb)
    tensor_image = transformed['image'].unsqueeze(0)  # Añadir dimensión de batch
    
    return tensor_image, image, original_shape

def load_model_alternative(model_dir, device):
    """Función alternativa para cargar el modelo probando diferentes archivos"""
    print("Probando cargar modelo con diferentes métodos...")
    
    # Lista de archivos posibles en orden de preferencia
    model_files = [
        'lightning_checkpoint.ckpt',
        'efficientdet_model.pth', 
        'efficientdet_weights.pth'
    ]
    
    config = get_efficientdet_config('tf_efficientdet_d0')
    config.num_classes = num_classes
    config.image_size = (image_size, image_size)
    
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_path):
            continue
            
        print(f"Intentando cargar: {model_file}")
        
        try:
            if model_file == 'lightning_checkpoint.ckpt':
                # Cargar checkpoint de Lightning completo
                from pytorch_lightning import LightningModule
                
                # Recrear la clase Lightning
                class EfficientDetLightningModel(LightningModule):
                    def __init__(self, model):
                        super().__init__()
                        self.model = model
                
                # Crear modelo base
                net = EfficientDet(config, pretrained_backbone=False)
                net.class_net = HeadNet(config, num_outputs=num_classes)
                from effdet import DetBenchTrain
                train_model = DetBenchTrain(net, config)
                
                # Cargar checkpoint
                pl_model = EfficientDetLightningModel.load_from_checkpoint(
                    model_path, 
                    model=train_model,
                    map_location=device
                )
                
                # Extraer el modelo para inferencia
                model = DetBenchPredict(pl_model.model.model)
                
            else:
                # Cargar pesos directamente
                net = EfficientDet(config, pretrained_backbone=False)
                net.class_net = HeadNet(config, num_outputs=num_classes)
                
                checkpoint = torch.load(model_path, map_location=device)
                
                if 'state_dict' in checkpoint:
                    # Procesar state_dict de Lightning
                    state_dict = {}
                    for key, value in checkpoint['state_dict'].items():
                        if key.startswith('model.model.'):
                            new_key = key.replace('model.model.', '')
                        elif key.startswith('model.'):
                            new_key = key.replace('model.', '')
                        else:
                            new_key = key
                        state_dict[new_key] = value
                    net.load_state_dict(state_dict)
                else:
                    net.load_state_dict(checkpoint)
                
                model = DetBenchPredict(net)
            
            model.to(device)
            model.eval()
            print(f"✅ Modelo cargado exitosamente desde: {model_file}")
            return model
            
        except Exception as e:
            print(f"❌ Error cargando {model_file}: {str(e)}")
            continue
    
    raise ValueError("No se pudo cargar ningún modelo. Verifica los archivos en la carpeta model_efficientDet/")

def postprocess_predictions(predictions, original_shape, confidence_threshold=0.5):
    """Postprocesa las predicciones del modelo"""
    
    # Debugging: Ver qué devuelve el modelo
    print(f"DEBUG: Tipo de predictions: {type(predictions)}")
    print(f"DEBUG: Shape de predictions: {predictions.shape if hasattr(predictions, 'shape') else 'No shape'}")
    print(f"DEBUG: Primeros valores: {predictions.flatten()[:10] if hasattr(predictions, 'flatten') else predictions}")
    
    detections = predictions[0]  # Primer elemento del batch
    print(f"DEBUG: Shape de detecciones: {detections.shape}")
    print(f"DEBUG: Número total de detecciones: {len(detections)}")
    
    # Extraer información
    boxes = detections[:, :4]  # [x1, y1, x2, y2]
    scores = detections[:, 4]  # Confianza
    labels = detections[:, 5].long()  # Etiquetas
    
    print(f"DEBUG: Scores máximo: {scores.max():.4f}, mínimo: {scores.min():.4f}")
    print(f"DEBUG: Scores > 0.1: {(scores > 0.1).sum()}")
    print(f"DEBUG: Scores > 0.05: {(scores > 0.05).sum()}")
    print(f"DEBUG: Scores > 0.01: {(scores > 0.01).sum()}")
    
    # Filtrar por confianza
    valid_detections = scores > confidence_threshold
    print(f"DEBUG: Detecciones válidas con threshold {confidence_threshold}: {valid_detections.sum()}")
    
    boxes = boxes[valid_detections]
    scores = scores[valid_detections]
    labels = labels[valid_detections]
    
    # Si no hay detecciones válidas, probar con threshold más bajo
    if len(boxes) == 0:
        print(f"WARNING: No hay detecciones con threshold {confidence_threshold}")
        print("Probando con threshold más bajo...")
        for low_threshold in [0.1, 0.05, 0.01]:
            valid_low = predictions[0][:, 4] > low_threshold
            if valid_low.sum() > 0:
                print(f"Con threshold {low_threshold}: {valid_low.sum()} detecciones")
                break
    
    # Escalar las cajas al tamaño original
    if len(boxes) > 0:
        scale_y = original_shape[0] / image_size
        scale_x = original_shape[1] / image_size
        
        boxes[:, [0, 2]] *= scale_x  # x1, x2
        boxes[:, [1, 3]] *= scale_y  # y1, y2
    
    return boxes, scores, labels
def postprocess_predictions(predictions, original_shape, confidence_threshold=0.5):
    """Postprocesa las predicciones del modelo"""
    detections = predictions[0]  # Primer elemento del batch
    
    # Extraer información
    boxes = detections[:, :4]  # [x1, y1, x2, y2]
    scores = detections[:, 4]  # Confianza
    labels = detections[:, 5].long()  # Etiquetas
    
    # Filtrar por confianza
    valid_detections = scores > confidence_threshold
    boxes = boxes[valid_detections]
    scores = scores[valid_detections]
    labels = labels[valid_detections]
    
    # Escalar las cajas al tamaño original
    scale_y = original_shape[0] / image_size
    scale_x = original_shape[1] / image_size
    
    boxes[:, [0, 2]] *= scale_x  # x1, x2
    boxes[:, [1, 3]] *= scale_y  # y1, y2
    
    return boxes, scores, labels

def draw_predictions(image, boxes, scores, labels, class_names):
    """Dibuja las predicciones en la imagen"""
    image_with_boxes = image.copy()
    
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.int().numpy()
        class_name = class_names[int(label)]
        color = COLORS.get(class_name, (255, 255, 255))
        
        # Dibujar caja
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # Dibujar etiqueta
        label_text = f"{class_name}: {score:.2f}"
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        
        # Fondo para el texto
        cv2.rectangle(image_with_boxes, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), 
                     color, -1)
        
        # Texto
        cv2.putText(image_with_boxes, label_text, 
                   (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return image_with_boxes

def diagnose_model(model, device):
    """Función para diagnosticar problemas del modelo"""
    print("\n" + "="*50)
    print("DIAGNÓSTICO DEL MODELO")
    print("="*50)
    
    # Test 1: Verificar que el modelo esté en modo eval
    print(f"1. Modo del modelo: {'EVAL' if not model.training else 'TRAIN'}")
    
    # Test 2: Crear una imagen de prueba
    test_image = torch.randn(1, 3, image_size, image_size).to(device)
    print(f"2. Imagen de prueba creada: {test_image.shape}")
    
    # Test 3: Hacer predicción
    try:
        with torch.no_grad():
            predictions = model(test_image)
        print(f"3. Predicción exitosa: {predictions.shape}")
        print(f"   Valores min/max: {predictions.min():.4f} / {predictions.max():.4f}")
        
        # Test 4: Verificar formato de salida
        detections = predictions[0]
        if detections.shape[1] >= 6:
            scores = detections[:, 4]
            labels = detections[:, 5]
            print(f"4. Formato correcto - Scores: {scores.shape}, Labels: {labels.shape}")
            print(f"   Scores estadísticas: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
            print(f"   Labels únicas: {torch.unique(labels)}")
        else:
            print(f"4. ERROR: Formato incorrecto - Shape: {detections.shape}")
            
    except Exception as e:
        print(f"3. ERROR en predicción: {e}")
        return False
    
    print("="*50)
    return True

def test_with_simple_image():
    """Crear una imagen de prueba simple para verificar detección"""
    # Crear imagen con formas simples
    img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 128  # Gris
    
    # Añadir rectángulos de colores diferentes
    cv2.rectangle(img, (50, 50), (150, 150), (255, 0, 0), -1)    # Azul
    cv2.rectangle(img, (200, 200), (300, 300), (0, 255, 0), -1)  # Verde  
    cv2.rectangle(img, (350, 100), (450, 200), (0, 0, 255), -1)  # Rojo
    
    return img
def predict_single_image(model, image_path, device, confidence_threshold=0.5, save_result=True):
    """Realiza predicción en una sola imagen"""
    print(f"Procesando imagen: {image_path}")
    
    # Preprocesar imagen
    tensor_image, original_image, original_shape = preprocess_image(image_path)
    tensor_image = tensor_image.to(device)
    
    # Inferencia
    with torch.no_grad():
        predictions = model(tensor_image)
    
    # Postprocesar
    boxes, scores, labels = postprocess_predictions(predictions, original_shape, confidence_threshold)
    
    print(f"Detectados {len(boxes)} objetos con confianza > {confidence_threshold}")
    
    # Mostrar resultados
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        class_name = CLASSES[int(label)]
        print(f"  {i+1}. {class_name}: {score:.3f}")
    
    # Dibujar predicciones
    if len(boxes) > 0:
        result_image = draw_predictions(original_image, boxes, scores, labels, CLASSES)
    else:
        result_image = original_image
    
    # Guardar resultado
    if save_result:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"resultado_{base_name}.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"Resultado guardado en: {output_path}")
    
    # Mostrar imagen
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Detecciones - {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()
    
    return boxes, scores, labels

def predict_from_camera(model, device, confidence_threshold=0.5):
    """Realiza predicción en tiempo real desde la cámara"""
    print("Iniciando predicción desde cámara. Presiona 'q' para salir.")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convertir frame a RGB para el modelo
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_shape = frame_rgb.shape[:2]
            
            # Transformar
            transformed = transform(image=frame_rgb)
            tensor_image = transformed['image'].unsqueeze(0).to(device)
            
            # Predicción
            with torch.no_grad():
                predictions = model(tensor_image)
            
            # Postprocesar
            boxes, scores, labels = postprocess_predictions(predictions, original_shape, confidence_threshold)
            
            # Dibujar en el frame original
            if len(boxes) > 0:
                frame = draw_predictions(frame, boxes, scores, labels, CLASSES)
            
            # Mostrar FPS aproximado
            cv2.putText(frame, f"Objetos detectados: {len(boxes)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('EfficientDet - Detección en Tiempo Real', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ================================
# FUNCIÓN PRINCIPAL
# ================================
def main():
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Cargar modelo - probar método alternativo si falla el principal
    try:
        model_path = os.path.join(model_dir, 'efficientdet_weights.pth')
        if os.path.exists(model_path):
            model = load_model(model_path, device)
        else:
            raise FileNotFoundError("Archivo principal no encontrado")
    except Exception as e:
        print(f"Método principal falló: {e}")
        print("Probando método alternativo...")
        model = load_model_alternative(model_dir, device)
    
    # Menú de opciones
    while True:
        print("\n" + "="*50)
        print("MENÚ DE INFERENCIA")
        print("="*50)
        print("1. Probar con imagen específica")
        print("2. Probar con cámara en tiempo real")
        print("3. Probar con múltiples imágenes en carpeta")
        print("4. 🔍 DIAGNOSTICAR MODELO")
        print("5. 🧪 PROBAR CON IMAGEN SINTÉTICA")
        print("6. Salir")
        print("="*50)
        
        opcion = input("Selecciona una opción (1-6): ").strip()
        
        if opcion == '1':
            image_path = input("Ingresa la ruta de la imagen: ").strip()
            if os.path.exists(image_path):
                confidence = float(input("Umbral de confianza (0.01-0.9, default 0.1): ") or 0.1)
                predict_single_image(model, image_path, device, confidence)
            else:
                print("La imagen no existe!")
                
        elif opcion == '2':
            confidence = float(input("Umbral de confianza (0.01-0.9, default 0.1): ") or 0.1)
            predict_from_camera(model, device, confidence)
            
        elif opcion == '3':
            folder_path = input("Ingresa la ruta de la carpeta: ").strip()
            if os.path.exists(folder_path):
                confidence = float(input("Umbral de confianza (0.01-0.9, default 0.1): ") or 0.1)
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                
                for filename in os.listdir(folder_path):
                    if any(filename.lower().endswith(ext) for ext in image_extensions):
                        image_path = os.path.join(folder_path, filename)
                        try:
                            predict_single_image(model, image_path, device, confidence)
                        except Exception as e:
                            print(f"Error procesando {filename}: {e}")
            else:
                print("La carpeta no existe!")
        
        elif opcion == '4':
            # Diagnóstico del modelo
            if diagnose_model(model, device):
                print("✅ Modelo funcionando correctamente")
            else:
                print("❌ Problemas detectados en el modelo")
                
        elif opcion == '5':
            # Probar con imagen sintética
            print("Creando imagen de prueba...")
            test_img = test_with_simple_image()
            
            # Guardar imagen temporal
            temp_path = "test_image.jpg"
            cv2.imwrite(temp_path, test_img)
            
            confidence = float(input("Umbral de confianza (0.01-0.9, default 0.01): ") or 0.01)
            predict_single_image(model, temp_path, device, confidence)
            
            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        elif opcion == '6':
            print("¡Hasta luego!")
            break
            
        else:
            print("Opción no válida!")

if __name__ == '__main__':
    main()