import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Conv2D, Reshape, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# Crear carpeta datasets
HOME = os.getcwd()
DATASET_DIR = f"{HOME}/dataset"
DATASET_NAME = "dataset-para-proyecto-vision-MobileNet_SSD"
DATASET_PATH = os.path.join(DATASET_DIR, DATASET_NAME)

os.makedirs(DATASET_DIR, exist_ok=True)
os.chdir(DATASET_DIR)

# Verificar si ya existe el dataset
if not os.path.exists(DATASET_PATH):
    from roboflow import Roboflow
    rf = Roboflow(api_key="RdgAUhTbcOD8jUWNIy9A")
    project = rf.workspace("proyectos-qu6sq").project("dataset-para-proyecto-vision")
    version = project.version(2)
    dataset = version.download("voc", location=DATASET_PATH)                           
    dataset_path = dataset.location  # <-- Definimos la ruta cuando se descarga
else:
    print(f"El dataset '{DATASET_NAME}' ya está disponible, no se descarga nuevamente.")
    dataset_path = DATASET_PATH  # <-- Definimos la ruta manualmente



# ====================== CONFIGURACIÓN ======================
CLASSES = ['Trash', 'Metal', 'Plastic', 'Glass', 'Cardboard', 'Paper', 'Compostable']
IMG_SIZE = 300
BATCH_SIZE = 32
EPOCHS = 20

# Configuración SSD
FEATURE_MAP_SIZE = 19  # Tamaño del feature map de salida (19x19 para MobileNetV2 con input 300x300)
NUM_PRIORS = 6         # Prior boxes por ubicación en el feature map
TOTAL_PRIORS = FEATURE_MAP_SIZE * FEATURE_MAP_SIZE * NUM_PRIORS  # 19*19*6 = 2166

# ====================== FUNCIONES DE DATOS ======================
def parse_annotation(xml_path):
    """Versión robusta que maneja XML corruptos y objetos inválidos"""
    try:
        if os.path.getsize(xml_path) == 0:
            return np.zeros((0, 4)), []
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        for obj in root.findall("object"):
            try:
                label = obj.find("name").text.lower()
                if label not in CLASSES:
                    continue
                    
                bbox = obj.find("bndbox")
                xmin = max(0, float(bbox.find("xmin").text))
                ymin = max(0, float(bbox.find("ymin").text))
                xmax = min(IMG_SIZE, float(bbox.find("xmax").text))
                ymax = min(IMG_SIZE, float(bbox.find("ymax").text))
                
                if xmin >= xmax or ymin >= ymax:
                    continue
                    
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(CLASSES.index(label))
                
            except Exception as e:
                continue
                
        return np.array(boxes), np.array(labels)
    
    except Exception:
        return np.zeros((0, 4)), []

def load_dataset(dataset_path):
    """Carga solo archivos válidos con manejo robusto de errores"""
    train_dir = os.path.join(dataset_path, "train")
    valid_pairs = []
    
    for file in os.listdir(train_dir):
        if file.endswith(".xml"):
            xml_path = os.path.join(train_dir, file)
            img_name = file.replace(".xml", ".jpg")
            img_path = os.path.join(train_dir, img_name)
            
            if os.path.exists(img_path):
                boxes, labels = parse_annotation(xml_path)
                if len(boxes) > 0:
                    valid_pairs.append((img_path, xml_path))
    
    images = []
    boxes_list = []
    labels_list = []
    
    for img_path, xml_path in valid_pairs:
        boxes, labels = parse_annotation(xml_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
        
        images.append(img)
        boxes_list.append(boxes)
        labels_list.append(labels)
    
    return np.array(images), boxes_list, labels_list

def build_ssd_model(num_classes=len(CLASSES)):
    """Modelo SSD con MobileNetV2 como backbone - Versión corregida"""
    # Base pre-entrenada
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    for layer in base_model.layers:
        layer.trainable = False  # Congela el backbone
    
    # Capas de detección SSD
    x = base_model.get_layer('block_13_expand_relu').output
    
    # Capa adicional para mejor representación
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    
    # Branch para localización (4 coords * num_priors)
    loc_output = Conv2D(NUM_PRIORS * 4, (3, 3), padding='same')(x)
    loc_output = Reshape((TOTAL_PRIORS, 4))(loc_output)
    
    # Branch para clasificación ((num_classes + 1) * num_priors)
    cls_output = Conv2D(NUM_PRIORS * (num_classes + 1), (3, 3), padding='same')(x)
    cls_output = Reshape((TOTAL_PRIORS, num_classes + 1))(cls_output)
    
    # Concatenar salidas
    predictions = Concatenate(axis=-1)([loc_output, cls_output])
    
    return Model(inputs=base_model.input, outputs=predictions)

def prepare_data(images, boxes_list, labels_list):
    """Prepara los datos para el modelo SSD"""
    y = np.zeros((len(images), TOTAL_PRIORS, 4 + len(CLASSES) + 1))
    
    for i, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
        # Normalizar coordenadas
        boxes_norm = boxes / IMG_SIZE
        
        # Asignación simplificada (en producción usar matching por IoU)
        for j, (box, label) in enumerate(zip(boxes_norm, labels)):
            if j >= TOTAL_PRIORS:
                break
                
            # Coordenadas
            y[i, j, :4] = box
            
            # Clase (one-hot)
            y[i, j, 4 + label] = 1
    
    return images, y

def ssd_loss(y_true, y_pred):
    """Función de pérdida para SSD con manejo correcto de dimensiones"""
    # Separar componentes
    pred_loc = y_pred[..., :4]  # [batch, 2166, 4]
    pred_cls = y_pred[..., 4:]  # [batch, 2166, num_classes+1]
    
    true_loc = y_true[..., :4]  # [batch, 2166, 4]
    true_cls = y_true[..., 4:]  # [batch, 2166, num_classes+1]
    
    # Máscara para objetos positivos (ignorar background)
    pos_mask = tf.reduce_max(true_cls[..., 1:], axis=-1)  # [batch, 2166]
    
    # Número de objetos positivos (evitar división por cero)
    num_pos = tf.maximum(tf.reduce_sum(pos_mask), 1.0)
    
    # ===== Pérdida de localización =====
    # Smooth L1 loss solo para objetos positivos
    loc_diff = tf.abs(true_loc - pred_loc)
    loc_loss = tf.where(
        loc_diff < 1.0,
        0.5 * tf.square(loc_diff),
        loc_diff - 0.5
    )
    loc_loss = tf.reduce_sum(loc_loss, axis=-1)  # [batch, 2166]
    loc_loss = tf.reduce_sum(loc_loss * pos_mask) / num_pos
    
    # ===== Pérdida de clasificación =====
    # Cross-entropy con manejo de dimensiones
    cls_loss = tf.keras.losses.categorical_crossentropy(
        true_cls, pred_cls, from_logits=False
    )  # [batch, 2166]
    cls_loss = tf.reduce_sum(cls_loss * pos_mask) / num_pos
    
    return loc_loss + cls_loss


# Añade esto al inicio del script (por ejemplo, después de crear DATASET_DIR)
MODEL_DIR = f"{HOME}/model_mobilenet"
os.makedirs(MODEL_DIR, exist_ok=True)

# ====================== ENTRENAMIENTO ======================
def main():
    print("\nCargando dataset...")
    images, boxes_list, labels_list = load_dataset(DATASET_PATH)
    
    print("\nPreparando datos...")
    X, y = prepare_data(images, boxes_list, labels_list)
    print(f"Forma de X (imágenes): {X.shape}")
    print(f"Forma de y (anotaciones): {y.shape}")
    
    print("\nConstruyendo modelo...")
    model = build_ssd_model()
    #model.summary()
    
    
    print("\nCompilando modelo...")
    # Compilar con la nueva función de pérdida
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=ssd_loss,
        metrics=None
    )
    
    # Entrenar
    print("\nEntrenando modelo...")
    history = model.fit(
        X, y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[
            ModelCheckpoint(
                os.path.join(MODEL_DIR, "best_ssd.keras"),  # Cambio aquí
                monitor='val_loss', 
                save_best_only=True
            ),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
    )
    
    # Cambia esta línea también
    model.save(os.path.join(MODEL_DIR, "ssd_mobilenet_final.keras"))

if __name__ == "__main__":
    main()