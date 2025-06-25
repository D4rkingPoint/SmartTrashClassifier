import os
from roboflow import Roboflow

# ================================
# CONFIGURACIÓN DEL DATASET
# ================================

# Ruta principal del proyecto
HOME = os.getcwd()
DATASET_DIR = os.path.join(HOME, "dataset")
DATASET_NAME = "dataset-para-proyecto-vision-efficientdet"
DATASET_PATH = os.path.join(DATASET_DIR, DATASET_NAME)

# Crear carpeta si no existe
os.makedirs(DATASET_DIR, exist_ok=True)
os.chdir(DATASET_DIR)

# Descargar dataset en formato VOC si no existe
if not os.path.exists(DATASET_PATH):
    rf = Roboflow(api_key="RdgAUhTbcOD8jUWNIy9A")
    project = rf.workspace("proyectos-qu6sq").project("dataset-para-proyecto-vision")
    version = project.version(2)
    dataset = version.download("voc", location=DATASET_PATH)
    dataset_path = dataset.location
else:
    print(f"El dataset '{DATASET_NAME}' ya está disponible, no se descarga nuevamente.")
    dataset_path = DATASET_PATH

# Volver a la ruta base
os.chdir(HOME)

# ================================
# CONVERSIÓN A TFRecord Y ENTRENAMIENTO EfficientDet
# ================================

# Requiere: create_pascal_tf_record.py ubicado en la raíz
# y carpeta VOCdevkit dentro del dataset descargado

os.system(
    "python create_pascal_tf_record.py "
    f"--label_map_path={dataset_path}/label_map.pbtxt "
    f"--data_dir={dataset_path}/VOCdevkit "
    "--year=VOC2023 --set=train "
    f"--output_path={dataset_path}/train.record"
)

os.system(
    "python create_pascal_tf_record.py "
    f"--label_map_path={dataset_path}/label_map.pbtxt "
    f"--data_dir={dataset_path}/VOCdevkit "
    "--year=VOC2023 --set=val "
    f"--output_path={dataset_path}/val.record"
)

# Entrenamiento con EfficientDet desde automl/efficientdet
os.chdir("automl/efficientdet")
os.system(
    "python main.py "
    "--mode=train "
    f"--training_file_pattern={dataset_path}/train.record "
    f"--validation_file_pattern={dataset_path}/val.record "
    "--model_name=efficientdet-d0 "
    f"--model_dir=models/efficientdet_recycler "
    "--hparams=configs/metal_recycler.yaml "
    "--num_epochs=100"
)
