import os
from roboflow import Roboflow
import cv2
import pytorch_lightning as pl
from pycocotools.coco import COCO
import torch
from torch.utils.data import Dataset
import numpy as np
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from pytorch_lightning import Trainer
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

# ================================
# CONFIGURATION
# ================================
HOME = os.getcwd()
DATASET_DIR = os.path.join(HOME, "dataset")
DATASET_NAME = "dataset-para-proyecto-vision-Efficientdet"
DATASET_PATH = os.path.join(DATASET_DIR, DATASET_NAME)

os.makedirs(DATASET_DIR, exist_ok=True)
os.chdir(DATASET_DIR)


if not os.path.exists(DATASET_PATH):
    rf = Roboflow(api_key="RdgAUhTbcOD8jUWNIy9A")
    project = rf.workspace("proyectos-qu6sq").project("dataset-para-proyecto-vision")
    version = project.version(3)
    dataset = version.download("coco", location=DATASET_PATH)
    dataset_path = dataset.location
else:
    print(f"Dataset '{DATASET_NAME}' ya está disponible, no se descarga nuevamente.")
    dataset_path = DATASET_PATH

os.chdir(HOME)

CLASSES = ['Trash', 'Metal', 'Plastic', 'Glass', 'Cardboard', 'Paper', 'Compostable']
num_classes = len(CLASSES)
image_size = 512  # Reduced size for better memory usage

# ================================
# MODEL CONFIGURATION
# ================================
config = get_efficientdet_config('tf_efficientdet_d0')
config.num_classes = num_classes
config.image_size = (image_size, image_size)

net = EfficientDet(config, pretrained_backbone=True)
net.class_net = HeadNet(config, num_outputs=num_classes)
model = DetBenchTrain(net, config)

# ================================
# DATA TRANSFORMS
# ================================
transform = A.Compose([
    A.Resize(image_size, image_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], 
bbox_params=A.BboxParams(
    format='pascal_voc', 
    label_fields=['class_labels'],
    min_visibility=0.4,
    min_area=8
))

# ================================
# DATASET CLASS
# ================================
class COCODetectionDataset(Dataset):
    def __init__(self, img_dir, ann_path, transforms=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        
        # Creamos un mapa para asegurar que usamos las clases correctas
        # Mapeamos los IDs de COCO [1, 2, ..., 7] a los índices del modelo [0, 1, ..., 6]
        coco_cat_ids = sorted(self.coco.getCatIds())
        self.class_map = {cat_id: i for i, cat_id in enumerate(c for c in coco_cat_ids if c > 0)}
        #print(f"Mapeo de clases creado (ID_COCO -> ID_Modelo): {self.class_map}")


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        try:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            img_info = self.coco.loadImgs(img_id)[0]
            path = img_info['file_name']
            img_path = os.path.join(self.img_dir, path)

            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"No se pudo leer la imagen: {img_path}")
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            boxes = []
            labels = []

            for ann in anns:
                # Ignoramos las anotaciones que no están en nuestro mapa (ej: categoría 0)
                if ann['category_id'] not in self.class_map:
                    continue
                
                x, y, w_bbox, h_bbox = ann['bbox']
                xmin = max(0, x)
                ymin = max(0, y)
                xmax = min(img.shape[1], x + w_bbox)
                ymax = min(img.shape[0], y + h_bbox)

                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    # Usamos el mapa para obtener la etiqueta correcta (0 a 6)
                    labels.append(self.class_map[ann['category_id']])

            # Si no hay cajas válidas, creamos tensores vacíos
            if not boxes:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)

            # Empaquetamos en un diccionario para la transformación
            sample = {
                'image': img,
                'bboxes': boxes,
                'class_labels': labels
            }

            # Aplicamos las transformaciones de Albumentations
            if self.transforms:
                transformed = self.transforms(**sample)
                img = transformed['image']
                boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
                labels = torch.as_tensor(transformed['class_labels'], dtype=torch.int64)

            # Creamos el diccionario de salida "target"
            target = {
                'bbox': boxes,
                'cls': labels,
                'img_size': torch.tensor([img.shape[1], img.shape[2]]), # Usamos el tamaño después de transformar
                'img_scale': torch.tensor(1.0)
            }

            return img, target

        except Exception as e:
            print(f"Error procesando el ID de imagen {img_id}: {str(e)}")
            # Devolvemos una muestra vacía para no detener el entrenamiento
            img = torch.zeros((3, image_size, image_size), dtype=torch.float32)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            return img, {'bbox': boxes, 'cls': labels, 'img_size': torch.tensor([image_size, image_size]), 'img_scale': torch.tensor(1.0)}

# ================================
# LIGHTNING MODULE
# ================================
class EfficientDetLightningModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, targets):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = torch.stack(images)
        
        # Convert targets to format expected by EfficientDet
        target_dict = {
            'bbox': [t['bbox'] for t in targets],
            'cls': [t['cls'] for t in targets],
            'img_size': torch.stack([t['img_size'] for t in targets]),
            'img_scale': torch.stack([t['img_scale'] for t in targets])
        }
        
        loss_dict = self.model(images, target_dict)
        loss = loss_dict['loss']
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = torch.stack(images)
        
        target_dict = {
            'bbox': [t['bbox'] for t in targets],
            'cls': [t['cls'] for t in targets],
            'img_size': torch.stack([t['img_size'] for t in targets]),
            'img_scale': torch.stack([t['img_scale'] for t in targets])
        }
        
        loss_dict = self.model(images, target_dict)
        val_loss = loss_dict['loss']
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
    
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',      # Reducirá el LR cuando la 'val_loss' deje de mejorar
            factor=0.1,      # Reduce el LR en un factor de 10
            patience=5,      # Número de épocas sin mejora antes de reducir el LR
            #verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss", # Métrica a monitorear
            },
        }

# ================================
# DATA LOADERS
# ================================
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

train_dataset = COCODetectionDataset(
    img_dir=os.path.join(DATASET_PATH, "train"),
    ann_path=os.path.join(DATASET_PATH, "train", "_annotations.coco.json"),
    transforms=transform
)

val_dataset = COCODetectionDataset(
    img_dir=os.path.join(DATASET_PATH, "valid"),
    ann_path=os.path.join(DATASET_PATH, "valid", "_annotations.coco.json"),
    transforms=transform
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=16, 
    shuffle=True, 
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True  # Add this for better performance
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=16, 
    shuffle=False, 
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True  # Add this for better performance
)


def validate_dataset(dataset):
    for i in range(len(dataset)):
        try:
            img, target = dataset[i]
            # Check labels are within bounds
            if len(target['cls']) > 0:
                assert (target['cls'] >= 0).all() and (target['cls'] < num_classes).all()
            # Check boxes are valid
            if len(target['bbox']) > 0:
                assert (target['bbox'][:, 2] > target['bbox'][:, 0]).all()
                assert (target['bbox'][:, 3] > target['bbox'][:, 1]).all()
        except Exception as e:
            print(f"Validation failed for sample {i}: {str(e)}")
            raise

# ================================
# TRAINING
# ================================

if __name__ == '__main__':
    # Windows-specific multiprocessing handling
    import multiprocessing
    multiprocessing.freeze_support()
    
    torch.set_float32_matmul_precision('high') # tarjeta de video

    # Initialize the model
    pl_model = EfficientDetLightningModel(model)

    # Configure trainer
    trainer = Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,  # Disable sanity check to see real errors
        precision=16,  # Try mixed precision
        deterministic=False  # For reproducibility
    )

    # Start training
    trainer.fit(pl_model, train_loader, val_loader)

    # Crear la carpeta si no existe
    model_dir = 'model_efficientDet'
    os.makedirs(model_dir, exist_ok=True)

    # Guardar el modelo completo (incluyendo la configuración)
    torch.save(pl_model.state_dict(), os.path.join(model_dir, 'efficientdet_model.pth'))

    # O guardar solo los pesos del modelo
    torch.save(model.model.state_dict(), os.path.join(model_dir, 'efficientdet_weights.pth'))

    # También puedes guardar el checkpoint de Lightning completo
    trainer.save_checkpoint(os.path.join(model_dir, 'lightning_checkpoint.ckpt'))

    print(f"Modelos guardados en la carpeta {model_dir}")