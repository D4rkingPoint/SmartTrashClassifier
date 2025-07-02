import os
from roboflow import Roboflow
import cv2
import pytorch_lightning as pl
from pycocotools.coco import COCO
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset
from pytorch_lightning import Trainer
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import TQDMProgressBar
# ================================
# CONFIGURATION
# ================================
HOME = os.getcwd()
DATASET_DIR = os.path.join(HOME, "dataset")
DATASET_NAME = "dataset-para-proyecto-vision-efficientdet"
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
    print(f"Dataset '{DATASET_NAME}' already exists, skipping download.")
    dataset_path = DATASET_PATH

os.chdir(HOME)

CLASSES = ['Trash', 'Metal', 'Plastic', 'Glass', 'Cardboard', 'Paper', 'Compostable']
num_classes = len(CLASSES)
image_size = 512

# ================================
# MODEL CONFIGURATION (Faster R-CNN)
# ================================
def get_fasterrcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # +1 para incluir la clase de fondo
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model

model = get_fasterrcnn_model(num_classes)

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

print("Verificando categorías en las anotaciones...")
coco = COCO(os.path.join(DATASET_PATH, "train", "_annotations.coco.json"))
cat_ids = coco.getCatIds()
print(f"Categorías encontradas: {cat_ids}")
print(f"Número de clases configuradas: {num_classes}")

# Verifica que todas las categorías están dentro del rango esperado
for cat_id in cat_ids:
    if cat_id < 1 or cat_id > num_classes:
        print(f"¡ADVERTENCIA! Categoría ID {cat_id} fuera de rango (1-{num_classes})")


# ================================
# DATASET CLASS
# ================================
class COCODetectionDataset(Dataset):
    def __init__(self, img_dir, ann_path, transforms=None, verbose_errors=False):  # Added verbose_errors parameter here
        self.img_dir = img_dir
        self.coco = COCO(ann_path)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.verbose_errors = verbose_errors  # Store the parameter
        
        # Mapeo especial para manejar el ID 0
        self.cat_ids = sorted(self.coco.getCatIds())
        print(f"Categorías originales en COCO: {self.cat_ids}")
        
        # Asegurar que ningún ID sea menor que 1
        self.class_map = {v: i+1 for i, v in enumerate(self.cat_ids) if v >= 1}
        
        # Manejo especial para ID 0 (si existe)
        if 0 in self.cat_ids:
            print("¡Advertencia: Se encontró categoría con ID 0! Mapeando a 1")
            self.class_map[0] = 1  # Asignamos a la primera clase
        
        print(f"Mapa de clases final: {self.class_map}")
            

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        try:
            img_id = self.ids[index]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            img_info = self.coco.loadImgs(img_id)[0]
            path = img_info['file_name']
            img_path = os.path.join(self.img_dir, path)
            
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape

            boxes = []
            labels = []

            for ann in anns:
                x, y, w_ann, h_ann = ann['bbox']
                xmin = max(0, x)
                ymin = max(0, y)
                xmax = min(w, x + w_ann)
                ymax = min(h, y + h_ann)
                
                if xmax > xmin and ymax > ymin:
                    label = self.class_map[ann['category_id']]
                    if label < 1 or label > num_classes:
                        raise ValueError(f"Label {label} fuera de rango (1-{num_classes})")
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(label)
                
                # Verificación crítica
                if len(boxes) != len(labels):
                    raise ValueError(f"Discrepancia en {img_path}: {len(boxes)} boxes vs {len(labels)} labels")

            # Verificar que boxes y labels coincidan
            if len(boxes) != len(labels):
                if self.verbose_errors:
                    print(f"Mismatch in image {img_path}: {len(boxes)} boxes vs {len(labels)} labels")
                # Si hay mismatch, usar solo las primeras n coincidencias
                n = min(len(boxes), len(labels))
                boxes = boxes[:n]
                labels = labels[:n]

            if len(boxes) == 0:
                img = torch.zeros((3, image_size, image_size), dtype=torch.float32)
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                transformed = self.transforms(image=img, bboxes=boxes, class_labels=labels)
                img = transformed['image']
                boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
                labels = torch.as_tensor(transformed['class_labels'], dtype=torch.int64)

            target = {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([img_id]),
                "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
            }

            return img, target

        except Exception as e:
            print(f"Error procesando imagen: {str(e)}")
            return self._create_dummy_sample()
        
    def _create_dummy_sample(self):
        """Crea una muestra dummy para mantener el entrenamiento"""
        img = torch.zeros((3, image_size, image_size), dtype=torch.float32)
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
        return img, {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([0]),
            "area": torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((0,), dtype=torch.int64)
        }

# ================================
# LIGHTNING MODULE
# ================================
class FasterRCNNLightningModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=['model'])
        
        # Para debug
        print(f"Model device: {next(self.model.parameters()).device}")

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        try:
            images, targets = batch
            images = torch.stack([img for img in images])
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            
            self.log_dict({f"train_{k}": v for k, v in loss_dict.items()})
            self.log("train_loss", loss, prog_bar=True)
            return loss
            
        except RuntimeError as e:
            print(f"Error en GPU: {e}")
            return None

    def validation_step(self, batch, batch_idx):
        try:
            images, targets = batch
            images = torch.stack([img for img in images])
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            with torch.no_grad():
                # En validación, el modelo puede devolver tanto detecciones como pérdidas
                outputs = self.model(images, targets)
                
                # Manejar ambos casos: evaluación (devuelve listas) y validación (devuelve dict)
                if isinstance(outputs, list):
                    # Si es evaluación, calcular métricas manualmente
                    return {"val_outputs": outputs}
                else:
                    # Si es validación (con targets), calcular pérdida
                    loss = sum(loss for loss in outputs.values())
                    self.log_dict({f"val_{k}": v for k, v in outputs.items()})
                    self.log("val_loss", loss, prog_bar=True)
                    return loss
                    
        except RuntimeError as e:
            print(f"Error en validación: {e}")
            return None

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=0.005, 
            momentum=0.9, 
            weight_decay=0.0005
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [scheduler]

# ================================
# DATA LOADERS
# ================================
def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = COCODetectionDataset(
    img_dir=os.path.join(DATASET_PATH, "train"),
    ann_path=os.path.join(DATASET_PATH, "train", "_annotations.coco.json"),
    transforms=transform,
    verbose_errors=False  # Silencia los mensajes de error
)

val_dataset = COCODetectionDataset(
    img_dir=os.path.join(DATASET_PATH, "valid"),
    ann_path=os.path.join(DATASET_PATH, "valid", "_annotations.coco.json"),
    transforms=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=2,  # Mismo tamaño que train para consistencia
    shuffle=False, 
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=False
)
# ================================
# TRAINING
# ================================
# ================================
if __name__ == '__main__':
    # Configuración inicial
    torch.set_float32_matmul_precision('high')
    torch._dynamo.config.suppress_errors = True
    torch.backends.cudnn.benchmark = True
    
    # Verificar datos
    print("Verificando muestra de datos...")
    sample_img, sample_target = train_dataset[0]
    print(f"Tamaño imagen: {sample_img.shape}")
    print(f"Ejemplo de target: { {k: v.shape if isinstance(v, torch.Tensor) else v for k, v in sample_target.items()} }")

    # Inicializar modelo
    pl_model = FasterRCNNLightningModel(model)
    
    # Configurar trainer
    trainer = Trainer(
        callbacks=[TQDMProgressBar(refresh_rate=10)],
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=True,
        max_epochs=10,
        accelerator="auto"
    )

    # Entrenamiento
    print("Iniciando entrenamiento...")
    trainer.fit(pl_model, train_loader, val_loader)

    # Guardar modelo
    model_dir = 'model_fasterrcnn'
    os.makedirs(model_dir, exist_ok=True)
    torch.save(pl_model.model.state_dict(), os.path.join(model_dir, 'fasterrcnn_weights.pth'))
    print(f"Modelo guardado en {model_dir}")