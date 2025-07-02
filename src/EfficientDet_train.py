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
            
            # Verify image exists and can be read
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
                
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Could not read image: {img_path}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape

            boxes = []
            labels = []

            for ann in anns:
                x, y, w, h = ann['bbox']
                xmin = max(0, x)
                ymin = max(0, y)
                xmax = min(w, x + w)
                ymax = min(h, y + h)
                
                # Ensure boxes are valid and have positive area
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    # Ensure class labels are within bounds (0 to num_classes-1)
                    label = ann['category_id']
                    if label < 0 or label >= num_classes:
                        raise ValueError(f"Invalid class label {label} in {img_path}")
                    labels.append(label)

            # If no valid boxes, return empty tensors
            if len(boxes) == 0:
                img = torch.zeros((3, image_size, image_size), dtype=torch.float32)
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                # Apply transforms
                transformed = self.transforms(image=img, bboxes=boxes, class_labels=labels)
                img = transformed['image']
                boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
                labels = torch.tensor(transformed['class_labels'], dtype=torch.int64)

            target = {
                'bbox': boxes,
                'cls': labels,
                'img_size': torch.tensor([img.shape[1], img.shape[2]]),
                'img_scale': torch.tensor(1.0)
            }

            return img, target

        except Exception as e:
            print(f"Error processing image {img_path if 'img_path' in locals() else 'unknown'}: {str(e)}")
            # Return a dummy sample
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
        return optimizer

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
    batch_size=4, 
    shuffle=True, 
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True  # Add this for better performance
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=4, 
    shuffle=False, 
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
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
    
    # Initialize the model
    pl_model = EfficientDetLightningModel(model)

    # Configure trainer
    trainer = Trainer(
        max_epochs=2,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=5,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,  # Disable sanity check to see real errors
        precision=16,  # Try mixed precision
        deterministic=True  # For reproducibility
    )

    # Start training
    trainer.fit(pl_model, train_loader, val_loader)

    # Crear la carpeta si no existe
    model_dir = 'model_efficientDet'
    os.makedirs(model_dir, exist_ok=True)

    # Guardar el modelo completo (incluyendo la configuración)
    torch.save(pl_model.state_dict(), os.path.join(model_dir, 'efficientdet_model.pth'))

    # O guardar solo los pesos del modelo
    torch.save(model.state_dict(), os.path.join(model_dir, 'efficientdet_weights.pth'))

    # También puedes guardar el checkpoint de Lightning completo
    trainer.save_checkpoint(os.path.join(model_dir, 'lightning_checkpoint.ckpt'))

    print(f"Modelos guardados en la carpeta {model_dir}")