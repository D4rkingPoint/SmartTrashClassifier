# SmartTrashClassifier

Este repositorio contiene los notebooks y scripts utilizados para entrenar y evaluar diferentes modelos de detección de objetos con el objetivo de clasificar residuos. El propósito principal es comparar el rendimiento de arquitecturas como YOLO, Faster R-CNN y EfficientDet utilizando un dataset personalizado de basura.

El análisis comparativo se basa en métricas estándar como mAP, Precisión, Recall y F1-Score, para determinar el modelo más adecuado y eficiente que posteriormente será implementado en una aplicación móvil.

## Integrantes
- Leonardo Antonio Ponce Toledo         UTFSM   leonardo.ponce@ums.cl
- Álvaro Alejandro Pozo Fuentes         UTFSM   alvaro.pozo@usm.cl
- Nicolás Benjamín Ramírez Rodríguez    UTFSM   nicolas.ramirezro@usm.cl

## Enlace Dataset

[Dataset en roboflow](https://universe.roboflow.com/proyectos-qu6sq/clasificacion-de-resuidos)

[Enlace aplicación móvil](https://github.com/D4rkingPoint/SmartTrashClassifier_App/tree/main)

[Descarga modelo Faster R-CNN](https://usmcl-my.sharepoint.com/:u:/g/personal/alvaro_pozo_usm_cl/EZZWsHFNeANNh-sCNP6FvIUBcjpVptJkfNZrcRCHIBJshA?e=Nwa1Qe)

## Modelos a comparar

- Yolov11
- Yolov8
- EfficientDet
- Faster R-CNN

## Métricas a comparar:

- mAP (mean Average Precision)
- Precision
- Recall
- F1-Score

## Instalaciones necesarias

### Yolo

```
pip install ultralytics
```

### Fastet R-CNN

```
pip install torch torchvision pycocotools
```

### EfficientDet

- pip install torch torchvision effdet albumentations pytorch-lightning