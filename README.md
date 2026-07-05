# Detección de peces y estimación de keypoints 2D mediante YOLO11s-Pose

Proyecto desarrollado para el curso **Desarrollo en Aplicaciones con Visión Artificial**.

## Descripción del proyecto

Este proyecto aborda la detección automática de peces en imágenes submarinas y la localización de puntos anatómicos de referencia. Las condiciones de iluminación, turbidez, densidad de peces y posibles superposiciones hacen que esta tarea sea más compleja que la detección en imágenes convencionales.

Para ello, se realizó el fine-tuning de un modelo **YOLO11s-Pose**, adaptado para detectar una sola clase, `fish`, y estimar cuatro keypoints 2D por pez: cabeza, cola, punto superior y punto inferior. El modelo fue entrenado y evaluado con imágenes de carpa herbívora obtenidas en distintas condiciones acuáticas.

El resultado del proyecto es un modelo capaz de detectar múltiples peces y localizar sus cuatro puntos anatómicos en una sola etapa. El trabajo se concentra únicamente en la detección y estimación de keypoints 2D; no se realizó el cálculo del tamaño ni la estimación de la masa de los peces.

## Objetivo

Desarrollar y evaluar un modelo basado en **YOLO11s-Pose** para detectar peces en imágenes submarinas y localizar cuatro keypoints anatómicos: cabeza, cola, punto superior y punto inferior.

## Metodología general

El proyecto considera las siguientes etapas:

1. Selección y preparación de imágenes de peces.
2. Anotación de los peces y sus puntos anatómicos.
3. División del dataset en conjuntos de entrenamiento, validación y prueba.
4. Fine-tuning de un modelo YOLO11 Pose.
5. Evaluación mediante métricas de detección y estimación de puntos clave.

## Modelo utilizado

Se emplea el modelo **YOLO11s Pose** de Ultralytics. Este modelo permite realizar dos tareas de manera conjunta:

- detectar la ubicación del pez mediante una caja delimitadora;
- predecir la posición de los puntos anatómicos definidos durante la anotación.

La configuración principal utilizada durante el entrenamiento fue:

- Modelo base: `yolo11s-pose.pt`
- Tamaño de imagen: `768 × 768`
- Número de épocas: `150`
- Batch size: `16`
- Desactivación de mosaic durante las últimas `10` épocas

## Contenido del repositorio

- `fish_keypoint.yaml`: configuración del dataset y de los puntos clave.
- `script_train_fine_tuning.py`: script utilizado para entrenar el modelo.
- `predict_test.py`: script para realizar predicciones sobre nuevas imágenes.
- `README.md`: descripción general del proyecto.

Los pesos entrenados, las imágenes y los resultados generados durante el entrenamiento no se incluyen directamente en el repositorio debido a su tamaño.

## Dataset

El dataset utilizado en el proyecto, junto con sus imágenes y anotaciones, se encuentra disponible en el siguiente enlace:

[Dataset del proyecto en Google Drive](https://drive.google.com/drive/folders/14G5qUpQH5qdSwMXRlEMci_c-Zf4z_bZS)

## Integrantes

- María Cristina Orihuela Flores
- María Emilia Ochoa Enríquez
- Juan Janpier Mío Mío
- Sintia Vallet Marín Rodríguez

## Referencia principal

El proyecto toma como referencia el trabajo:

J. Wang, Z. Cheng, M. Lin, R. Yang y Q. Huang, “FishKP-YOLOv11: An Automatic Estimation Model for Fish Size and Mass in Complex Underwater Environments,” *Animals*, vol. 15, no. 19, art. 2862, 2025.

[https://doi.org/10.3390/ani15192862](https://doi.org/10.3390/ani15192862)