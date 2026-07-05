# Detección de peces y estimación de keypoints 2D mediante YOLO11s-Pose

Proyecto desarrollado para el curso **Desarrollo en Aplicaciones con Visión Artificial**.

## Descripción del proyecto

La medición del tamaño y la masa de los peces normalmente requiere manipularlos de manera directa. Además de demandar tiempo, este procedimiento puede generar estrés en los animales y dificultar la realización de mediciones frecuentes.

Este proyecto propone una alternativa basada en visión artificial para detectar peces en imágenes y localizar puntos anatómicos relevantes de su cuerpo. Para ello, se realiza el fine-tuning de un modelo **YOLO11 Pose**, el cual permite identificar al pez y estimar la posición de sus puntos clave.

A partir de estos puntos se pueden calcular medidas morfométricas, como la longitud y el ancho corporal, que posteriormente pueden relacionarse con la masa del pez. De esta manera, se busca sentar las bases para un sistema automático, rápido y no invasivo de monitoreo.

## Objetivo

Desarrollar un modelo de visión artificial capaz de detectar peces y localizar puntos anatómicos de referencia para obtener medidas corporales útiles en la estimación de su tamaño y masa.

## Metodología general

El proyecto considera las siguientes etapas:

1. Selección y preparación de imágenes de peces.
2. Anotación de los peces y sus puntos anatómicos.
3. División del dataset en conjuntos de entrenamiento, validación y prueba.
4. Fine-tuning de un modelo YOLO11 Pose.
5. Evaluación mediante métricas de detección y estimación de puntos clave.
6. Obtención de medidas morfométricas a partir de las coordenadas predichas.
7. Análisis de la relación entre las medidas corporales y la masa del pez.

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