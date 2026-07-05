# Fish Pose Estimation with YOLO11

Computer vision project for detecting fish and localizing anatomical keypoints using a custom YOLO11 pose model. The predicted keypoints can be used as the basis for geometric measurements and future fish size or weight estimation.

## Project objectives

- Detect fish in images.
- Locate relevant anatomical keypoints.
- Evaluate the model using pose-estimation metrics.
- Use the predicted geometry for subsequent morphometric analysis.

## Model and training configuration

The project uses the Ultralytics implementation of **YOLO11 Pose** with the following main configuration:

- Base model: `yolo11s-pose.pt`
- Image size: `768 × 768`
- Epochs: `150`
- Batch size: `16`
- Mosaic augmentation disabled during the last `10` epochs

```python
from multiprocessing import freeze_support
from ultralytics import YOLO


def main():
    model = YOLO("yolo11s-pose.pt")

    model.train(
        data="fish_keypoint.yaml",
        imgsz=768,
        epochs=150,
        batch=16,
        project="runs_s/train",
        name="fish_yolo11s_pose",
        workers=0,
        close_mosaic=10,
    )


if __name__ == "__main__":
    freeze_support()
    main()
```

## Requirements

- Python 3.10 or later
- PyTorch
- Ultralytics
- OpenCV
- NumPy

Install the main dependency with:

```bash
pip install ultralytics
```

## Dataset configuration

The dataset must be described in `fish_keypoint.yaml`, including the paths to the training and validation images, the class names, and the keypoint configuration.

Example:

```yaml
path: path/to/dataset
train: images/train
val: images/val

names:
  0: fish

kpt_shape: [NUMBER_OF_KEYPOINTS, 3]
```

Replace `NUMBER_OF_KEYPOINTS` and the dataset paths with the values used in the project.

## Training

Run the Python script containing the training configuration:

```bash
python train.py
```

If the training script has a different filename, replace `train.py` with that filename.

## Outputs

Ultralytics stores the generated results in the directory configured through the `project` and `name` parameters. Typical outputs include:

- Model weights
- Training and validation curves
- Confusion matrices
- Precision, recall, and mAP metrics
- Prediction examples

Generated training runs and large model files are not included in this repository by default.

## Project status

This project is currently under development as part of the course **Development of Computer Vision Applications**.

## Author

**María Cristina Orihuela Flores**  
GitHub: [@zerocris13](https://github.com/zerocris13)
