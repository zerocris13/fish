from multiprocessing import freeze_support
from ultralytics import YOLO


def main():
    model = YOLO("yolo11s-pose.pt")

    model.train(
        data="F:/Cristina/fish/fish_keypoint.yaml",
        imgsz=768,
        epochs=150,
        batch=16,
        project="F:/Cristina/fish/runs_s_768_con_flip/train",
        name="fish_yolo11s_pose",
        workers=0,
        close_mosaic=10        
    )


if __name__ == "__main__":
    freeze_support()
    main()