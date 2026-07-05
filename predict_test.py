from pathlib import Path
import csv

from ultralytics import YOLO


def main():
    model_path = (
        "F:/Cristina/fish/runs_s_768_con_flip/train/"
        "fish_yolo11s_pose/weights/best.pt"
    )
    source_path = "F:/Cristina/fish/images/test"

    project_path = "F:/Cristina/fish/runs_s_768_con_flip/predict"
    experiment_name = "yolo11s_pose_test"

    model = YOLO(model_path)

    results = model.predict(
        source=source_path,
        imgsz=768,
        conf=0.45,
        iou=0.70,
        save=True,
        save_txt=True,
        save_conf=True,
        project=project_path,
        name=experiment_name,
        exist_ok=True,
        show_labels=True,
        show_conf=True,
        line_width=2,
        verbose=True
    )

    # Ruta final de salida
    output_dir = Path(project_path) / experiment_name
    csv_path = output_dir / "predictions.csv"

    # Crear CSV con cajas y keypoints
    fieldnames = [
        "image",
        "detection_id",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "x_center",
        "y_center",
        "width",
        "height",
        "kp1_x",
        "kp1_y",
        "kp1_conf",
        "kp2_x",
        "kp2_y",
        "kp2_conf",
        "kp3_x",
        "kp3_y",
        "kp3_conf",
        "kp4_x",
        "kp4_y",
        "kp4_conf",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            image_name = Path(result.path).name

            if result.boxes is None or len(result.boxes) == 0:
                continue

            # Bounding boxes en píxeles
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            boxes_xywh = result.boxes.xywh.cpu().numpy()

            # Clases y confianza
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            # Keypoints en píxeles
            keypoints_xy = None
            keypoints_conf = None

            if result.keypoints is not None:
                keypoints_xy = result.keypoints.xy.cpu().numpy()

                if result.keypoints.conf is not None:
                    keypoints_conf = result.keypoints.conf.cpu().numpy()

            for detection_id in range(len(boxes_xyxy)):
                x1, y1, x2, y2 = boxes_xyxy[detection_id]
                xc, yc, width, height = boxes_xywh[detection_id]

                class_id = classes[detection_id]
                class_name = result.names[class_id]

                row = {
                    "image": image_name,
                    "detection_id": detection_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(confidences[detection_id]),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "x_center": float(xc),
                    "y_center": float(yc),
                    "width": float(width),
                    "height": float(height),
                }

                # Guardar los cuatro keypoints
                for kp_index in range(4):
                    if (
                        keypoints_xy is not None
                        and detection_id < len(keypoints_xy)
                        and kp_index < len(keypoints_xy[detection_id])
                    ):
                        kp_x, kp_y = keypoints_xy[detection_id][kp_index]

                        if keypoints_conf is not None:
                            kp_conf = keypoints_conf[detection_id][kp_index]
                        else:
                            kp_conf = ""

                        row[f"kp{kp_index + 1}_x"] = float(kp_x)
                        row[f"kp{kp_index + 1}_y"] = float(kp_y)
                        row[f"kp{kp_index + 1}_conf"] = (
                            float(kp_conf) if kp_conf != "" else ""
                        )
                    else:
                        row[f"kp{kp_index + 1}_x"] = ""
                        row[f"kp{kp_index + 1}_y"] = ""
                        row[f"kp{kp_index + 1}_conf"] = ""

                writer.writerow(row)

    print("\nPredicción terminada.")
    print(f"Resultados guardados en: {output_dir}")
    print(f"CSV guardado en: {csv_path}")


if __name__ == "__main__":
    main()