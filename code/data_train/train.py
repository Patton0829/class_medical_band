from ultralytics import YOLO
import torch
from pathlib import Path
import yaml


def main():
    train_py_dir = Path(__file__).resolve().parent
    project_root = train_py_dir.parent.parent
    train_model_config_path = train_py_dir / "config" / "train_model_config.yaml"
    dataset_root = project_root / "dataset" / "label_box_image_dataset_det_yolov8"
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    with train_model_config_path.open("r", encoding="utf-8") as f:
        train_data_config = yaml.safe_load(f)
    # Avoid ambiguity from Ultralytics relative-path resolution by writing a resolved YAML.
    train_data_config["path"] = str(dataset_root)
    resolved_train_model_config_path = train_py_dir / "config" / "train_model_config.resolved.yaml"
    with resolved_train_model_config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(train_data_config, f, allow_unicode=True, sort_keys=False)

    if torch.cuda.is_available():
        train_device = "0"
        export_device = 0
        use_half = True
    elif torch.backends.mps.is_available():
        train_device = "mps"
        export_device = "cpu"
        use_half = False
    else:
        train_device = "cpu"
        export_device = "cpu"
        use_half = False

    # 3. 数据训练
    model = YOLO("yolo11s.yaml").load(r"./yolo11s.pt")  # build from YAML and transfer weights
    model.train(data=str(resolved_train_model_config_path), batch=16, epochs=300, imgsz=1280, device=train_device)

    # 3. 模型转换
    path = model.export(format='onnx', half=use_half, imgsz=1280, device=export_device)
    print(path)


if __name__ == '__main__':
    main()
