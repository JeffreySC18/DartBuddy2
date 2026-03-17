"""
Fine-tune YOLOv8-Pose on the converted DeepDarts dataset.

YOLOv8 pose models predict:
  - Bounding boxes around each object (board / dart)
  - Keypoints within each box
     - board instance -> 4 calibration keypoints
     - dart instance  -> 1 tip keypoint (3 padded to zeros)

"""

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("Install ultralytics: pip install ultralytics")


# Model options
POSE_MODELS = {
    'nano':   'yolov8n-pose.pt',
    'small':  'yolov8s-pose.pt',
    'medium': 'yolov8m-pose.pt',
}


def train(
    data_yaml:  str   = 'data_yolo/dartboard_pose.yaml',
    model_size: str   = 'nano',
    epochs:     int   = 100,
    imgsz:      int   = 800,        
    batch:      int   = 8,
    device:     str   = 'cpu',
    patience:   int   = 20,
    project:    str   = 'runs/dartboard_pose',
    name:       str   = 'train',
    resume:     bool  = False,
):
    base_model = POSE_MODELS.get(model_size, model_size)
    model = YOLO(base_model)

    print(f"Training YOLOv8-Pose ({base_model})")
    print(f"  data:   {data_yaml}")
    print(f"  epochs: {epochs}  imgsz: {imgsz}  batch: {batch}  device: {device}\n")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        project=project,
        name=name,
        resume=resume,
        save=True,
        plots=True,
        # Augmentations (see YOLOv8 docs for details)
        hsv_v=0.5,
        hsv_s=0.4,
        hsv_h=0.01,
        # Moderate rotation since camera may not be perfectly square-on
        degrees=10.0,
        # Scale variation for different distances
        scale=0.3,
        translate=0.1,
        # Horizontal flip is valid (board is symmetric)
        fliplr=0.5,
        flipud=0.0,    # boards don't appear upside down
        mosaic=0.5,
    )

    best = Path(project) / name / 'weights' / 'best.pt'
    print(f"\n Training complete. Best weights: {best}")
    return results


def validate(weights: str, data_yaml: str):
    model = YOLO(weights)
    metrics = model.val(data=data_yaml)
    print(f"\nBox mAP50:     {metrics.box.map50:.4f}")
    print(f"Box mAP50-95:  {metrics.box.map:.4f}")
    # Pose metrics (keypoint mAP)
    if hasattr(metrics, 'pose'):
        print(f"Pose mAP50:    {metrics.pose.map50:.4f}")
        print(f"Pose mAP50-95: {metrics.pose.map:.4f}")
    return metrics


def export(weights: str, fmt: str = 'onnx', imgsz: int = 800):
    model = YOLO(weights)
    out   = model.export(format=fmt, imgsz=imgsz)
    print(f"✅ Exported to: {out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd')

    t = sub.add_parser('train')
    t.add_argument('--data',    default='data_yolo/dartboard_pose.yaml')
    t.add_argument('--model',   default='nano', choices=['nano','small','medium'])
    t.add_argument('--epochs',  type=int, default=100)
    t.add_argument('--imgsz',   type=int, default=800)
    t.add_argument('--batch',   type=int, default=8)
    t.add_argument('--device',  default='cpu')
    t.add_argument('--resume',  action='store_true')

    v = sub.add_parser('val')
    v.add_argument('--weights', required=True)
    v.add_argument('--data',    default='data_yolo/dartboard_pose.yaml')

    e = sub.add_parser('export')
    e.add_argument('--weights', required=True)
    e.add_argument('--format',  default='onnx')

    args = p.parse_args()

    if args.cmd == 'train':
        train(
            data_yaml=args.data,
            model_size=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            resume=args.resume,
        )
    elif args.cmd == 'val':
        validate(args.weights, args.data)
    elif args.cmd == 'export':
        export(args.weights, args.format)
    else:
        p.print_help()