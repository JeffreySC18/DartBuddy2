import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("pip install ultralytics")

POSE_MODELS = {
    'nano':   'yolov8n-pose.pt',
    'small':  'yolov8s-pose.pt',
    'medium': 'yolov8m-pose.pt',
}


def train(
    data_yaml:  str  = 'data_yolo/dartboard_pose.yaml',
    model_size: str  = 'medium',
    epochs:     int  = 200,
    imgsz:      int  = 800,
    batch:      int  = 16,
    device:     str  = 'cuda',
    patience:   int  = 40,
    project:    str  = 'runs/dartboard_pose',
    name:       str  = 'train',
    resume:     bool = False,
    finetune:   str  = None,
):
    base_model = finetune if finetune else POSE_MODELS.get(model_size, model_size)
    model = YOLO(base_model)

    print(f"Training YOLOv8-Pose ({base_model})")
    print(f"  data: {data_yaml}  epochs: {epochs}  batch: {batch}  device: {device}\n")

    model.train(
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
        hsv_v=0.5,
        hsv_s=0.4,
        hsv_h=0.01,
        degrees=30.0,
        perspective=0.001,
        scale=0.4,
        translate=0.1,
        fliplr=0.5,
        flipud=0.0,
        mosaic=0.5,
    )

    best = Path(project) / name / 'weights' / 'best.pt'
    print(f"\nDone. Best weights: {best}")


def validate(weights: str, data_yaml: str):
    model = YOLO(weights)
    metrics = model.val(data=data_yaml)
    print(f"\nBox  mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
    if hasattr(metrics, 'pose'):
        print(f"Pose mAP50: {metrics.pose.map50:.4f}  mAP50-95: {metrics.pose.map:.4f}")


def export(weights: str, fmt: str = 'onnx', imgsz: int = 800):
    model = YOLO(weights)
    out = model.export(format=fmt, imgsz=imgsz)
    print(f"Exported to: {out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd')

    t = sub.add_parser('train')
    t.add_argument('--data',     default='data_yolo/dartboard_pose.yaml')
    t.add_argument('--model',    default='small', choices=['nano', 'small', 'medium'])
    t.add_argument('--epochs',   type=int, default=200)
    t.add_argument('--batch',    type=int, default=16)
    t.add_argument('--device',   default='cuda')
    t.add_argument('--resume',   action='store_true')
    t.add_argument('--finetune', default=None, help='Path to existing .pt to fine-tune from')

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
            batch=args.batch,
            device=args.device,
            resume=args.resume,
            finetune=args.finetune,
        )
    elif args.cmd == 'val':
        validate(args.weights, args.data)
    elif args.cmd == 'export':
        export(args.weights, args.format)
    else:
        p.print_help()