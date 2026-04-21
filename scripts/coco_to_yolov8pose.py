"""
Convert Roboflow COCO keypoint export -> YOLOv8 Pose label format.

Board  (class 0): 4 keypoints  [top, right, bottom, left]
Dart   (class 1): 1 tip keypoint, padded to 4 with zeros

Output line format (17 values):
  class cx cy w h  kp0x kp0y kp0v  kp1x kp1y kp1v  kp2x kp2y kp2v  kp3x kp3y kp3v

Usage:
    python coco_to_yolov8pose.py \
        --coco    _annotations.coco.json \
        --out     data_yolo \
        --val_ratio 0.2
"""

import argparse
import json
import random
import shutil
from pathlib import Path


def convert(coco_path: str, images_dir: str, out_root: str, val_ratio: float = 0.2):
    with open(coco_path) as f:
        data = json.load(f)

    # category_id -> yolo class index (skip supercategory 'dartbuddy' id=0)
    cat_map = {}
    for cat in data['categories']:
        if cat['name'] == 'board':
            cat_map[cat['id']] = 0
        elif cat['name'] == 'dart':
            cat_map[cat['id']] = 1

    # image_id -> image info
    images = {img['id']: img for img in data['images']}

    # image_id -> list of annotations
    from collections import defaultdict
    anns_by_image = defaultdict(list)
    for ann in data['annotations']:
        if ann['category_id'] not in cat_map:
            continue
        anns_by_image[ann['image_id']].append(ann)

    # train/val split
    random.seed(42)
    image_ids = list(anns_by_image.keys())
    random.shuffle(image_ids)
    n_val = max(1, int(len(image_ids) * val_ratio))
    val_ids = set(image_ids[:n_val])

    out = Path(out_root)
    for split in ('train', 'val'):
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)

    stats = {'train': 0, 'val': 0, 'skipped': 0}

    for img_id, anns in anns_by_image.items():
        img_info = images[img_id]
        W = img_info['width']
        H = img_info['height']
        fname = img_info['file_name']

        src = Path(images_dir) / fname
        if not src.exists():
            stats['skipped'] += 1
            continue

        lines = []
        for ann in anns:
            cls = cat_map[ann['category_id']]
            bx, by, bw, bh = (float(v) for v in ann['bbox'])

            # Normalise bbox to 0-1 (COCO bbox is x,y,w,h in pixels)
            cx = (bx + bw / 2) / W
            cy = (by + bh / 2) / H
            nw = bw / W
            nh = bh / H

            cx, cy, nw, nh = (max(0.0, min(1.0, v)) for v in (cx, cy, nw, nh))

            raw_kps = ann.get('keypoints', [])
            # raw_kps is flat [x, y, v, x, y, v, ...]
            parsed = []
            for i in range(0, len(raw_kps), 3):
                kx = float(raw_kps[i])     / W
                ky = float(raw_kps[i + 1]) / H
                kv = int(raw_kps[i + 2])
                parsed.append((
                    max(0.0, min(1.0, kx)),
                    max(0.0, min(1.0, ky)),
                    kv,
                ))

            # Pad / truncate to exactly 4 keypoints
            while len(parsed) < 4:
                parsed.append((0.0, 0.0, 0))
            parsed = parsed[:4]

            kp_str = ' '.join(f'{kx:.6f} {ky:.6f} {kv}' for kx, ky, kv in parsed)
            lines.append(f'{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} {kp_str}')

        split = 'val' if img_id in val_ids else 'train'
        label_path = out / 'labels' / split / (Path(fname).stem + '.txt')
        label_path.write_text('\n'.join(lines), encoding='utf-8')

        shutil.copy(src, out / 'images' / split / fname)
        stats[split] += 1

    print(f"Done.  train={stats['train']}  val={stats['val']}  skipped={stats['skipped']}")


def write_yaml(out_root: str):
    content = f"""path: {Path(out_root).resolve()}
train: images/train
val:   images/val

nc: 2
names: ['board', 'dart']

kpt_shape: [4, 3]
flip_idx: [0, 3, 2, 1]
"""
    p = Path(out_root) / 'dartboard_pose.yaml'
    p.write_text(content)
    print(f"YAML written to {p}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--coco',      required=True, help='Path to _annotations.coco.json')
    p.add_argument('--images',    required=True, help='Folder containing the image files')
    p.add_argument('--out',       default='data_yolo')
    p.add_argument('--val_ratio', type=float, default=0.2)
    args = p.parse_args()

    convert(args.coco, args.images, args.out, args.val_ratio)
    write_yaml(args.out)