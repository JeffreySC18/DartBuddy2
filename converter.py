"""
Converts DeepDarts labels.pkl -> YOLOv8 keypoint (pose) format.

Split strategy: 80/20 per-image within each folder, so every folder type
(empty boards, 1 dart, 2 darts, 3 darts) is represented in both train and val.

Usage:
    python convert_deepdarts_to_yolov8.py 
        --labels  labels.pkl 
        --images  Megafolder/cropped_images/800 
        --out     data_yolo
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


# Train/val split

def build_split_map(df, val_ratio: float = 0.2, seed: int = 42) -> dict:
    """
    Returns a dict mapping (img_folder, img_name) -> 'train' or 'val'.
    Splits within each folder independently so every folder type
    (empty board, 1 dart, 2 darts etc.) contributes to both splits.
    """
    random.seed(seed)
    split_map = {}
    for folder, group in df.groupby('img_folder'):
        rows = list(group.itertuples())
        random.shuffle(rows)
        n_val = max(1, int(len(rows) * val_ratio))
        for i, row in enumerate(rows):
            split = 'val' if i < n_val else 'train'
            split_map[(row.img_folder, row.img_name)] = split
    return split_map


# Geometry

def board_bbox_normalised(bbox_raw, img_w: int, img_h: int) -> tuple:
    """
    Convert DeepDarts bbox [y1, y2, x1, x2] pixels -> YOLOv8 (cx,cy,w,h) 0-1.
    """
    y1, y2, x1, x2 = bbox_raw
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return cx, cy, w, h


def board_bbox_from_keypoints(cal_xy_norm, pad: float = 0.08) -> tuple:
    """Fallback: derive board bbox from calibration keypoints."""
    xs, ys = cal_xy_norm[:, 0], cal_xy_norm[:, 1]
    cx = (xs.min() + xs.max()) / 2
    cy = (ys.min() + ys.max()) / 2
    w  = (xs.max() - xs.min()) * (1 + pad)
    h  = (ys.max() - ys.min()) * (1 + pad)
    return cx, cy, w, h


def dart_bbox_normalised(tip_norm, box_size: float = 0.07) -> tuple:
    """Small fixed-size bbox centred on dart tip."""
    return tip_norm[0], tip_norm[1], box_size, box_size


# Core converter

def convert(
    labels_path: str,
    images_root: str,
    out_root: str,
    dataset: str = 'both',
    val_ratio: float = 0.2,
    dart_box_size: float = 0.07,
):
    print(f"Loading {labels_path} ...")
    df = pd.read_pickle(labels_path)
    print(f"Columns: {list(df.columns)}")
    print(f"Total rows: {len(df)}")

    # Filter by dataset
    if dataset != 'both':
        df = df[df['img_folder'].str.startswith(dataset)]
        print(f"Rows after filtering for '{dataset}': {len(df)}")

    # Drop unannotated rows
    df = df[df['xy'].notna()]
    print(f"Rows with annotations: {len(df)}")

    # Build per-folder split map
    split_map = build_split_map(df, val_ratio=val_ratio)
    n_val   = sum(1 for v in split_map.values() if v == 'val')
    n_train = sum(1 for v in split_map.values() if v == 'train')
    print(f"Split: {n_train} train / {n_val} val (per-folder {int((1-val_ratio)*100)}/{int(val_ratio*100)})")

    # Create output dirs
    out = Path(out_root)
    for split in ('train', 'val'):
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)

    stats = {'train': 0, 'val': 0, 'skipped': 0, 'no_bbox': 0, 'darts': 0}

    for _, row in df.iterrows():
        folder   = row['img_folder']
        img_name = row['img_name']
        xy_raw   = row['xy']
        bbox_raw = row.get('bbox')

        # Find image on disk
        src = Path(images_root) / folder / img_name
        if not src.exists():
            stats['skipped'] += 1
            continue

        # Get image dimensions
        img = Image.open(src)
        img_w, img_h = img.size

        # Parse xy
        xy = np.array(xy_raw, dtype=np.float32)
        if xy.ndim == 1:
            xy = xy.reshape(-1, 2)

        cal_pts    = xy[:4]
        dart_pts   = xy[4:] if len(xy) > 4 else np.zeros((0, 2))
        valid_darts = [d for d in dart_pts if not (d[0] == 0 and d[1] == 0)]
        stats['darts'] += len(valid_darts)

        lines = []

        # Class 0: board
        if bbox_raw is not None:
            bcx, bcy, bw, bh = board_bbox_normalised(bbox_raw, img_w, img_h)
        else:
            stats['no_bbox'] += 1
            bcx, bcy, bw, bh = board_bbox_from_keypoints(cal_pts)

        kp_str = ' '.join(f"{kp[0]:.6f} {kp[1]:.6f} 1" for kp in cal_pts)
        lines.append(f"0 {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f} {kp_str}")

        # Class 1: dart (tip kp + 3 zero-padded to match kpt_shape=4)
        for tip in valid_darts:
            dcx, dcy, dw, dh = dart_bbox_normalised(tip, dart_box_size)
            tip_kps = (f"{tip[0]:.6f} {tip[1]:.6f} 1 "
                       f"0.000000 0.000000 0 "
                       f"0.000000 0.000000 0 "
                       f"0.000000 0.000000 0")
            lines.append(f"1 {dcx:.6f} {dcy:.6f} {dw:.6f} {dh:.6f} {tip_kps}")

        # Write label and copy image
        split = split_map[(folder, img_name)]
        (out / 'labels' / split / (src.stem + '.txt')).write_text(
            '\n'.join(lines), encoding='utf-8')
        shutil.copy(src, out / 'images' / split / img_name)
        stats[split] += 1

    print(f"\nConversion complete.")
    print(f"   train images : {stats['train']}")
    print(f"   val images   : {stats['val']}")
    print(f"   total darts  : {stats['darts']}")
    print(f"   skipped      : {stats['skipped']}  (image not found on disk)")
    print(f"   bbox fallback: {stats['no_bbox']}  (bbox derived from keypoints)")


# YAML

def write_yaml(out_root: str) -> str:
    content = f"""# DeepDarts -> YOLOv8 Pose
path: {Path(out_root).resolve()}
train: images/train
val:   images/val

nc: 2
names: ['board', 'dart']

# 4 keypoints per instance, 3 dims each (x, y, visibility)
# board: kp0-3 = calibration points
# dart:  kp0 = tip, kp1-3 = padding zeros
kpt_shape: [4, 3]
"""
    yaml_path = Path(out_root) / 'dartboard_pose.yaml'
    yaml_path.write_text(content, encoding='utf-8')
    print(f"YAML written to {yaml_path}")
    return str(yaml_path)


# CLI

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--labels',    required=True,
                   help='Path to labels.pkl')
    p.add_argument('--images',    required=True,
                   help='Path to cropped_images/800')
    p.add_argument('--out',       default='data_yolo',
                   help='Output directory')
    p.add_argument('--dataset',   default='both', choices=['d1', 'd2', 'both'])
    p.add_argument('--val_ratio', type=float, default=0.2,
                   help='Fraction of each folder to use for val (default 0.2)')
    p.add_argument('--dart_box',  type=float, default=0.07)
    args = p.parse_args()

    convert(
        labels_path=args.labels,
        images_root=args.images,
        out_root=args.out,
        dataset=args.dataset,
        val_ratio=args.val_ratio,
        dart_box_size=args.dart_box,
    )
    write_yaml(args.out)

    print(f"""
Command:
    python train.py train --data {args.out}/dartboard_pose.yaml --epochs 100 --batch 16 --device cuda
""")