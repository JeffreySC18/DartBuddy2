"""
convert_deepdarts_to_yolov8.py
Convert DeepDarts labels.pkl -> YOLOv8 keypoint (pose) format.

Key fix: The raw bbox in labels.pkl is in original full-resolution photo
pixel coordinates (~4000x3000), but the cropped images and xy keypoints
are 800x800 normalised. Using the raw bbox produces values like cx=1.9,
w=1.8 which are way outside 0-1 and cause the board class to never be
detected. Solution: always derive the board bbox from the 4 calibration
keypoints, which are already correctly normalised to the cropped image.

Usage:
    python converter.py --labels labels.pkl --images Megafolder/cropped_images/800 --out data_yolo
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def build_split_map(df, val_ratio=0.2, seed=42):
    """
    80/20 split per-image within each folder, so every folder type
    (empty board, 1 dart, 2 darts, 3 darts) appears in both splits.
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


def board_bbox_from_keypoints(cal_xy_norm, pad=0.08):
    """
    Derive board bounding box from the 4 calibration keypoints.
    cal_xy_norm: shape (4, 2), values normalised 0-1.
    Returns (cx, cy, w, h) normalised 0-1.
    """
    xs, ys = cal_xy_norm[:, 0], cal_xy_norm[:, 1]
    cx = float((xs.min() + xs.max()) / 2)
    cy = float((ys.min() + ys.max()) / 2)
    w  = float((xs.max() - xs.min()) * (1 + pad))
    h  = float((ys.max() - ys.min()) * (1 + pad))
    return cx, cy, w, h


def clamp(v):
    return max(0.0, min(1.0, float(v)))


def convert(labels_path, images_root, out_root, dataset='both',
            val_ratio=0.2, dart_box_size=0.07):

    print(f"Loading {labels_path} ...")
    df = pd.read_pickle(labels_path)
    print(f"Columns: {list(df.columns)}")
    print(f"Total rows: {len(df)}")

    if dataset != 'both':
        df = df[df['img_folder'].str.startswith(dataset)]
        print(f"Rows after filtering for '{dataset}': {len(df)}")

    df = df[df['xy'].notna()]
    print(f"Rows with annotations: {len(df)}")

    split_map = build_split_map(df, val_ratio=val_ratio)
    n_val   = sum(1 for v in split_map.values() if v == 'val')
    n_train = sum(1 for v in split_map.values() if v == 'train')
    print(f"Split: {n_train} train / {n_val} val")

    out = Path(out_root)
    for split in ('train', 'val'):
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)

    stats = {'train': 0, 'val': 0, 'skipped': 0, 'darts': 0}

    for _, row in df.iterrows():
        folder   = row['img_folder']
        img_name = row['img_name']
        xy_raw   = row['xy']

        src = Path(images_root) / folder / img_name
        if not src.exists():
            stats['skipped'] += 1
            continue

        # Parse xy - normalised 0-1 relative to the 800x800 cropped image
        xy = np.array(xy_raw, dtype=np.float32)
        if xy.ndim == 1:
            xy = xy.reshape(-1, 2)

        cal_pts    = xy[:4]
        dart_pts   = xy[4:] if len(xy) > 4 else np.zeros((0, 2))
        valid_darts = [d for d in dart_pts if not (d[0] == 0 and d[1] == 0)]
        stats['darts'] += len(valid_darts)

        lines = []

        # Class 0: board
        # Derive bbox entirely from calibration keypoints.
        bcx, bcy, bw, bh = board_bbox_from_keypoints(cal_pts)

        # Clamp everything to 0-1
        bcx, bcy, bw, bh = clamp(bcx), clamp(bcy), clamp(bw), clamp(bh)
        kp_str = ' '.join(
            f"{clamp(kp[0]):.6f} {clamp(kp[1]):.6f} 1"
            for kp in cal_pts
        )
        lines.append(f"0 {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f} {kp_str}")

        # Class 1: dart
        # 1 tip keypoint + 3 zero-padded to satisfy kpt_shape=[4,3]
        for tip in valid_darts:
            dcx, dcy = clamp(tip[0]), clamp(tip[1])
            tip_kps = (f"{dcx:.6f} {dcy:.6f} 1 "
                       f"0.000000 0.000000 0 "
                       f"0.000000 0.000000 0 "
                       f"0.000000 0.000000 0")
            lines.append(
                f"1 {dcx:.6f} {dcy:.6f} "
                f"{dart_box_size:.6f} {dart_box_size:.6f} {tip_kps}"
            )

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


def write_yaml(out_root):
    content = f"""
path: {Path(out_root).resolve()}
train: images/train
val:   images/val

nc: 2
names: ['board', 'dart']

# 4 keypoints per instance, 3 dims each (x, y, visibility)
# board: kp0-3 = calibration points (top, right, bottom, left)
# dart:  kp0 = tip, kp1-3 = padding zeros
kpt_shape: [4, 3]

# Horizontal flip remapping: top stays, right<->left swap, bottom stays
flip_idx: [0, 3, 2, 1]
"""
    yaml_path = Path(out_root) / 'dartboard_pose.yaml'
    yaml_path.write_text(content, encoding='utf-8')
    print(f"YAML written to {yaml_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--labels',    required=True)
    p.add_argument('--images',    required=True)
    p.add_argument('--out',       default='data_yolo')
    p.add_argument('--dataset',   default='both', choices=['d1', 'd2', 'both'])
    p.add_argument('--val_ratio', type=float, default=0.2)
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
Next step:
    python train.py train --data {args.out}/dartboard_pose.yaml --epochs 100 --batch 16 --device cuda
""")