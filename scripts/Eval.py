import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("pip install ultralytics")

try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
           3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

RINGS = {
    'bull':         0.035,
    'outer_bull':   0.095,
    'triple_inner': 0.550,
    'triple_outer': 0.600,
    'double_inner': 0.950,
    'double_outer': 1.000,
}

CAL_NORM = np.array([
    [-0.156, -0.988],   # top    (5/20)
    [ 0.156,  0.988],   # bottom (17/3)
    [-0.988,  0.156],   # left   (8/11)
    [ 0.988, -0.156],   # right  (13/6)
], dtype=np.float32)


# ── Display constants ────────────────────────────────────────────────────────

WINDOW_NAME = 'DartBuddy Eval'
FONT = cv2.FONT_HERSHEY_DUPLEX

COLORS = {
    'background': (18, 18, 18),
    'panel_bg':   (0, 0, 0),
    'original':   (220, 220, 220),
    'model2':     (0, 200, 255),     # amber  - train2 / original model
    'model4':     (80, 220, 80),     # green  - train4 / custom data model
    'prompt':     (245, 245, 245),
    'muted':      (155, 155, 155),
    'accent':     (180, 255, 0),
    'danger':     (80, 80, 255),
    'line':       (55, 55, 55),
}


# ── Scoring helpers ──────────────────────────────────────────────────────────

def dart_score(nx: float, ny: float) -> str:
    dist = np.hypot(nx, ny)

    if dist <= RINGS['bull']:
        return 'Bull (50)'
    if dist <= RINGS['outer_bull']:
        return 'Outer Bull (25)'
    if dist > RINGS['double_outer']:
        return 'Miss'

    angle = (np.degrees(np.arctan2(nx, -ny)) + 360 + 9) % 360
    sector = SECTORS[int(angle / 18) % 20]

    if RINGS['triple_inner'] <= dist <= RINGS['triple_outer']:
        return f'T{sector}'
    if RINGS['double_inner'] <= dist <= RINGS['double_outer']:
        return f'D{sector}'

    return str(sector)


def estimate_homography(cal_pts_px, board_radius):
    if len(cal_pts_px) < 4:
        return None
    dst = CAL_NORM * board_radius
    try:
        H, _ = cv2.findHomography(
            np.array(cal_pts_px, dtype=np.float32),
            dst.astype(np.float32),
            cv2.RANSAC,
            5.0,
        )
        return H
    except Exception:
        return None


def pixel_to_board_norm(px, py, board_center, board_radius, H):
    if H is not None:
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, H)[0][0]
        return transformed[0] / board_radius, transformed[1] / board_radius
    cx, cy = board_center
    return (px - cx) / board_radius, (py - cy) / board_radius


# ── Image loading ────────────────────────────────────────────────────────────

def read_image_bgr(path: Path):
    if Image is not None and ImageOps is not None:
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img).convert('RGB')
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

    return cv2.imread(str(path))


# ── Inference ────────────────────────────────────────────────────────────────

def run_model(model, frame, conf=0.45, device='cuda'):
    results = model.predict(frame, conf=conf, device=device, verbose=False, imgsz=800)
    result = results[0]
    names = result.names

    board_center = None
    board_radius = None
    H = None
    darts = []

    # Pass 1: board
    for i, box in enumerate(result.boxes):
        if names[int(box.cls[0])] != 'board':
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        board_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        board_radius = max(x2 - x1, y2 - y1) / 2

        cal_pts = []
        if result.keypoints is not None and i < len(result.keypoints):
            kps = result.keypoints[i].xy[0].cpu().numpy()
            for kp in kps:
                if not (kp[0] == 0 and kp[1] == 0):
                    cal_pts.append(kp)

        if len(cal_pts) >= 4:
            H = estimate_homography(cal_pts[:4], board_radius)
        break

    # Pass 2: darts
    dart_idx = 0
    for i, box in enumerate(result.boxes):
        if names[int(box.cls[0])] != 'dart' or dart_idx >= 3:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        tip_px, tip_py = (x1 + x2) / 2, float(y2)

        if result.keypoints is not None and i < len(result.keypoints):
            kp = result.keypoints[i].xy[0].cpu().numpy()
            if kp.shape[0] > 0 and not (kp[0, 0] == 0 and kp[0, 1] == 0):
                tip_px, tip_py = float(kp[0, 0]), float(kp[0, 1])

        label = '?'
        if board_center and board_radius:
            nx, ny = pixel_to_board_norm(tip_px, tip_py, board_center, board_radius, H)
            label = dart_score(nx, ny)

        darts.append({
            'score': label,
            'conf': round(float(box.conf[0]), 2),
            'tip': (int(tip_px), int(tip_py)),
            'bbox': (x1, y1, x2, y2),
        })
        dart_idx += 1

    return darts, board_center is not None


# ── Drawing helpers ──────────────────────────────────────────────────────────

def fit_text(text, max_chars=70):
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + '...'


def draw_text(img, text, x, y, scale=0.55, color=None, thickness=1):
    if color is None:
        color = COLORS['prompt']
    cv2.putText(img, str(text), (int(x), int(y)), FONT, scale, color, thickness, cv2.LINE_AA)


def draw_lines(img, lines, x, y, line_gap=27, scale=0.55, color=None):
    for i, line in enumerate(lines):
        draw_text(img, line, x, y + i * line_gap, scale=scale, color=color)


def letterbox(frame, panel_w, panel_h):
    """Resize frame into a fixed-size panel without stretching it."""
    h, w = frame.shape[:2]
    scale = min(panel_w / w, panel_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    panel = np.full((panel_h, panel_w, 3), COLORS['panel_bg'], dtype=np.uint8)
    x0 = (panel_w - new_w) // 2
    y0 = (panel_h - new_h) // 2
    panel[y0:y0 + new_h, x0:x0 + new_w] = resized
    return panel, scale, x0, y0


def scale_point(pt, scale, x0, y0):
    x, y = pt
    return int(round(x * scale + x0)), int(round(y * scale + y0))


def scale_bbox(bbox, scale, x0, y0):
    x1, y1, x2, y2 = bbox
    sx1, sy1 = scale_point((x1, y1), scale, x0, y0)
    sx2, sy2 = scale_point((x2, y2), scale, x0, y0)
    return sx1, sy1, sx2, sy2


def draw_panel_title(panel, title, color):
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (0, 0, 0), -1)
    draw_text(panel, title, 12, 28, scale=0.6, color=color, thickness=1)


def draw_predictions(panel, darts, color, scale, x0, y0):
    for i, d in enumerate(darts):
        sx1, sy1, sx2, sy2 = scale_bbox(d['bbox'], scale, x0, y0)
        tx, ty = scale_point(d['tip'], scale, x0, y0)

        cv2.rectangle(panel, (sx1, sy1), (sx2, sy2), color, 2)
        cv2.circle(panel, (tx, ty), 6, color, -1)
        cv2.circle(panel, (tx, ty), 6, (255, 255, 255), 1)

        text = f"#{i + 1} {d['score']} ({d['conf']:.0%})"
        text_x = min(max(tx + 8, 8), panel.shape[1] - 150)
        text_y = min(max(ty - 8, 55), panel.shape[0] - 10)
        draw_text(panel, text, text_x, text_y, scale=0.48, color=color, thickness=1)


def make_image_panel(frame, darts, title, color, panel_w, panel_h, show_detections=True):
    panel, scale, x0, y0 = letterbox(frame, panel_w, panel_h)
    draw_panel_title(panel, title, color)
    if show_detections:
        draw_predictions(panel, darts, color, scale, x0, y0)
    return panel


def score_list_text(label, darts):
    scores = [d['score'] for d in darts]
    if not scores:
        scores = ['none detected']
    return f"{label}: {', '.join(scores)}"


def build_base_display(frame, darts2, darts4, img_idx, total, img_path, elapsed_ms, panel_w, panel_h):
    display_w = panel_w * 3
    header_h = 126

    header = np.full((header_h, display_w, 3), COLORS['background'], dtype=np.uint8)
    draw_text(header, f"Image {img_idx}/{total} | {fit_text(Path(img_path).name, 85)}",
              12, 30, scale=0.65, color=COLORS['prompt'])
    draw_text(header, f"Inference: {elapsed_ms:.0f} ms | Original image shown on left | Inputs use number keys in this popup",
              12, 60, scale=0.52, color=COLORS['muted'])
    draw_text(header, score_list_text('Train2', darts2), 12, 92, scale=0.50, color=COLORS['model2'])
    draw_text(header, score_list_text('Train4', darts4), display_w // 2, 92, scale=0.50, color=COLORS['model4'])

    original = make_image_panel(
        frame, [], 'Original image (no detections)', COLORS['original'], panel_w, panel_h, show_detections=False
    )
    model2 = make_image_panel(
        frame, darts2, 'Model A: train2 / original', COLORS['model2'], panel_w, panel_h, show_detections=True
    )
    model4 = make_image_panel(
        frame, darts4, 'Model B: train4 / custom data', COLORS['model4'], panel_w, panel_h, show_detections=True
    )

    divider = np.full((panel_h, 2, 3), COLORS['line'], dtype=np.uint8)
    image_row = np.hstack([original, divider, model2, divider, model4])

    # Make the header match the row width after the dividers are added.
    if header.shape[1] != image_row.shape[1]:
        pad_w = image_row.shape[1] - header.shape[1]
        if pad_w > 0:
            header = np.hstack([header, np.full((header_h, pad_w, 3), COLORS['background'], dtype=np.uint8)])
        else:
            header = header[:, :image_row.shape[1]]

    return np.vstack([header, image_row])


def render_prompt_display(base_display, prompt, status_lines, allowed_values, error_message=None):
    prompt_h = 154
    h, w = base_display.shape[:2]
    panel = np.full((prompt_h, w, 3), COLORS['background'], dtype=np.uint8)

    cv2.line(panel, (0, 0), (w, 0), COLORS['line'], 1)
    draw_text(panel, prompt, 12, 32, scale=0.66, color=COLORS['accent'], thickness=1)

    key_text = f"Press: {', '.join(str(v) for v in allowed_values)}    |    S = skip image    |    Q / Esc = quit + summarize"
    draw_text(panel, key_text, 12, 65, scale=0.54, color=COLORS['prompt'], thickness=1)

    detail_y = 96
    if error_message:
        draw_text(panel, error_message, 12, 96, scale=0.54, color=COLORS['danger'], thickness=1)
        detail_y = 126

    draw_lines(panel, status_lines, 12, detail_y, line_gap=23, scale=0.48, color=COLORS['muted'])
    return np.vstack([base_display, panel])


def resize_for_screen(display, max_w=1800, max_h=1000):
    h, w = display.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        display = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return display


# ── Popup input ──────────────────────────────────────────────────────────────

def wait_for_count(base_display, prompt, min_value, max_value, status_lines, max_window_w, max_window_h):
    allowed_values = list(range(min_value, max_value + 1))
    error_message = None

    while True:
        display = render_prompt_display(
            base_display,
            prompt=prompt,
            status_lines=status_lines,
            allowed_values=allowed_values,
            error_message=error_message,
        )
        shown = resize_for_screen(display, max_window_w, max_window_h)
        cv2.imshow(WINDOW_NAME, shown)

        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            return 'quit'
        if key in (ord('s'), ord('S')):
            return 'skip'

        ch = chr(key) if 48 <= key <= 57 else ''
        if ch:
            value = int(ch)
            if min_value <= value <= max_value:
                return value

        error_message = f"Invalid key. Enter a number from {min_value} to {max_value}."


def collect_image_scores(base_display, args):

    if args.darts_per_image is None:
        real_darts = wait_for_count(
            base_display,
            prompt="How many real darts are present in this image?",
            min_value=0,
            max_value=3,
            status_lines=status,
            max_window_w=args.max_window_width,
            max_window_h=args.max_window_height,
        )
        if real_darts in ('skip', 'quit'):
            return real_darts
    else:
        real_darts = args.darts_per_image

    status2 = [
        f"Real darts for this image: {real_darts}",
        "Step 2/3: enter how many Train2 scored correctly.",
    ]
    correct_model2 = wait_for_count(
        base_display,
        prompt="How many darts did Train2 get right?",
        min_value=0,
        max_value=real_darts,
        status_lines=status2,
        max_window_w=args.max_window_width,
        max_window_h=args.max_window_height,
    )
    if correct_model2 in ('skip', 'quit'):
        return correct_model2

    status3 = [
        f"Real darts: {real_darts}    Train2 correct: {correct_model2}/{real_darts}",
        "Step 3/3: enter how many Train4 scored correctly.",
    ]
    correct_model4 = wait_for_count(
        base_display,
        prompt="How many darts did Train4 get right?",
        min_value=0,
        max_value=real_darts,
        status_lines=status3,
        max_window_w=args.max_window_width,
        max_window_h=args.max_window_height,
    )
    if correct_model4 in ('skip', 'quit'):
        return correct_model4

    return real_darts, correct_model2, correct_model4


# ── Results summary ─────────────────────────────────────────────────────────

def model_stats(results, key):
    reviewed = [r for r in results if not r['skipped']]
    images = len(reviewed)
    darts_reviewed = sum(r['real_darts'] for r in reviewed)
    darts_correct = sum(r[f'correct_{key}'] for r in reviewed)
    darts_wrong = darts_reviewed - darts_correct
    dart_accuracy = (darts_correct / darts_reviewed * 100) if darts_reviewed else 0.0
    avg_correct = (darts_correct / images) if images else 0.0
    avg_real_darts = (darts_reviewed / images) if images else 0.0
    perfect_images = sum(
        1 for r in reviewed
        if r['real_darts'] > 0 and r[f'correct_{key}'] == r['real_darts']
    )

    return {
        'images': images,
        'darts_reviewed': darts_reviewed,
        'darts_correct': darts_correct,
        'darts_wrong': darts_wrong,
        'dart_accuracy': dart_accuracy,
        'avg_correct': avg_correct,
        'avg_real_darts': avg_real_darts,
        'perfect_images': perfect_images,
    }


def print_results(results, model2_name, model4_name):
    print("\n" + "=" * 78)
    print("  DART SCORING EVALUATION RESULTS")
    print("=" * 78)

    skipped = sum(1 for r in results if r['skipped'])
    reviewed = [r for r in results if not r['skipped']]
    real_darts_reviewed = sum(r['real_darts'] for r in reviewed)

    print(f"\n  Images reviewed      : {len(reviewed)}")
    print(f"  Skipped images       : {skipped}")
    print(f"  Real darts reviewed  : {real_darts_reviewed}")
    if reviewed:
        print(f"  Avg real darts/image : {real_darts_reviewed / len(reviewed):.2f}")

    stats2 = model_stats(results, 'model2')
    stats4 = model_stats(results, 'model4')

    for stats, name in [(stats2, model2_name), (stats4, model4_name)]:
        if stats['images'] == 0:
            print(f"\n  {name}: no images reviewed")
            continue

        print(f"\n  {name}")
        print(f"    Darts reviewed      : {stats['darts_reviewed']}")
        print(f"    Darts correct       : {stats['darts_correct']}  ({stats['dart_accuracy']:.1f}%)")
        print(f"    Darts wrong         : {stats['darts_wrong']}  ({100 - stats['dart_accuracy']:.1f}%)")
        print(f"    Avg correct/image   : {stats['avg_correct']:.2f}/{stats['avg_real_darts']:.2f}")
        print(f"    Perfect images      : {stats['perfect_images']}/{stats['images']}")

    if reviewed:
        delta = stats4['darts_correct'] - stats2['darts_correct']
        delta_pct = stats4['dart_accuracy'] - stats2['dart_accuracy']
        print("\n  Model comparison")
        print(f"    Train4 - Train2 correct-dart difference : {delta:+d}")
        print(f"    Train4 - Train2 accuracy difference     : {delta_pct:+.1f} percentage points")

        print("\n  Per-image breakdown:")
        print(f"  {'#':<4} {'File':<35} {'Darts':>5} {'Train2':>10} {'Train4':>10} {'Δ':>5}")
        print("  " + "-" * 75)
        for i, r in enumerate(reviewed, 1):
            a = r['correct_model2']
            b = r['correct_model4']
            d = r['real_darts']
            print(f"  {i:<4} {fit_text(Path(r['image']).name, 35):<35} {d:>5} {a:>5}/{d:<4} {b:>5}/{d:<4} {b-a:>+5}")

    print("=" * 78)


def accuracy_percent(correct, total):
    if total == 0:
        return ''
    return round(correct / total * 100, 2)


def save_results_csv(results, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'image',
            'skipped',
            'real_darts',
            'train2_predictions',
            'train4_predictions',
            'train2_correct_darts',
            'train4_correct_darts',
            'train2_accuracy_percent',
            'train4_accuracy_percent',
            'train4_minus_train2_correct_darts',
        ])
        writer.writeheader()

        for r in results:
            skipped = r['skipped']
            real_darts = '' if skipped else r['real_darts']
            c2 = '' if skipped else r['correct_model2']
            c4 = '' if skipped else r['correct_model4']
            writer.writerow({
                'image': r['image'],
                'skipped': skipped,
                'real_darts': real_darts,
                'train2_predictions': ', '.join(r['darts_model2']),
                'train4_predictions': ', '.join(r['darts_model4']),
                'train2_correct_darts': c2,
                'train4_correct_darts': c4,
                'train2_accuracy_percent': '' if skipped else accuracy_percent(c2, real_darts),
                'train4_accuracy_percent': '' if skipped else accuracy_percent(c4, real_darts),
                'train4_minus_train2_correct_darts': '' if skipped else c4 - c2,
            })

    print(f"\nSaved CSV results to: {out_path}")


def append_skipped(results, img_path, darts2, darts4):
    results.append({
        'image': str(img_path),
        'darts_model2': [d['score'] for d in darts2],
        'darts_model4': [d['score'] for d in darts4],
        'real_darts': 0,
        'correct_model2': 0,
        'correct_model4': 0,
        'skipped': True,
    })


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model2', required=True, help='Path to train2 best.pt')
    parser.add_argument('--model4', required=True, help='Path to train4 best.pt')
    parser.add_argument('--images', required=True, help='Directory of evaluation images')
    parser.add_argument('--n', type=int, default=30, help='Number of images to evaluate')
    parser.add_argument('--conf', type=float, default=0.45)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional random seed. Leave unset for a different random sample each run.')
    parser.add_argument('--darts-per-image', type=int, default=None,
                        help='Optional fixed number of real darts in every image. Leave unset for mixed 0/1/2/3-dart datasets.')
    parser.add_argument('--out', default='eval_results.csv',
                        help='CSV file to save detailed results for your report.')
    parser.add_argument('--panel-width', type=int, default=520,
                        help='Width of each image panel in the popup.')
    parser.add_argument('--panel-height', type=int, default=560,
                        help='Height of each image panel in the popup.')
    parser.add_argument('--max-window-width', type=int, default=1800,
                        help='Maximum displayed popup width.')
    parser.add_argument('--max-window-height', type=int, default=1000,
                        help='Maximum displayed popup height.')
    args = parser.parse_args()

    if args.n <= 0:
        raise SystemExit('--n must be greater than 0')
    if args.darts_per_image is not None and not (0 <= args.darts_per_image <= 3):
        raise SystemExit('--darts-per-image must be between 0 and 3')
    if args.panel_width < 250 or args.panel_height < 250:
        raise SystemExit('--panel-width and --panel-height should both be at least 250')

    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    all_images = sorted(p for p in Path(args.images).rglob('*') if p.suffix.lower() in exts)
    if not all_images:
        raise SystemExit(f"No images found in {args.images}")

    rng = random.Random(args.seed)
    sample = rng.sample(all_images, min(args.n, len(all_images)))

    if args.seed is None:
        print(f"Sampled {len(sample)} random images from {args.images}")
    else:
        print(f"Sampled {len(sample)} images from {args.images} with seed {args.seed}")

    print("Loading models...")
    model2 = YOLO(args.model2)
    model4 = YOLO(args.model4)
    print("Models loaded.\n")

    print("Popup controls:")
    print("  0-3  enter count")
    print("  S    skip image")
    print("  Q    quit and summarize current results")
    print("  Esc  quit and summarize current results")
    print("\nThe left panel is the original image with no detections.\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    results = []

    for idx, img_path in enumerate(sample, 1):
        frame = read_image_bgr(img_path)
        if frame is None:
            print(f"  Could not read {img_path}, skipping.")
            continue

        t0 = time.perf_counter()
        darts2, _ = run_model(model2, frame, args.conf, args.device)
        darts4, _ = run_model(model4, frame, args.conf, args.device)
        elapsed = (time.perf_counter() - t0) * 1000

        base_display = build_base_display(
            frame=frame,
            darts2=darts2,
            darts4=darts4,
            img_idx=idx,
            total=len(sample),
            img_path=img_path,
            elapsed_ms=elapsed,
            panel_w=args.panel_width,
            panel_h=args.panel_height,
        )

        print(f"\n[{idx}/{len(sample)}] {img_path.name}  (inference: {elapsed:.0f} ms)")
        print("  Original image is shown in the left panel with no detections.")
        print(f"  Model A / train2 darts: {[d['score'] for d in darts2] or '(none detected)'}")
        print(f"  Model B / train4 darts: {[d['score'] for d in darts4] or '(none detected)'}")

        score_result = collect_image_scores(base_display, args)

        if score_result == 'quit':
            print_results(results, 'Model A: train2 / original', 'Model B: train4 / custom data')
            save_results_csv(results, args.out)
            cv2.destroyAllWindows()
            return

        if score_result == 'skip':
            print("  skipped")
            append_skipped(results, img_path, darts2, darts4)
            continue

        real_darts, count_model2, count_model4 = score_result
        results.append({
            'image': str(img_path),
            'darts_model2': [d['score'] for d in darts2],
            'darts_model4': [d['score'] for d in darts4],
            'real_darts': real_darts,
            'correct_model2': count_model2,
            'correct_model4': count_model4,
            'skipped': False,
        })

        print(f"  recorded: real darts={real_darts}, train2={count_model2}/{real_darts}, train4={count_model4}/{real_darts}")

    cv2.destroyAllWindows()
    print_results(results, 'Model A: train2 / original', 'Model B: train4 / custom data')
    save_results_csv(results, args.out)


if __name__ == '__main__':
    main()
