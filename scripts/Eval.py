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


# ── Dartboard geometry (same as your scoring.py) ────────────────────────────

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


WINDOW_NAME = 'DartBuddy Eval'
FONT = cv2.FONT_HERSHEY_DUPLEX


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
            cv2.RANSAC, 5.0,
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


# ── Inference ────────────────────────────────────────────────────────────────

def run_model(model, frame, conf=0.45, device='cuda'):
    results = model.predict(frame, conf=conf, device=device, verbose=False, imgsz=800)
    result  = results[0]
    names   = result.names

    board_center = None
    board_radius = None
    H            = None
    darts        = []

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
            'conf':  round(float(box.conf[0]), 2),
            'tip':   (int(tip_px), int(tip_py)),
            'bbox':  (x1, y1, x2, y2),
        })
        dart_idx += 1

    return darts, board_center is not None


# ── Drawing ──────────────────────────────────────────────────────────────────

COLORS = {
    'original': (220, 220, 220),
    'model2':   (0,   200, 255),   # amber  - train2 / original
    'model4':   (80,  220, 80),    # green  - train4 / custom data
    'prompt':   (245, 245, 245),
    'muted':    (155, 155, 155),
    'accent':   (180, 255, 0),
    'danger':   (80, 80, 255),
}


def annotate(frame, darts, color, label_prefix):
    vis = frame.copy()
    for i, d in enumerate(darts):
        tx, ty = d['tip']
        x1, y1, x2, y2 = d['bbox']
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.circle(vis, (tx, ty), 6, color, -1)
        cv2.circle(vis, (tx, ty), 6, (255, 255, 255), 1)
        text = f"#{i + 1} {d['score']} ({d['conf']:.0%})"
        cv2.putText(vis, text, (tx + 8, ty - 8),
                    FONT, 0.55, color, 1, cv2.LINE_AA)
    cv2.putText(vis, label_prefix, (10, 28),
                FONT, 0.8, color, 2, cv2.LINE_AA)
    return vis


def label_original(frame):
    vis = frame.copy()
    cv2.putText(vis, 'Original image (no detections)', (10, 28),
                FONT, 0.8, COLORS['original'], 2, cv2.LINE_AA)
    return vis


def build_base_display(frame, darts2, darts4, img_idx, total, img_path, elapsed_ms):
    h, w = frame.shape[:2]

    original = label_original(frame)
    model2   = annotate(frame, darts2, COLORS['model2'], 'Model A: train2 / original')
    model4   = annotate(frame, darts4, COLORS['model4'], 'Model B: train4 / custom data')

    combined = np.hstack([original, model2, model4])

    bar = np.zeros((68, w * 3, 3), dtype=np.uint8)
    cv2.putText(bar, f"Image {img_idx}/{total}  |  {Path(img_path).name}",
                (10, 27), FONT, 0.65, COLORS['prompt'], 1, cv2.LINE_AA)
    cv2.putText(bar, f"Inference: {elapsed_ms:.0f} ms  |  Inputs are entered in this popup with number keys",
                (10, 55), FONT, 0.55, COLORS['muted'], 1, cv2.LINE_AA)

    return np.vstack([bar, combined])


def score_list_text(label, darts):
    scores = [d['score'] for d in darts]
    if not scores:
        scores = ['none detected']
    return f"{label}: {', '.join(scores)}"


def put_text_lines(img, lines, x, y, line_gap=27, scale=0.62, color=None):
    if color is None:
        color = COLORS['prompt']
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * line_gap), FONT, scale, color, 1, cv2.LINE_AA)


def render_prompt_display(base_display, prompt, status_lines, allowed_values, error_message=None):
    prompt_h = 154
    h, w = base_display.shape[:2]
    panel = np.zeros((prompt_h, w, 3), dtype=np.uint8)

    cv2.putText(panel, prompt, (10, 31), FONT, 0.68, COLORS['accent'], 1, cv2.LINE_AA)

    key_text = f"Press one of: {', '.join(str(v) for v in allowed_values)}    |    S = skip image    |    Q / Esc = quit"
    cv2.putText(panel, key_text, (10, 65), FONT, 0.55, COLORS['prompt'], 1, cv2.LINE_AA)

    if error_message:
        cv2.putText(panel, error_message, (10, 96), FONT, 0.55, COLORS['danger'], 1, cv2.LINE_AA)
        details_y = 126
    else:
        details_y = 98

    put_text_lines(panel, status_lines, 10, details_y, line_gap=24, scale=0.5, color=COLORS['muted'])

    return np.vstack([base_display, panel])


def resize_for_screen(display, max_w=1800, max_h=1000):
    h, w = display.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        display = cv2.resize(display, (int(w * scale), int(h * scale)))
    return display


# ── Popup input ──────────────────────────────────────────────────────────────

def wait_for_count(base_display, prompt, min_value, max_value, status_lines):
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
        cv2.imshow(WINDOW_NAME, resize_for_screen(display))

        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            return 'quit'
        if key in (ord('s'), ord('S')):
            return 'skip'

        ch = chr(key) if key < 128 else ''
        if ch.isdigit():
            value = int(ch)
            if min_value <= value <= max_value:
                return value

        error_message = f"Invalid input. Use a number from {min_value} to {max_value}, S, Q, or Esc."


def collect_popup_scores(base_display, darts2, darts4, fixed_darts_per_image):
    status_lines = [
        score_list_text('Train2 predictions', darts2),
        score_list_text('Train4 predictions', darts4),
    ]

    if fixed_darts_per_image is None:
        real_darts = wait_for_count(
            base_display,
            prompt='Step 1/3: How many real darts are present in the original image?',
            min_value=0,
            max_value=3,
            status_lines=status_lines,
        )
        if real_darts in {'skip', 'quit'}:
            return real_darts, None, None
    else:
        real_darts = fixed_darts_per_image

    status_lines_with_real = [
        f'Real darts selected: {real_darts}',
        score_list_text('Train2 predictions', darts2),
        score_list_text('Train4 predictions', darts4),
    ]

    step_a = 'Step 2/3' if fixed_darts_per_image is None else 'Step 1/2'
    correct_model2 = wait_for_count(
        base_display,
        prompt=f'{step_a}: How many darts did Train2 score correctly? 0-{real_darts}',
        min_value=0,
        max_value=real_darts,
        status_lines=status_lines_with_real,
    )
    if correct_model2 in {'skip', 'quit'}:
        return correct_model2, real_darts, None

    status_lines_with_a = [
        f'Real darts selected: {real_darts}',
        f'Train2 correct darts selected: {correct_model2}/{real_darts}',
        score_list_text('Train4 predictions', darts4),
    ]

    step_b = 'Step 3/3' if fixed_darts_per_image is None else 'Step 2/2'
    correct_model4 = wait_for_count(
        base_display,
        prompt=f'{step_b}: How many darts did Train4 score correctly? 0-{real_darts}',
        min_value=0,
        max_value=real_darts,
        status_lines=status_lines_with_a,
    )
    if correct_model4 in {'skip', 'quit'}:
        return correct_model4, real_darts, correct_model2

    return 'ok', real_darts, correct_model2, correct_model4


# ── Results summary ──────────────────────────────────────────────────────────

def model_stats(results, key):
    reviewed = [r for r in results if not r['skipped']]
    images = len(reviewed)
    darts_reviewed = sum(r['real_darts'] for r in reviewed)
    darts_correct = sum(r[f'correct_{key}'] for r in reviewed)
    darts_wrong = darts_reviewed - darts_correct
    dart_accuracy = (darts_correct / darts_reviewed * 100) if darts_reviewed else 0.0
    avg_correct = (darts_correct / images) if images else 0.0
    avg_real_darts = (darts_reviewed / images) if images else 0.0
    perfect_images = sum(1 for r in reviewed if r[f'correct_{key}'] == r['real_darts'])

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
            print(f"  {i:<4} {Path(r['image']).name:<35} {d:>5} {a:>5}/{d:<4} {b:>5}/{d:<4} {b-a:>+5}")

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


def skipped_result(img_path, darts2, darts4, real_darts=0):
    return {
        'image':          str(img_path),
        'darts_model2':   [d['score'] for d in darts2],
        'darts_model4':   [d['score'] for d in darts4],
        'real_darts':     real_darts,
        'correct_model2': 0,
        'correct_model4': 0,
        'skipped':        True,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model2', required=True, help='Path to train2 best.pt')
    p.add_argument('--model4', required=True, help='Path to train4 best.pt')
    p.add_argument('--images', required=True, help='Directory of evaluation images')
    p.add_argument('--n',      type=int, default=30, help='Number of images to evaluate')
    p.add_argument('--conf',   type=float, default=0.45)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed',   type=int, default=None,
                   help='Optional random seed. Leave unset for a different random sample each run.')
    p.add_argument('--darts-per-image', type=int, default=None,
                   help='Optional fixed number of real darts in every image. Leave unset for mixed 1/2/3-dart datasets.')
    p.add_argument('--out', default='eval_results.csv',
                   help='CSV file to save detailed results for your report.')
    args = p.parse_args()

    if args.n <= 0:
        raise SystemExit('--n must be greater than 0')
    if args.darts_per_image is not None and not (0 <= args.darts_per_image <= 3):
        raise SystemExit('--darts-per-image must be between 0 and 3')

    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_images = sorted(
        p for p in Path(args.images).rglob('*') if p.suffix.lower() in exts
    )
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

    print("Use the OpenCV popup window for all evaluation inputs.")
    print("Click the popup if key presses are not registering.")
    print("Keys: 0-3 for counts, S to skip image, Q or Esc to quit and summarize current results.\n")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    results = []

    for idx, img_path in enumerate(sample, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  Could not read {img_path}, skipping.")
            continue

        t0 = time.perf_counter()
        darts2, board2 = run_model(model2, frame, args.conf, args.device)
        darts4, board4 = run_model(model4, frame, args.conf, args.device)
        elapsed = (time.perf_counter() - t0) * 1000

        base_display = build_base_display(frame, darts2, darts4, idx, len(sample), img_path, elapsed)

        print(f"\n[{idx}/{len(sample)}] {img_path.name}  (inference: {elapsed:.0f}ms)")
        print(f"  Model A / train2 darts: {[d['score'] for d in darts2] or '(none detected)'}")
        print(f"  Model B / train4 darts: {[d['score'] for d in darts4] or '(none detected)'}")
        print("  Enter counts in the popup window.")

        popup_result = collect_popup_scores(base_display, darts2, darts4, args.darts_per_image)
        action = popup_result[0]

        if action == 'quit':
            print_results(results, 'Model A: train2 / original', 'Model B: train4 / custom data')
            save_results_csv(results, args.out)
            cv2.destroyAllWindows()
            return

        if action == 'skip':
            real_darts = popup_result[1] if len(popup_result) > 1 and popup_result[1] is not None else 0
            print("  skipped")
            results.append(skipped_result(img_path, darts2, darts4, real_darts))
            continue

        _, real_darts, count_model2, count_model4 = popup_result

        results.append({
            'image':          str(img_path),
            'darts_model2':   [d['score'] for d in darts2],
            'darts_model4':   [d['score'] for d in darts4],
            'real_darts':     real_darts,
            'correct_model2': count_model2,
            'correct_model4': count_model4,
            'skipped':        False,
        })

        print(f"  recorded: real darts={real_darts}, train2={count_model2}/{real_darts}, train4={count_model4}/{real_darts}")

    cv2.destroyAllWindows()
    print_results(results, 'Model A: train2 / original', 'Model B: train4 / custom data')
    save_results_csv(results, args.out)


if __name__ == '__main__':
    main()
