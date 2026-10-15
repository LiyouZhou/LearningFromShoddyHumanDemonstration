"""Track the gripper fingers and tabletop tape with SAM 2 and write gripper_and_object_scale.csv.

The CSV is written next to the frames, which is where encode_human_video.py expects it.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

TAPE_OBJ_ID = 1
FINGERS_OBJ_ID = 2

# Prompt points (x, y) clicked on frame 0 of human_demo_4.
DEFAULT_TAPE_POINTS = [[710, 100], [710, 200], [710, 250]]
DEFAULT_FINGER_POINTS = [[200, 600], [800, 600]]


def parse_points(values: list[float]) -> np.ndarray:
    if len(values) % 2:
        raise argparse.ArgumentTypeError("Points need an even number of values: x1 y1 x2 y2 ...")
    return np.array(values, dtype=np.float32).reshape(-1, 2)


def track(predictor, frames_dir: Path, prompts: dict[int, np.ndarray]) -> dict[int, dict[int, np.ndarray]]:
    state = predictor.init_state(video_path=str(frames_dir))
    for obj_id, points in prompts.items():
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=obj_id,
            points=points,
            labels=np.ones(len(points), np.int32),
        )
    return {
        frame_idx: {
            obj_id: (mask_logits[i] > 0.0).cpu().numpy().squeeze()
            for i, obj_id in enumerate(obj_ids)
        }
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state)
    }


def tape_width(mask: np.ndarray) -> int:
    xs = np.nonzero(mask)[1]
    return int(xs.max() - xs.min() + 1) if len(xs) else 0


def finger_distance(mask: np.ndarray) -> float:
    """Centroid distance between the two largest mask islands (0 if fewer than two)."""
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8) * 255, connectivity=8
    )
    if num_labels < 3:
        return 0.0
    largest = sorted(range(1, num_labels), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)[:2]
    return float(np.linalg.norm(centroids[largest[0]] - centroids[largest[1]]))


def tape_scale(widths: np.ndarray) -> np.ndarray:
    """Inverted, min-max normalised and 3-frame smoothed tape width (narrower tape = camera higher)."""
    curve = -widths
    curve = (curve - curve.min()) / (curve.max() - curve.min())
    padded = np.pad(curve, 1, mode="edge")
    return np.convolve(padded, np.ones(3, dtype=np.float32) / 3, mode="valid")


def main() -> None:
    args = parse_args()
    frames_dir = Path(args.frames_dir)

    predictor = build_sam2_video_predictor(args.model_cfg, args.checkpoint)
    with torch.inference_mode():
        segments = track(
            predictor,
            frames_dir,
            {TAPE_OBJ_ID: args.tape_points, FINGERS_OBJ_ID: args.finger_points},
        )

    widths, gripper = [], []
    for frame_idx in sorted(segments):
        masks = segments[frame_idx]
        # Reuse the previous width when the tape mask is lost so the columns stay aligned.
        width = tape_width(masks[TAPE_OBJ_ID]) if TAPE_OBJ_ID in masks else 0
        widths.append(width or (widths[-1] if widths else 0))
        gripper.append(finger_distance(masks[FINGERS_OBJ_ID]) if FINGERS_OBJ_ID in masks else 0.0)

    scale = tape_scale(np.asarray(widths, dtype=np.float32))

    output = Path(args.output) if args.output else frames_dir / "gripper_and_object_scale.csv"
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "gripper_width_px", "object_scale"])
        for i, (g, s) in enumerate(zip(gripper, scale), start=1):
            writer.writerow([i, g, float(s)])
    print(f"Saved {len(gripper)} rows to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", required=True, help="Folder of JPEG frames named 000001.jpg, ...")
    parser.add_argument("--output", default=None, help="CSV path (default: <frames-dir>/gripper_and_object_scale.csv).")
    parser.add_argument("--checkpoint", default="sam2_checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument(
        "--tape-points", type=float, nargs="+", default=sum(DEFAULT_TAPE_POINTS, []),
        help="Foreground clicks on frame 0 for the tabletop tape: x1 y1 x2 y2 ...",
    )
    parser.add_argument(
        "--finger-points", type=float, nargs="+", default=sum(DEFAULT_FINGER_POINTS, []),
        help="Foreground clicks on frame 0 for the fingers: x1 y1 x2 y2 ...",
    )
    args = parser.parse_args()
    args.tape_points = parse_points(args.tape_points)
    args.finger_points = parse_points(args.finger_points)
    return args


if __name__ == "__main__":
    main()
