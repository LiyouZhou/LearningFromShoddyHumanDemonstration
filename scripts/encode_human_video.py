#!/usr/bin/env python3
"""Encode video frames into records with metadata, images, and DINOv3 embeddings."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from policy import DINOv3VisionEncoder

RECORD_FIELDS = ("frame_number", "timestamp", "embedding")


def _center_crop_resize(frame: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize an RGB frame so its center crop matches DINOv3's input size."""
    frame = frame[:-130, :]

    height, width = frame.shape[:2]
    scale = size / min(height, width)
    resized = cv2.resize(
        frame,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    top = (resized.shape[0] - size) // 2
    left = (resized.shape[1] - size) // 2 - 50
    return np.ascontiguousarray(resized[top : top + size, left : left + size])


def encode_video(
    video_path: str | Path,
    output_path: str | Path,
    model_name: str,
    checkpoint_path: str,
    device: torch.device,
    max_frames: int | None = None,
    gripper_closed_threshold: float | None = None,
    debug_dir: str | Path | None = None,
) -> np.ndarray:
    """Encode every selected video frame and save a structured NumPy array."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise ValueError(f"Video has an invalid frames-per-second value: {fps}")

    # load csv file from the frames directory
    import csv
    csv_path = Path(str(video_path).split(".")[0] + "_frames") / "gripper_and_object_scale.csv"
    relative_scale = []
    gripper_open = []
    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            gripper_open.append(float(row["gripper_width_px"]))
            relative_scale.append(float(row["object_scale"]))

    encoder = DINOv3VisionEncoder(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
    ).to(device)
    encoder.eval()
    debug_path = Path(debug_dir) if debug_dir is not None else None
    if debug_path is not None:
        debug_path.mkdir(parents=True, exist_ok=True)
    frame_numbers: list[int] = []
    timestamps: list[float] = []
    embeddings: list[np.ndarray] = []
    original_frames: list[np.ndarray] = []

    try:
        with torch.inference_mode():
            frame_number = 0
            while max_frames is None or frame_number < max_frames:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                cropped_frame_rgb = _center_crop_resize(frame_rgb)

                if debug_path is not None:
                    debug_frame_path = debug_path / f"frame_{frame_number:06d}.jpg"
                    if not cv2.imwrite(
                        str(debug_frame_path),
                        cv2.cvtColor(cropped_frame_rgb, cv2.COLOR_RGB2BGR),
                    ):
                        raise OSError(f"Could not write debug frame: {debug_frame_path}")

                image = torch.from_numpy(cropped_frame_rgb)
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                embedding = encoder.global_embedding(image).squeeze(0).cpu().numpy()
                frame_numbers.append(frame_number)
                timestamps.append(frame_number / fps)
                embeddings.append(embedding.astype(np.float32, copy=False))
                original_frames.append(frame_rgb)
                frame_number += 1
    finally:
        capture.release()

    if not embeddings:
        raise ValueError(f"Video contains no decodable frames: {video_path}")

    records = np.empty(
        len(embeddings),
        dtype=[
            ("frame_number", np.int64),
            ("timestamp", np.float64),
            ("embedding", np.float32, (embeddings[0].shape[0],)),
            ("original_frame", np.uint8, original_frames[0].shape),
            ("gripper_state", np.uint8),
            ("object_scale", np.float32),
        ],
    )
    records["frame_number"] = frame_numbers
    records["timestamp"] = timestamps
    records["embedding"] = np.stack(embeddings)
    records["original_frame"] = original_frames
    records["gripper_state"] = (np.asarray(gripper_open) > gripper_closed_threshold).astype(np.uint8)
    records["object_scale"] = np.asarray(relative_scale, dtype=np.float32)
    np.save(output_path, records)
    return records


def load_video_embeddings(path: str | Path) -> np.ndarray:
    """Load and validate the frame-level embedding array for an imitation reward."""
    records = np.load(path, allow_pickle=False)
    if records.ndim != 1 or records.dtype.names is None:
        raise ValueError(f"Expected a one-dimensional structured array in {path}.")
    if not set(RECORD_FIELDS).issubset(records.dtype.names):
        raise ValueError(
            f"Expected fields {RECORD_FIELDS} in {path}; found {records.dtype.names}."
        )
    if records.size == 0:
        raise ValueError(f"Embedding array is empty: {path}")
    if records["embedding"].ndim != 2:
        raise ValueError(f"Embedding field must have shape (frames, dimensions): {path}")
    if "gripper_state" in records.dtype.names and not np.all(
        np.isin(records["gripper_state"], (0, 1))
    ):
        raise ValueError(f"Gripper states must contain only 0 (closed) or 1 (open): {path}")
    if not np.all(np.diff(records["frame_number"]) > 0):
        raise ValueError(f"Frame numbers must be strictly increasing: {path}")
    if not np.all(np.diff(records["timestamp"]) >= 0):
        raise ValueError(f"Timestamps must be nondecreasing: {path}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="human_demo.mp4", help="Input MP4 file.")
    parser.add_argument(
        "--output",
        default="human_demo_embeddings.npy",
        help="Output structured NumPy array.",
    )
    parser.add_argument("--model-name", default="dinov3_vitb16")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "DINO_CHECKPOINT",
            os.path.join(torch.hub.get_dir(), "checkpoints", "EUPE-ViT-B.pt"),
        ),
        help="Local pretrained DINOv3 checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for encoding.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional limit, useful for a quick smoke test.",
    )
    parser.add_argument(
        "--gripper-closed-threshold",
        type=int,
        help="value in px indicating the threshold for the gripper being considered closed.",
    )
    parser.add_argument(
        "--debug-dir",
        help="Optional directory in which to save decoded frames as JPEGs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    records = encode_video(
        video_path=args.video,
        output_path=args.output,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        device=torch.device(args.device),
        max_frames=args.max_frames,
        gripper_closed_threshold=args.gripper_closed_threshold,
        debug_dir=args.debug_dir,
    )
    print(
        f"Saved {len(records)} DINOv3 embeddings with dimension "
        f"{records['embedding'].shape[1]} and gripper states to {args.output}."
    )
