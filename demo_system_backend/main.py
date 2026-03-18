import argparse
import os
from typing import Optional

import cv2
import numpy as np

from demo_system_backend.core.detection_processor import DetectionProcessor
from demo_system_backend.draw_utils import draw_detection_overlays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline backend-only version of demo_system.\n\n"
            "Given an input MP4, run the same detection pipeline as the GUI demo "
            "and export an annotated low-resolution MP4."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input video file (e.g., input.mp4).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output.mp4",
        help="Path to output video file (default: output.mp4).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help='Computation device for the detector (default: "cuda:0"). '
        'Use "cpu" if no GPU is available.',
    )
    parser.add_argument(
        "--mode",
        default="balanced",
        choices=["lightweight", "balanced", "performance"],
        help="Model mode consistent with demo_system (default: balanced).",
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=640,
        help="Maximum width for processing/writing frames (default: 640). "
        "This keeps output size and bitrate low.",
    )
    return parser.parse_args()


def create_video_writer(
    output_path: str, fps: float, frame_size: tuple[int, int]
) -> cv2.VideoWriter:
    """Create an MP4 writer with reasonable low-quality settings."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = frame_size
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {output_path}")
    return writer


def resize_for_max_width(
    frame: np.ndarray, max_width: int
) -> tuple[np.ndarray, float]:
    """Resize frame to max_width while keeping aspect ratio; return frame and scale."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    scale = max_width / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def process_video(
    input_path: str,
    output_path: str,
    device: str = "cuda:0",
    mode: str = "balanced",
    max_width: int = 640,
) -> None:
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    # Try to get metadata
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if src_fps <= 0:
        src_fps = 30.0

    # Initialize detection processor (same backend as GUI)
    print(f"[backend] Initializing DetectionProcessor (mode={mode}, device={device})")
    detector = DetectionProcessor(mode=mode, device=device)

    writer: Optional[cv2.VideoWriter] = None
    frame_idx = 0

    print(f"[backend] Starting processing: {input_path}")
    print(f"[backend] Output will be written to: {output_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Optionally downscale to keep output small
        frame_proc, _ = resize_for_max_width(frame, max_width=max_width)

        # Run the same detection pipeline as the GUI, focusing on pose/action
        detection_types = ["pose", "action"]
        results = detector.process_frame(frame_proc, detection_types)

        # Draw overlays (skeletons + boxes), similar to GUI visualization
        annotated = draw_detection_overlays(
            frame_proc,
            results if isinstance(results, dict) else {},
            show_boxes=True,
            show_skeleton=True,
        )

        if writer is None:
            h_out, w_out = annotated.shape[:2]
            writer = create_video_writer(output_path, src_fps, (w_out, h_out))

        writer.write(annotated)

        if frame_idx % 50 == 0:
            print(f"[backend] Processed {frame_idx} frames...")

    cap.release()
    if writer is not None:
        writer.release()

    # Clean up models/resources
    if hasattr(detector, "cleanup"):
        detector.cleanup()

    print(f"[backend] Finished. Total frames: {frame_idx}")
    print(f"[backend] Output saved to: {output_path}")


def main() -> None:
    args = parse_args()
    process_video(
        input_path=args.input,
        output_path=args.output,
        device=args.device,
        mode=args.mode,
        max_width=args.max_width,
    )


if __name__ == "__main__":
    main()

