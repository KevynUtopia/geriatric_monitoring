"""
demo_system_backend — offline video inference entry point.

Usage:
    python main.py -i input.mp4 [-o output.mp4] [--device cuda:0] \
                   [--frame_step 8] [--anomaly_threshold 0.145]
"""

import argparse
import os
from typing import Any, Dict, Optional

import cv2
from tqdm import tqdm

from pipeline import (
    PipelineState,
    preprocess,
    detection,
    action,
    biomarker,
    postprocess,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline backend-only video inference pipeline."
    )
    parser.add_argument("--input", "-i", required=True, help="Input video path.")
    parser.add_argument("--output", "-o", default="output.mp4", help="Output video path.")
    parser.add_argument("--device", default="cuda:0", help="Compute device.")
    parser.add_argument(
        "--mode",
        default="balanced",
        choices=["lightweight", "balanced", "performance"],
        help="Model mode.",
    )
    parser.add_argument("--max_width", type=int, default=640, help="Max output width.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to action-recognition model checkpoint (.pth).",
    )
    parser.add_argument(
        "--yolo_model",
        default="backend_weights/yolo11n-pose.pt",
        help="Path to YOLO pose detection model (.pt).",
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=8,
        help="Keep every N-th frame (default: 8, i.e. keep 1 out of 8).",
    )
    parser.add_argument(
        "--anomaly_threshold",
        type=float,
        default=0.145,
        help="Variance threshold for anomaly detection (default: 0.145).",
    )
    parser.add_argument(
        "--top_identity",
        type=int,
        default=1,
        help="Track only the top-N highest-confidence persons per frame (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input video does not exist: {args.input}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {args.input}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if src_fps <= 0:
        src_fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    state = PipelineState(mode=args.mode, device=args.device,
                          checkpoint=args.checkpoint,
                          yolo_model=args.yolo_model,
                          anomaly_threshold=args.anomaly_threshold,
                          top_identity=args.top_identity)
    writer: Optional[cv2.VideoWriter] = None

    out_fps = src_fps / args.frame_step
    kept_frames = total_frames // args.frame_step

    print(f"[main] Processing: {args.input} -> {args.output}")
    print(f"[main] {total_frames} total frames, keep every {args.frame_step}th "
          f"-> ~{kept_frames} frames, output fps={out_fps:.1f}")
    print(f"[main] anomaly_threshold={args.anomaly_threshold}")

    processed = 0
    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")

    for frame_idx in range(total_frames):
        ret, raw_frame = cap.read()
        pbar.update(1)
        if not ret:
            continue
        if frame_idx % args.frame_step != 0:
            continue

        # ── 1. preprocess ──
        # Input:  raw_frame — BGR uint8 np.ndarray of original resolution
        # Output: frame     — BGR uint8 np.ndarray resized so long side <= max_width
        frame = preprocess(raw_frame, max_width=args.max_width)

        # ── 2. detection (YOLO pose + BoT-SORT tracking) ──
        # Input:  frame — BGR uint8 np.ndarray (H, W, 3)
        # Output: det_results — dict with keys:
        #           'keypoints'  : np.ndarray (N, 17, 2) or None
        #           'scores'     : np.ndarray (N, 17) or None
        #           'boxes'      : list of np.ndarray (4,) xyxy per person, or None
        #           'track_ids'  : list of int (BoT-SORT identity) per person, or None
        #           'count'      : int — number of detected persons
        det_results = detection(frame, state)

        # ── 3. action (spatio-temporal action recognition, per-identity) ──
        # Input:  frame, det_results, state (accumulates a sliding window of frames)
        # Output: action_results — dict or None (None when clip buffer < 16 frames)
        #         When not None:
        #           'identity_results': {track_id: {
        #               'raw_scores'  : np.ndarray (81,) — sigmoid action logits
        #               'top_actions' : list of (action_name, score) tuples (top-3)
        #               'pca_scalar'  : float or None — PCA-reduced wellness score
        #               'scores'      : float — same as pca_scalar or 0
        #               'num_frames'  : int
        #               'boxes'       : np.ndarray (K, 4)
        #           }}
        #           'total_frames': int
        action_results = action(frame, det_results, state)

        # ── 4. biomarker (PCA scalar wellness + variance anomaly detection) ──
        # Input:  action_results (or None), state (holds per-identity score history)
        # Output: bio — dict or None (None when action_results is None)
        #         When not None:
        #           'anomaly_detected' : bool — True if any identity is anomalous
        #           'identities': {track_id: {
        #               'pca_scalar'  : float or None
        #               'scores'      : float
        #               'variance'    : float — normalized variance of recent scores
        #               'anomaly'     : bool  — True if variance > anomaly_threshold
        #               'top_actions' : list of (action_name, score) tuples
        #           }}
        bio = biomarker(action_results, state)

        # ── 5. postprocess (draw skeleton / boxes / identity labels → annotated frame) ──
        # Input:  frame, det_results, bio (optional)
        # Output: annotated — BGR uint8 np.ndarray with visual overlays
        annotated = postprocess(frame, det_results, biomarker_summary=bio)

        if writer is None:
            h_out, w_out = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, out_fps, (w_out, h_out))

        writer.write(annotated)
        processed += 1

        # ── DELIVER ──
        # Consolidated per-frame output bundle for downstream use.
        #
        # Fields:
        #   frame_idx      — int, 0-based index in the original video
        #   det_results    — detection dict (boxes, keypoints, track_ids, ...)
        #   action_results — action recognition dict or None; when present, each
        #                    identity's entry contains:
        #                      'raw_scores'  : np.ndarray (81,) — per-class sigmoid
        #                                      logits from the AVA action classifier.
        #                                      Index 0 is background; indices 1-80
        #                                      correspond to the 80 AVA action classes.
        #                      'top_actions' : list of (action_name: str, score: float)
        #                                      — human-readable mapping of the highest-
        #                                      scoring actions (name looked up from
        #                                      ava_label_map.txt, 1-indexed class_id →
        #                                      action name).
        #                      'pca_scalar'  : float or None — PCA-reduced scalar
        #                                      summarizing the 81-d score vector.
        #   bio            — biomarker summary dict or None; when present, each
        #                    identity's entry contains:
        #                      'variance'    : float — normalized variance of the
        #                                      recent pca_scalar history (window=5).
        #                      'anomaly'     : bool  — True when variance exceeds
        #                                      --anomaly_threshold.
        #                      'top_actions' : same as above, forwarded for convenience.
        #   annotated      — BGR uint8 np.ndarray, the visualized output frame
        DELIVER: Dict[str, Any] = {
            "frame_idx": frame_idx,
            "det_results": det_results,
            "action_results": action_results,
            "biomarker_summary": bio,
            "annotated": annotated,
        }

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()
    state.cleanup()

    print(f"[main] Done. {processed} frames written -> {args.output}")


if __name__ == "__main__":
    main()
