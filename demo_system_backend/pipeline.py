"""
Pipeline module — the five stages called by main.py's while-loop.

Every function has explicit inputs and outputs. No async, no threads,
no deques — just a simple linear flow.

    preprocess(raw_frame, max_width)  → frame
    detection(frame, state)           → det_results
    action(frame, det_results, state) → action_results | None
    biomarker(action_results, state)  → biomarker_summary | None
    postprocess(frame, det_results, biomarker_summary) → annotated_frame
"""

import numpy as np
import cv2
from typing import Any, Dict, List, Optional

from draw_utils import draw_detection_overlays


# ---------------------------------------------------------------------------
# Pipeline state — instantiated once, shared across frames
# ---------------------------------------------------------------------------

class PipelineState:
    """Mutable state carried across frames (model handles, buffers, counters)."""

    def __init__(self, mode: str = "balanced", device: str = "cuda:0",
                 checkpoint: str = None,
                 yolo_model: str = "backend_weights/yolo11n-pose.pt",
                 anomaly_threshold: float = 0.145,
                 top_identity: int = 1) -> None:
        self.mode = mode
        self.device = device
        self.checkpoint = checkpoint
        self.yolo_model = yolo_model
        self.anomaly_threshold = anomaly_threshold
        self.top_identity = top_identity

        # -- Models (initialized once) --
        self.pose_model = None
        self.action_model = None
        self._init_models()

        # -- Frame accumulation for action (simple lists, window of 5) --
        self.clip_buffer: List[np.ndarray] = []
        self.boxes_buffer: List[np.ndarray] = []
        self.ids_buffer: List[np.ndarray] = []
        self.clip_window: int = 64
        self.sampling_count: int = 0
        self.action_every_n: int = 5

        # -- Biomarker per-identity history --
        self.idx_var: Dict[int, List[float]] = {}

    def _init_models(self) -> None:
        try:
            from models.pose_detector import PoseDetector
            self.pose_model = PoseDetector(
                mode=self.mode, device=self.device,
                model_path=self.yolo_model,
                top_k=self.top_identity)
            print("[pipeline] PoseDetector initialized")
        except Exception as e:
            print(f"[pipeline] Failed to init PoseDetector: {e}")

        try:
            from models.action_recognizer import ActionRecognizer
            self.action_model = ActionRecognizer(
                device=self.device, num_actions=10,
                checkpoint=self.checkpoint)
            print("[pipeline] ActionRecognizer initialized")
        except Exception as e:
            print(f"[pipeline] Failed to init ActionRecognizer: {e}")

    def cleanup(self) -> None:
        if self.pose_model and hasattr(self.pose_model, "cleanup"):
            self.pose_model.cleanup()
        print("[pipeline] Cleaned up")


# ---------------------------------------------------------------------------
# 1. preprocess
# ---------------------------------------------------------------------------

def preprocess(raw_frame: np.ndarray, max_width: int = 640) -> np.ndarray:
    """Resize the raw frame so the long side <= max_width (keeps aspect ratio).

    Input:  raw BGR frame from cv2.VideoCapture
    Output: resized BGR frame
    """
    h, w = raw_frame.shape[:2]
    if w > max_width or h > max_width:
        scale = min(max_width / w, max_width / h)
        raw_frame = cv2.resize(
            raw_frame,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return raw_frame


# ---------------------------------------------------------------------------
# 2. detection  (detect + track in one step via YOLO + BoT-SORT)
# ---------------------------------------------------------------------------

def detection(frame: np.ndarray, state: PipelineState) -> Dict[str, Any]:
    """Run YOLO pose detection with built-in BoT-SORT tracking.

    Input:  preprocessed frame, pipeline state (holds pose_model)
    Output: dict with 'keypoints', 'scores', 'boxes', 'track_ids', 'count', etc.

    NOTE: detection and tracking happen together inside YOLO's model.track().
    A future refactor may split them into separate stages.
    """
    if state.pose_model is None or not getattr(state.pose_model, "is_initialized", False):
        return {"keypoints": None, "scores": None, "count": 0}

    return state.pose_model.detect(frame)


# ---------------------------------------------------------------------------
# 3. action  (spatio-temporal action recognition, synchronous, per-identity)
# ---------------------------------------------------------------------------

def action(
    frame: np.ndarray,
    det_results: Dict[str, Any],
    state: PipelineState,
) -> Optional[Dict[str, Any]]:
    """Accumulate frames into a sliding window; run action recognition when full.

    Input:  current frame, detection results (boxes + track_ids), pipeline state
    Output: action_results dict from ActionRecognizer.recognize(), or None if
            the window is not yet full / no valid detections.

    This is fully synchronous — it blocks until recognition finishes.
    """
    state.sampling_count += 1
    if state.sampling_count % state.action_every_n != 0:
        return None

    pose_boxes = det_results.get("boxes")
    pose_ids = det_results.get("track_ids")

    # Only accumulate frames that have valid detections
    if pose_boxes is not None:
        state.clip_buffer.append(frame.astype(np.float32))
        state.boxes_buffer.append(np.array(pose_boxes, dtype=np.float32))
        state.ids_buffer.append(np.array(pose_ids, dtype=np.int32))

    # Keep only the last `clip_window` entries
    if len(state.clip_buffer) > state.clip_window:
        state.clip_buffer = state.clip_buffer[-state.clip_window:]
        state.boxes_buffer = state.boxes_buffer[-state.clip_window:]
        state.ids_buffer = state.ids_buffer[-state.clip_window:]

    # Need at least 16 frames (model requires 16-frame input)
    if len(state.clip_buffer) < 16:
        return None

    if state.action_model is None:
        return None

    # Synchronous call — blocks until done
    action_results = state.action_model.recognize(
        list(state.clip_buffer),
        list(state.boxes_buffer),
        list(state.ids_buffer),
    )

    return action_results


# ---------------------------------------------------------------------------
# 4. biomarker  (PCA scalar + variance-based anomaly detection)
# ---------------------------------------------------------------------------

def biomarker(
    action_results: Optional[Dict[str, Any]],
    state: PipelineState,
) -> Optional[Dict[str, Any]]:
    """Compute per-identity biomarker wellness scores from action results.

    Input:  action_results from action(), pipeline state (holds idx_var history)
    Output: dict with 'anomaly_detected' flag and per-identity breakdown,
            or None if action_results is None.

    The PCA reduction (score_dict → pca_scalar) is already done inside
    ActionRecognizer.action_recognition(). This function accumulates the
    per-identity scalars over time and applies variance-based anomaly detection.
    """
    if action_results is None or not isinstance(action_results, dict):
        return None

    identity_results = action_results.get("identity_results", {})
    anomaly_detected = False

    summary: Dict[str, Any] = {
        "anomaly_detected": False,
        "identities": {},
    }

    for identity_id, res in identity_results.items():
        pca_scalar = res.get("pca_scalar")
        score_val = res.get("scores", 0)

        # Accumulate wellness scores over time
        if identity_id not in state.idx_var:
            state.idx_var[identity_id] = []
        state.idx_var[identity_id].append(score_val if score_val is not None else 0)

        # Variance-based anomaly detection on the last 5 scores
        recent = np.array(state.idx_var[identity_id][-5:], dtype=np.float64)
        denom = np.max(recent) - np.min(recent) + 1e-9
        normalized = (recent - np.min(recent)) / denom
        variance = float(np.var(normalized))

        is_anomaly = variance > state.anomaly_threshold
        if is_anomaly:
            anomaly_detected = True
            print(f"[biomarker] Identity {identity_id} ANOMALY (var={variance:.4f})")
        else:
            print(f"[biomarker] Identity {identity_id} normal  (var={variance:.4f})")

        summary["identities"][identity_id] = {
            "pca_scalar": pca_scalar,
            "scores": score_val,
            "variance": variance,
            "anomaly": is_anomaly,
            "top_actions": res.get("top_actions", []),
        }

    summary["anomaly_detected"] = anomaly_detected
    return summary


# ---------------------------------------------------------------------------
# 5. postprocess  (draw overlays → annotated frame)
# ---------------------------------------------------------------------------

def postprocess(
    frame: np.ndarray,
    det_results: Dict[str, Any],
    biomarker_summary: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Draw detection overlays (skeleton + boxes + identity labels) and biomarker info.

    Input:  preprocessed frame, detection results, optional biomarker summary
    Output: annotated BGR frame ready to be written to the output video
    """
    annotated = draw_detection_overlays(
        frame,
        {"pose": det_results} if det_results else {},
        show_boxes=True,
        show_skeleton=True,
    )

    # Per-person identity labels drawn at box center
    _draw_identity_labels(annotated, det_results, biomarker_summary)

    # Global anomaly banner
    if biomarker_summary and biomarker_summary.get("anomaly_detected"):
        cv2.putText(
            annotated,
            "ANOMALY",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated


def _draw_identity_labels(
    frame: np.ndarray,
    det_results: Dict[str, Any],
    biomarker_summary: Optional[Dict[str, Any]],
) -> None:
    """Draw a small PID tag at the top-left corner of each person's bounding box."""
    boxes = det_results.get("boxes")
    track_ids = det_results.get("track_ids")
    if boxes is None or track_ids is None:
        return

    id_colors = [(0, 200, 255), (0, 255, 0), (255, 0, 255),
                 (255, 255, 0), (0, 255, 255), (255, 128, 0)]

    bio_identities = {}
    if biomarker_summary and isinstance(biomarker_summary.get("identities"), dict):
        bio_identities = biomarker_summary["identities"]

    for box, tid in zip(boxes, track_ids):
        if box is None or tid is None:
            continue
        try:
            box_arr = np.array(box, dtype=float).reshape(-1)
            if box_arr.size < 4 or not np.all(np.isfinite(box_arr[:4])):
                continue
            x1, y1 = int(box_arr[0]), int(box_arr[1])
            color = id_colors[int(tid) % len(id_colors)]

            label = f"P{int(tid)}"
            cv2.putText(
                frame, label, (x1 + 2, y1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )

            tid_int = int(tid)
            if tid_int in bio_identities and bio_identities[tid_int].get("anomaly"):
                cv2.putText(
                    frame, "!", (x1 + 2, y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA,
                )
        except Exception:
            continue
