import cv2
import numpy as np
from typing import Any, Dict


def draw_detection_overlays(
    frame: np.ndarray,
    detection_results: Dict[str, Any],
    show_boxes: bool = True,
    show_skeleton: bool = True,
) -> np.ndarray:
    """Draw detection overlays (pose skeletons and bounding boxes) on a frame.

    This is a backend-only version of the overlay logic from the GUI
    `VideoDisplay` component in `demo_system/ui/video_display.py`, but
    without any PyQt dependencies.
    """
    try:
        frame_copy = frame.copy()

        # Draw pose detection results
        if "pose" in detection_results and detection_results["pose"]:
            pose_results = detection_results["pose"]

            # Skeletons
            if pose_results.get("keypoints") is not None and show_skeleton:
                frame_copy = draw_pose_skeleton(frame_copy, pose_results)

            # Bounding boxes
            if show_boxes:
                boxes = pose_results.get("boxes")
                box_confidences = pose_results.get("box_confidences")
                track_ids = pose_results.get("track_ids")

                if boxes is not None:
                    for i, box in enumerate(boxes):
                        if box is None:
                            continue
                        try:
                            box_array = np.array(box, dtype=float).reshape(-1)
                            if box_array.size < 4 or not np.all(
                                np.isfinite(box_array[:4])
                            ):
                                continue

                            x1f, y1f, x2f, y2f = box_array[:4]
                            # Ensure ordering
                            x1f, x2f = (x1f, x2f) if x1f <= x2f else (x2f, x1f)
                            y1f, y2f = (y1f, y2f) if y1f <= y2f else (y2f, y1f)

                            h_, w_ = frame_copy.shape[:2]
                            x1 = int(max(0, min(w_ - 1, x1f)))
                            y1 = int(max(0, min(h_ - 1, y1f)))
                            x2 = int(max(0, min(w_ - 1, x2f)))
                            y2 = int(max(0, min(h_ - 1, y2f)))

                            if x2 <= x1 or y2 <= y1:
                                continue

                            # Different colors for different identities
                            colors = [
                                (0, 200, 255),
                                (0, 255, 0),
                                (255, 0, 255),
                            ]  # BGR
                            color = colors[i % len(colors)]

                            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

                            conf = (
                                float(box_confidences[i])
                                if box_confidences and i < len(box_confidences)
                                else 0.0
                            )
                            track_id = (
                                track_ids[i]
                                if track_ids and i < len(track_ids)
                                else None
                            )

                            if track_id is not None:
                                label = f"ID:{track_id} {conf*100:.1f}%"
                            else:
                                label = f"{conf*100:.1f}%"

                            cv2.putText(
                                frame_copy,
                                label,
                                (x1, max(0, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                1,
                                cv2.LINE_AA,
                            )
                        except Exception:
                            # Skip this box if anything goes wrong
                            continue

        return frame_copy
    except Exception:
        # Fail-safe: return original frame on any error
        return frame


def draw_pose_skeleton(frame: np.ndarray, pose_results: Dict[str, Any]) -> np.ndarray:
    """Draw pose skeletons (keypoints + connections) on a frame."""
    try:
        keypoints = pose_results.get("keypoints")
        scores = pose_results.get("scores")

        if keypoints is None or len(keypoints) == 0:
            return frame

        # Skeleton connections (COCO-like)
        skeleton_connections = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),  # Head
            (5, 6),
            (5, 7),
            (7, 9),
            (6, 8),
            (8, 10),  # Arms
            (5, 11),
            (6, 12),
            (11, 12),  # Torso
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),  # Legs
        ]

        colors = [(0, 200, 255), (0, 255, 0), (255, 0, 255)]  # BGR
        keypoint_colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255)]

        for person_idx, person_keypoints in enumerate(keypoints):
            person_color = colors[person_idx % len(colors)]
            keypoint_color = keypoint_colors[person_idx % len(keypoint_colors)]

            # Keypoints
            for i in range(len(person_keypoints)):
                try:
                    kp = person_keypoints[i]
                    x = float(kp[0])
                    y = float(kp[1])

                    if x == 0.0 and y == 0.0:
                        continue

                    if scores is None or scores[person_idx, i] > 0.5:
                        if (
                            not np.isnan(x)
                            and not np.isnan(y)
                            and np.isfinite(x)
                            and np.isfinite(y)
                        ):
                            center = (int(x), int(y))
                            cv2.circle(frame, center, 4, keypoint_color, -1)
                except Exception:
                    continue

            # Skeleton connections
            for start_idx, end_idx in skeleton_connections:
                try:
                    if (
                        start_idx >= len(person_keypoints)
                        or end_idx >= len(person_keypoints)
                    ):
                        continue

                    sx = float(person_keypoints[start_idx][0])
                    sy = float(person_keypoints[start_idx][1])
                    ex = float(person_keypoints[end_idx][0])
                    ey = float(person_keypoints[end_idx][1])

                    if (sx == 0.0 and sy == 0.0) or (ex == 0.0 and ey == 0.0):
                        continue

                    if scores is not None:
                        if (
                            scores[person_idx, start_idx] <= 0.5
                            or scores[person_idx, end_idx] <= 0.5
                        ):
                            continue

                    if (
                        not np.isnan(sx)
                        and not np.isnan(sy)
                        and np.isfinite(sx)
                        and np.isfinite(sy)
                        and not np.isnan(ex)
                        and not np.isnan(ey)
                        and np.isfinite(ex)
                        and np.isfinite(ey)
                    ):
                        start_pt = (int(sx), int(sy))
                        end_pt = (int(ex), int(ey))
                        cv2.line(frame, start_pt, end_pt, person_color, 2)
                except Exception:
                    continue

        return frame
    except Exception:
        return frame

