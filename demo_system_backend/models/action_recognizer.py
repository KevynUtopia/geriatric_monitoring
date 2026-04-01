"""
Action recognition model (per-identity) using local mmlab_local extraction.

recognize() runs the action model on a clip window and returns raw
per-identity scores. Biomarker analysis (PCA aggregation, anomaly detection)
is handled by pipeline.biomarker(), NOT here.
"""

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from sklearn.decomposition import PCA

from mmlab_local.image_utils import imnormalize_, imresize, rescale_size
from mmlab_local.label_utils import load_ava_label_map, top_k_actions
from mmlab_local.model_builder import build_action_model
from mmlab_local.model_config import DATA_PREPROCESSOR_CFG
from mmlab_local.structures import ActionDataSample, InstanceData


class ActionRecognizer:
    """Spatio-temporal action recognition, organized by identity."""

    def __init__(self, device: str = "cpu", num_actions: int = 10,
                 checkpoint: str = None):
        self.device = device
        self.num_actions = num_actions
        self.checkpoint = checkpoint
        self._load_model()

        self.label_map = load_ava_label_map()
        self.score_dict: Dict[int, List[torch.Tensor]] = {}
        self.idx_pca: Dict[int, PCA] = {}

    def _load_model(self) -> None:
        self.action_recognizer = build_action_model(
            checkpoint=self.checkpoint, device=self.device)

    # ------------------------------------------------------------------
    # Public API — called by pipeline.action()
    # ------------------------------------------------------------------

    def recognize(
        self,
        clip: List[np.ndarray],
        boxes: List[np.ndarray],
        ids: List[np.ndarray],
    ) -> Dict[str, Any]:
        """Run action recognition on a clip, grouped by identity.

        Input:
            clip  -- list of T frames (np.float32, BGR)
            boxes -- list of T arrays, each (N, 4) xyxy
            ids   -- list of T arrays, each (N,) int track-ids

        Output:
            dict with:
                'identity_results': {id: {'scores', 'pca_scalar', ...}}
                'total_frames': int
        """
        unique_ids = set()
        for frame_ids in ids:
            if len(frame_ids) > 0:
                unique_ids.update(frame_ids.tolist())
        unique_ids = sorted(unique_ids)

        if not unique_ids:
            return {"identity_results": {}, "total_frames": len(clip)}

        identity_results: Dict[int, Dict[str, Any]] = {}

        for identity_id in unique_ids:
            identity_clip, identity_boxes = self._extract_identity_data(
                clip, boxes, ids, identity_id)
            if len(identity_clip) == 0:
                continue

            frames_arr = self._preprocess_clip(identity_clip)
            fwd = self._action_forward(
                frames_arr, identity_boxes, identity_id)

            identity_results[identity_id] = {
                "scores": fwd["pca_scalar"] if fwd["pca_scalar"] is not None else 0,
                "num_frames": int(frames_arr.shape[0]),
                "boxes": identity_boxes,
                "pca_scalar": fwd["pca_scalar"],
                "raw_scores": fwd["raw_scores"],
                "top_actions": fwd["top_actions"],
            }

        return {
            "identity_results": identity_results,
            "total_frames": len(clip),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_frames(clip: List[np.ndarray],
                       num_frames: int = 16,
                       frame_interval: int = 4) -> List[np.ndarray]:
        """Uniformly sample `num_frames` from `clip` at `frame_interval`.

        Mirrors SampleAVAFrames(clip_len=16, frame_interval=4).
        """
        total = len(clip)
        needed = num_frames * frame_interval
        if total >= needed:
            start = (total - needed) // 2
            indices = list(range(start, start + needed, frame_interval))
        elif total >= num_frames:
            indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
        else:
            indices = list(range(total))
            while len(indices) < num_frames:
                indices.append(indices[-1])
        return [clip[i] for i in indices]

    def _preprocess_clip(self, clip: List[np.ndarray]) -> np.ndarray:
        """Sample 16 frames and convert to float32 (no spatial resize here)."""
        if len(clip) == 0:
            return np.zeros((0, 224, 224, 3), dtype=np.float32)
        sampled = self._sample_frames(clip, num_frames=16, frame_interval=4)
        frames = [f.astype(np.float32) for f in sampled]
        return np.stack(frames, axis=0)

    def _extract_identity_data(
        self,
        clip: List[np.ndarray],
        boxes: List[np.ndarray],
        ids: List[np.ndarray],
        target_id: int,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        identity_clip: List[np.ndarray] = []
        identity_boxes: List[np.ndarray] = []

        for frame, frame_boxes, frame_ids in zip(clip, boxes, ids):
            mask = frame_ids == target_id
            if np.any(mask):
                identity_clip.append(frame)
                identity_boxes.append(frame_boxes[mask])

        if identity_boxes:
            boxes_arr = np.concatenate(identity_boxes, axis=0)
        else:
            boxes_arr = np.zeros((0, 4))

        return identity_clip, boxes_arr

    def _action_forward(
        self,
        clip: np.ndarray,
        human_detections: np.ndarray,
        identity_id: int,
    ) -> Dict[str, Any]:
        """Run forward pass for one identity's clip.

        Returns dict with:
            raw_scores  — 1-D numpy array (81,) of sigmoid action scores
            top_actions — list of (action_name, score) tuples
            pca_scalar  — float when >= 5 score vectors accumulated, else None
        """
        human_detections = np.expand_dims(human_detections, axis=1)

        _, w, h, _ = clip.shape
        new_w, new_h = rescale_size((w, h), (256, np.Inf))
        frames = [imresize(img, (new_w, new_h)) for img in clip]
        w_ratio, h_ratio = new_w / w, new_h / h

        for i in range(len(human_detections)):
            det = human_detections[i]
            det[:, 0:4:2] *= w_ratio
            det[:, 1:4:2] *= h_ratio
            human_detections[i] = det[:, :4]

        img_norm_cfg = dict(
            mean=np.array(DATA_PREPROCESSOR_CFG['mean']),
            std=np.array(DATA_PREPROCESSOR_CFG['std']),
            to_rgb=False,
        )

        center_ind = len(human_detections) // 2
        proposal = torch.from_numpy(human_detections[center_ind])

        imgs = [f.astype(np.float32) for f in frames]
        _ = [imnormalize_(img, **img_norm_cfg) for img in imgs]
        input_array = np.stack(imgs).transpose((3, 0, 1, 2))
        input_tensor = torch.from_numpy(input_array).unsqueeze(0).to(
            self.device)

        datasample = ActionDataSample()
        datasample.proposals = InstanceData(bboxes=proposal).to(self.device)
        datasample.set_metainfo(dict(img_shape=(new_h, new_w)))

        with torch.no_grad():
            result = self.action_recognizer(
                input_tensor, [datasample], mode="predict")
            scores = result[0].pred_instances.scores.cpu()

        raw_scores = scores.detach().cpu().numpy().reshape(-1)
        top_actions = top_k_actions(raw_scores, self.label_map, k=3)

        if identity_id not in self.score_dict:
            self.score_dict[identity_id] = []
        self.score_dict[identity_id].append(scores)

        pca_scalar = None
        if len(self.score_dict[identity_id]) >= 5:
            last_five = self.score_dict[identity_id][-5:]
            X = np.stack(
                [t.detach().cpu().numpy().reshape(-1) for t in last_five],
                axis=0)

            if identity_id not in self.idx_pca:
                self.idx_pca[identity_id] = PCA(n_components=1)

            comp = self.idx_pca[identity_id].fit_transform(X)
            pca_scalar = float(comp.mean())

        return {
            "raw_scores": raw_scores,
            "top_actions": top_actions,
            "pca_scalar": pca_scalar,
        }
