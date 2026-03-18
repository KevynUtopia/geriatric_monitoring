"""
Action recognition model - copied from demo_system.models.action_recognizer
and namespaced under demo_system_backend.models.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from mmaction.registry import MODELS
import mmcv
import torch
from mmaction.structures import ActionDataSample
from mmengine.structures import InstanceData
import mmengine

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import FactorAnalysis, PCA


class ActionRecognizer:
    """Action recognition module, organized by identity IDs."""

    def __init__(self, device: str = "cpu", num_actions: int = 10):
        self.device = device
        self.num_actions = num_actions
        self.model = None
        self._load_model()

        self.selected_scores: List[Any] = []
        self.factor_analysis = FactorAnalysis(n_components=3, rotation="varimax")
        self.pca = PCA()

        self.unique_ids: List[int] = []
        self.score_dict: Dict[int, List[torch.Tensor]] = {}
        self.pca_score: Dict[int, float] = {}

        self.idx_pca: Dict[int, PCA] = {}
        self.idx_var: Dict[int, List[float]] = {}

    def _load_model(self) -> None:
        """Load actual action model via MMAction2."""
        self.model = "placeholder"

        # NOTE: paths are kept as in original; adjust if needed for your environment.
        model_config = (
            "/home/wzhangbu/Desktop/AoE_Demo/snh_demo/mmaction2/configs/detection/"
            "videomae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py"
        )
        action_model = (
            "https://download.openmmlab.com/mmaction/detection/ava/"
            "slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/"
            "slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth"
        )
        self.config = mmengine.Config.fromfile(model_config)
        self.action_recognizer = MODELS.build(self.config.model).to("cuda:0")

    def preprocess(self, clip: List[np.ndarray]) -> np.ndarray:
        """Preprocess frames: stack and resize to 224x224."""
        if len(clip) == 0:
            return np.zeros((0, 224, 224, 3), dtype=np.float32)

        import cv2

        frames: List[np.ndarray] = []
        for frame in clip:
            if frame.dtype != np.float32:
                f = frame.astype(np.float32)
            else:
                f = frame
            f = cv2.resize(f, (224, 224))
            frames.append(f)
        frames_arr = np.stack(frames, axis=0)
        return frames_arr

    def infer_logits(self, frames: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Simple deterministic logits placeholder (not used by identity pipeline)."""
        if frames.size == 0:
            return np.zeros((self.num_actions,), dtype=np.float32)

        mean_rgb = frames.mean(axis=(0, 1, 2))
        h, w = frames.shape[1], frames.shape[2]
        areas = (np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1]))
        area_feat = np.mean(areas) / (max(h * w, 1))

        feat = np.concatenate([mean_rgb / 255.0, np.array([area_feat], dtype=np.float32)])
        logits = np.zeros((self.num_actions,), dtype=np.float32)
        for i in range(self.num_actions):
            logits[i] = (feat * (i + 1)).sum()
        return logits

    def recognize(
        self, clip: List[np.ndarray], boxes: List[np.ndarray], ids: List[np.ndarray]
    ) -> Dict[str, Any]:
        """Run action recognition over a clip and associated boxes, organized by identity."""
        unique_ids = set()
        for frame_ids in ids:
            if len(frame_ids) > 0:
                unique_ids.update(frame_ids)
        unique_ids = list(unique_ids)

        if not unique_ids:
            return {"identity_results": {}, "total_frames": len(clip)}

        self.unique_ids = unique_ids
        identity_results: Dict[int, Dict[str, Any]] = {}

        print("unique_ids", unique_ids)
        for identity_id in unique_ids:
            identity_clip, identity_boxes = self._extract_identity_data(
                clip, boxes, ids, identity_id
            )

            if len(identity_clip) == 0:
                continue

            frames_arr = self.preprocess(identity_clip)

            scores = self.action_recognition(frames_arr, identity_boxes, identity_id)
            scores = 0 if scores is None else scores
            if identity_id not in self.idx_var:
                self.idx_var[identity_id] = [scores]
            else:
                self.idx_var[identity_id].append(scores)

            identity_results[identity_id] = {
                "scores": scores,
                "num_frames": int(frames_arr.shape[0]),
                "boxes": identity_boxes,
                "pca_scalar": self.pca_score.get(identity_id),
            }

        anomaly_detected = False
        if unique_ids is not None:
            print("_______________Lastest 5 Timestamp________________________")
            for id_ in unique_ids:
                tmp = self.idx_var[id_][-5:]
                tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp) + 1e-9)
                if np.var(tmp) > 0.145:
                    print(f"⚠️ Identity {id_} anomaly detected: {np.var(tmp)}")
                    anomaly_detected = True
                else:
                    print(f"✔ Identity {id_} wellness scores: {np.var(tmp)}")
            print("_______________________________________")
        torch.cuda.empty_cache()
        return {
            "identity_results": identity_results,
            "total_frames": len(clip),
            "anomaly_detected": anomaly_detected,
        }

    def _extract_identity_data(
        self,
        clip: List[np.ndarray],
        boxes: List[np.ndarray],
        ids: List[np.ndarray],
        target_id: int,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Extract frames and boxes for a specific identity."""
        identity_clip: List[np.ndarray] = []
        identity_boxes: List[np.ndarray] = []

        for frame_idx, (frame, frame_boxes, frame_ids) in enumerate(zip(clip, boxes, ids)):
            identity_mask = frame_ids == target_id
            if np.any(identity_mask):
                identity_clip.append(frame)
                identity_boxes_in_frame = frame_boxes[identity_mask]
                identity_boxes.append(identity_boxes_in_frame)

        if identity_boxes:
            identity_boxes_arr = (
                np.concatenate(identity_boxes, axis=0)
                if len(identity_boxes) > 0
                else np.zeros((0, 4))
            )
        else:
            identity_boxes_arr = np.zeros((0, 4))

        return identity_clip, identity_boxes_arr

    def action_recognition(self, clip, human_detections, identity_id):
        """Action recognition on a short clip for one identity."""
        human_detections = np.expand_dims(human_detections, axis=1)

        _, w, h, _ = clip.shape
        short_side = 512
        new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))
        frames = [mmcv.imresize(img, (new_w, new_h)) for img in clip]
        w_ratio, h_ratio = new_w / w, new_h / h

        for i in range(len(human_detections)):
            det = human_detections[i]
            det[:, 0:4:2] *= w_ratio
            det[:, 1:4:2] *= h_ratio
            human_detections[i] = det[:, :4]

        img_norm_cfg = dict(
            mean=np.array(self.config.model.data_preprocessor.mean),
            std=np.array(self.config.model.data_preprocessor.std),
            to_rgb=False,
        )

        assert len(human_detections) == len(clip), "The number of human detections and frames should match."
        center_ind = len(human_detections) // 2
        proposal = torch.from_numpy(human_detections[center_ind])

        imgs = [f.astype(np.float32) for f in frames]
        _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]
        input_array = np.stack(imgs).transpose((3, 0, 1, 2))

        input_tensor = torch.from_numpy(input_array).unsqueeze(0).to("cuda:0")

        datasample = ActionDataSample()
        datasample.proposals = InstanceData(bboxes=proposal).to("cuda:0")
        datasample.set_metainfo(dict(img_shape=(new_h, new_w)))
        with torch.no_grad():
            result = self.action_recognizer(input_tensor, [datasample], mode="predict")
            scores = result[0].pred_instances.scores.cpu()
            if identity_id not in self.score_dict:
                self.score_dict[identity_id] = [scores]
            else:
                self.score_dict[identity_id].append(scores)

            print(identity_id, len(self.score_dict[identity_id]))

            if len(self.score_dict[identity_id]) >= 5:
                last_five = self.score_dict[identity_id][-5:]
                X = np.stack(
                    [t.detach().cpu().numpy().reshape(-1) for t in last_five], axis=0
                )

                if identity_id not in self.idx_pca:
                    self.idx_pca[identity_id] = PCA(n_components=1)

                comp = self.idx_pca[identity_id].fit_transform(X)
                pca_scalar = float(comp.mean())
                return pca_scalar

