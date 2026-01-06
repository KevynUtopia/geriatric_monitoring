"""
Action recognition model skeleton

Mimics an engine.action_recognition(clip, boxes) interface:
- Input: clip (deque/list of np.ndarray frames, float32), boxes (deque/list of [x1,y1,x2,y2])
- Output: dict with a single representative logits array for the clip

Replace the placeholder implementation with your real action model.
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from mmaction.registry import MODELS
import mmcv
import torch
from mmaction.structures import ActionDataSample
from mmengine.structures import InstanceData
import mmengine

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import FactorAnalysis,PCA


class ActionRecognizer:
    """Skeleton action recognition module.

    This class is intentionally minimal. Plug in your actual model by
    replacing `_load_model` and `infer_logits` implementations.
    """

    def __init__(self, device: str = "cpu", num_actions: int = 10):
        self.device = device
        self.num_actions = num_actions
        self.model = None
        self._load_model()

        self.selected_scores = []
        self.factor_analysis = FactorAnalysis(n_components=3, rotation='varimax')
        self.pca = PCA()

        self.unique_ids = []
        self.score_dict = {}
        self.pca_score = {}

        self.idx_pca = {}
        self.idx_var = {}


    def _load_model(self) -> None:
        """Load your actual action model here.

        For now, we keep a lightweight placeholder to keep the plumbing working.
        """
        self.model = "placeholder"

        model_config = "/home/wzhangbu/Desktop/AoE_Demo/snh_demo/mmaction2/configs/detection/videomae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py"
        action_model ="https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth"
        self.config = mmengine.Config.fromfile(model_config)
        self.action_recognizer = MODELS.build(self.config.model).to("cuda:0")

    def preprocess(self, clip: List[np.ndarray]) -> np.ndarray:
        """Preprocess inputs for the model.

        - Stacks frames into a numpy array of shape (T, H, W, C) in float32
        - Converts boxes list into array of shape (T, 4) by repeating/aligning
        """
        if len(clip) == 0:
            return np.zeros((0, 224, 224, 3), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)

        # Simple center-crop/resize placeholder: just resize frames to 224x224
        import cv2
        frames = []
        for frame in clip:
            if frame.dtype != np.float32:
                f = frame.astype(np.float32)
            else:
                f = frame
            f = cv2.resize(f, (224, 224))
            frames.append(f)
        frames_arr = np.stack(frames, axis=0)  # (T, 224, 224, 3)


        return frames_arr

    def infer_logits(self, frames: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Placeholder inference that returns a deterministic logits vector.

        Replace with your real model forward pass. We use simple statistics
        over the clip and box area to produce a stable pseudo-logits vector.
        """
        if frames.size == 0:
            return np.zeros((self.num_actions,), dtype=np.float32)

        # Compute simple features: mean intensity per channel and mean box area fraction
        mean_rgb = frames.mean(axis=(0, 1, 2))  # (3,)
        h, w = frames.shape[1], frames.shape[2]
        areas = (np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1]))
        # Normalize area by frame area (assumes original box scale; this is a heuristic placeholder)
        area_feat = np.mean(areas) / (max(h * w, 1))

        # Build a feature vector and map to logits by simple linear projections
        feat = np.concatenate([mean_rgb / 255.0, np.array([area_feat], dtype=np.float32)])  # (4,)
        # Expand to num_actions deterministically
        logits = np.zeros((self.num_actions,), dtype=np.float32)
        for i in range(self.num_actions):
            logits[i] = (feat * (i + 1)).sum()

        return logits

    def recognize(self, clip: List[np.ndarray], boxes: List[np.ndarray], 
    ids: List[np.ndarray]) -> Dict[str, Any]:
        """Run action recognition over a clip and associated boxes, organized by identity.

        Args:
            clip: List of frames (np.ndarray)
            boxes: List of boxes for each frame (np.ndarray of shape (N, 4))
            ids: List of identity IDs for each frame (np.ndarray of shape (N,))

        Returns a dict with:
          - 'identity_results': Dict mapping identity_id to results
          - 'total_frames': Total number of frames processed
        """
        
        # Get unique identity IDs
        unique_ids = set()
        for frame_ids in ids:
            if len(frame_ids) > 0:
                unique_ids.update(frame_ids)
        unique_ids = list(unique_ids)
        
        if not unique_ids:
            return {
                'identity_results': {},
                'total_frames': len(clip)
            }
        self.unique_ids = unique_ids
        
        identity_results = {}

        # Process each identity separately
        print("unique_ids", unique_ids)
        for identity_id in unique_ids:
            # Extract frames and boxes for this identity
            identity_clip, identity_boxes = self._extract_identity_data(clip, boxes, ids, identity_id)
            
            if len(identity_clip) == 0:
                continue
                
            # Preprocess the identity-specific clip
            frames_arr = self.preprocess(identity_clip)
            

            # Process through action recognition
            scores = self.action_recognition(frames_arr, identity_boxes, identity_id)
            scores = 0 if scores is None else scores
            if identity_id not in self.idx_var:
                self.idx_var[identity_id] = [scores]
            else:
                self.idx_var[identity_id].append(scores)

            
            # Store results for this identity
            identity_results[identity_id] = {
                'scores': scores,
                'num_frames': int(frames_arr.shape[0]),
                'boxes': identity_boxes,
                'pca_scalar': self.pca_score.get(identity_id)
            }
        
        # Check for anomalies and return status
        anomaly_detected = False
        if unique_ids is not None:
            print("_______________Lastest 5 Timestamp________________________")
            for id in unique_ids:
                # print(f"\u2714 Identity {id} wellness scores: {self.idx_var[id][-5:]}")
                tmp = self.idx_var[id][-5:]
                # normalize tmp to 0-1
                tmp = (tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp) + 1e-9)
                if np.var(tmp) > 0.145:
                    print(f"\u26A0\uFE0F Identity {id} anomaly detected: {np.var(tmp)}")
                    anomaly_detected = True
                else:
                    print(f"\u2714 Identity {id} wellness scores: {np.var(tmp)}")
            print("_______________________________________")
        torch.cuda.empty_cache()
        return {
            'identity_results': identity_results,
            'total_frames': len(clip),
            'anomaly_detected': anomaly_detected
        }
    
    def _extract_identity_data(self, clip: List[np.ndarray], boxes: List[np.ndarray], 
                              ids: List[np.ndarray], target_id: int) -> Tuple[List[np.ndarray], np.ndarray]:
        """Extract frames and boxes for a specific identity.
        
        Args:
            clip: List of frames
            boxes: List of boxes for each frame
            ids: List of identity IDs for each frame
            target_id: The identity ID to extract
            
        Returns:
            Tuple of (identity_clip, identity_boxes) where:
            - identity_clip: List of frames where the target identity appears
            - identity_boxes: Array of boxes for the target identity
        """
        identity_clip = []
        identity_boxes = []
        
        for frame_idx, (frame, frame_boxes, frame_ids) in enumerate(zip(clip, boxes, ids)):
            # Find indices where the target identity appears
            identity_mask = (frame_ids == target_id)
            
            if np.any(identity_mask):
                # Add this frame to the identity clip
                identity_clip.append(frame)
                
                # Extract boxes for this identity in this frame
                identity_boxes_in_frame = frame_boxes[identity_mask]
                identity_boxes.append(identity_boxes_in_frame)
        
        # Convert boxes list to array if we have data
        if identity_boxes:
            # For now, we'll concatenate all boxes from all frames
            # You might want to modify this based on your specific needs
            identity_boxes_arr = np.concatenate(identity_boxes, axis=0) if len(identity_boxes) > 0 else np.zeros((0, 4))
        else:
            identity_boxes_arr = np.zeros((0, 4))
            
        return identity_clip, identity_boxes_arr
        
    def action_recognition(self, clip, human_detections, identity_id):
        # torch.cuda.empty_cache()

        # resize frames to shortside
        # Add one dimension at the second axis of human_detections
        human_detections = np.expand_dims(human_detections, axis=1) # trial, for each identity

        
        _, w, h, _ = clip.shape
        # short_side = min(w, h)
        short_side = 512
        # new_w, new_h = 512, 960
        new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))
        frames = [mmcv.imresize(img, (new_w, new_h)) for img in clip]  # This will still work with deque
        w_ratio, h_ratio = new_w / w, new_h / h

        for i in range(len(human_detections)):
            det = human_detections[i]
            det[:, 0:4:2] *= w_ratio
            det[:, 1:4:2] *= h_ratio
            human_detections[i] = det[:, :4]

        img_norm_cfg = dict(
            mean=np.array(self.config.model.data_preprocessor.mean),
            std=np.array(self.config.model.data_preprocessor.std),
            to_rgb=False)

        # take the key frame at the center of the clip for action detection
        assert len(human_detections) == len(clip), \
            "The number of human detections and reid features should match."
        center_ind = len(human_detections) // 2
        proposal = torch.from_numpy(human_detections[center_ind])

        imgs = [f.astype(np.float32) for f in frames]
        _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]
        # THWC -> CTHW -> 1CTHW
        input_array = np.stack(imgs).transpose((3, 0, 1, 2))

        input_tensor = torch.from_numpy(input_array).unsqueeze(0).to("cuda:0")

        datasample = ActionDataSample()
        datasample.proposals = InstanceData(bboxes=proposal).to("cuda:0")
        datasample.set_metainfo(dict(img_shape=(new_h, new_w)))
        with (torch.no_grad()):

            result = self.action_recognizer(input_tensor, [datasample], mode='predict')


            scores = result[0].pred_instances.scores.cpu()
            if identity_id not in self.score_dict:
                self.score_dict[identity_id] = [scores]
            else:
                self.score_dict[identity_id].append(scores)

            print(identity_id, len(self.score_dict[identity_id]))

            # When we have at least 5 score tensors for this identity, aggregate last 5 via PCA -> scalar
            if len(self.score_dict[identity_id]) >= 5:
                last_five = self.score_dict[identity_id][-5:]
                # Convert each tensor [1, 81] -> (81,) and stack => (5, 81)
                X = np.stack([t.detach().cpu().numpy().reshape(-1) for t in last_five], axis=0)
                # PCA to 1 component => (5, 1)

                if identity_id not in self.idx_pca:
                    self.idx_pca[identity_id] = PCA(n_components=1)
                
                comp = self.idx_pca[identity_id].fit_transform(X)
                # Reduce to single scalar by averaging across 5 timestamps
                pca_scalar = float(comp.mean())
                return pca_scalar




