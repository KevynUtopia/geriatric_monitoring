"""
Action recognition model skeleton

Mimics an engine.action_recognition(clip, boxes) interface:
- Input: clip (deque/list of np.ndarray frames, float32), boxes (deque/list of [x1,y1,x2,y2])
- Output: dict with a single representative logits array for the clip

Replace the placeholder implementation with your real action model.
"""

from typing import Any, Dict, List, Tuple, Optional
import numpy as np
# from mmaction.registry import MODELS

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

    def _load_model(self) -> None:
        """Load your actual action model here.

        For now, we keep a lightweight placeholder to keep the plumbing working.
        """
        self.model = "placeholder"
        model_config = "/Users/kevynzhang/codespace/snh_demo/mmaction2/configs/detection/slowonly/slowonly_kinetics400-pretrained-r101_8xb16-8x8x1-20e_ava21-rgb.py"
        action_model ="https://download.openmmlab.com/mmaction/detection/ava/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb/slowonly_omnisource_pretrained_r101_8x8x1_20e_ava_rgb_20201217-16378594.pth"

        # self.action_recognizer = MODELS.build(model_config)

    def preprocess(self, clip: List[np.ndarray], boxes: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
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

        # Align boxes to T length; use the last available box if fewer boxes
        if len(boxes) == 0:
            boxes_arr = np.zeros((len(frames_arr), 4), dtype=np.float32)
        else:
            last_box = np.array(boxes[-1], dtype=np.float32)
            boxes_arr = np.stack([
                np.array(boxes[i], dtype=np.float32) if i < len(boxes) else last_box
                for i in range(len(frames_arr))
            ], axis=0)

        return frames_arr, boxes_arr

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

    def recognize(self, clip: List[np.ndarray], boxes: List[np.ndarray]) -> Dict[str, Any]:
        """Run action recognition over a clip and associated boxes.

        Returns a dict with:
          - 'logits': np.ndarray of shape (num_actions,)
          - 'num_frames': T used
        """
        frames_arr, boxes_arr = self.preprocess(clip, boxes)
        logits = self.infer_logits(frames_arr, boxes_arr)
        return {
            'logits': logits,
            'num_frames': int(frames_arr.shape[0])
        }
        
        # def action_recognition(self, image, human_detections):
    #     # torch.cuda.empty_cache()

    #     # resize frames to shortside
        
    #     w, h, _ = image.shape
    #     # short_side = min(w, h)
    #     short_side = 512
    #     new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))
    #     frames = [mmcv.imresize(img, (new_w, new_h)) for img in image]  # This will still work with deque
    #     w_ratio, h_ratio = new_w / w, new_h / h

    #     for i in range(len(human_detections)):
    #         det = human_detections[i]
    #         det[:, 0:4:2] *= w_ratio
    #         det[:, 1:4:2] *= h_ratio
    #         human_detections[i] = det[:, :4]

    #     img_norm_cfg = dict(
    #         mean=np.array(self.config.model.data_preprocessor.mean),
    #         std=np.array(self.config.model.data_preprocessor.std),
    #         to_rgb=False)

    #     # take the key frame at the center of the clip for action detection
    #     assert len(human_detections) == len(self.clip_reid_bag), \
    #         "The number of human detections and reid features should match."
    #     center_ind = len(human_detections) // 2
    #     proposal, reid = human_detections[center_ind], self.clip_reid_bag[center_ind]

    #     imgs = [f.astype(np.float32) for f in frames]
    #     _ = [mmcv.imnormalize_(img, **img_norm_cfg) for img in imgs]
    #     # THWC -> CTHW -> 1CTHW
    #     input_array = np.stack(imgs).transpose((3, 0, 1, 2))[np.newaxis]
    #     input_tensor = torch.from_numpy(input_array)

    #     datasample = ActionDataSample()
    #     datasample.proposals = InstanceData(bboxes=proposal)
    #     datasample.set_metainfo(dict(img_shape=(new_h, new_w)))
    #     with (torch.no_grad()):
    #         # print the source code of self.action_recognizer
    #         result = self.action_recognizer(input_tensor, [datasample], mode='predict')
    #         scores = result[0].pred_instances.scores
    #         print(scores.shape)
    #         exit(0)


    #         prediction = []

    #         # N proposals
    #         for i in range(proposal.shape[0]):
    #             prediction.append([])
    #         for i in range(1, scores.shape[1]+1):
    #             if i not in self.label_map:
    #                 continue
    #             row_all = []
    #             for j in range(proposal.shape[0]):
    #                 action = self.label_map[i]
    #                 prediction[j].append((action, scores[j, i-1].item()))


    #         # the first valid prediction of each clip is enough for analysis
    #         return prediction


