# Extracted from mmdet — BaseDetector / TwoStageDetector / FastRCNN.
# Inference-only: loss() and _forward() raise NotImplementedError.
# No registry, no config parsing, no data_preprocessor auto-build.

from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .structures import InstanceData


# Type aliases
SampleList = list  # List[ActionDataSample]
InstanceList = List[InstanceData]


# ---------------------------------------------------------------------------
#  BaseDetector  (from mmdet/models/detectors/base.py + mmengine BaseModel)
# ---------------------------------------------------------------------------

class BaseDetector(nn.Module, metaclass=ABCMeta):
    """Inference-only base detector.

    ``forward(inputs, data_samples, mode)`` dispatches to ``predict``
    (the only mode we need).  The ``data_preprocessor`` is stored but
    **never called automatically** — ``action_recognizer.py`` already
    preprocesses frames manually before calling ``forward``.
    """

    def __init__(self, data_preprocessor=None, init_cfg=None):
        super().__init__()
        # Store as a sub-module so its params appear in state_dict
        if data_preprocessor is not None and isinstance(
                data_preprocessor, nn.Module):
            self.data_preprocessor = data_preprocessor
        else:
            self.data_preprocessor = nn.Identity()

    def forward(self, inputs: Tensor,
                data_samples: Optional[SampleList] = None,
                mode: str = 'tensor'):
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        elif mode == 'loss':
            raise NotImplementedError('Training not supported')
        else:
            raise RuntimeError(f'Invalid mode "{mode}"')

    @abstractmethod
    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        pass

    def _forward(self, batch_inputs, batch_data_samples=None):
        raise NotImplementedError('tensor mode not needed for inference')

    @abstractmethod
    def extract_feat(self, batch_inputs: Tensor):
        pass

    # Utility used by TwoStageDetector.predict to attach results
    @staticmethod
    def add_pred_to_datasample(data_samples, results_list):
        for data_sample, pred_instances in zip(data_samples, results_list):
            data_sample.pred_instances = pred_instances
        return data_samples

    # Properties expected by RoI head code
    @property
    def with_neck(self):
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_bbox(self):
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))


# ---------------------------------------------------------------------------
#  TwoStageDetector  (from mmdet, inference-only)
# ---------------------------------------------------------------------------

class TwoStageDetector(BaseDetector):
    """Inference-only two-stage detector.

    Components (backbone, roi_head, data_preprocessor) are set by the
    model builder — *not* built from config dicts via a registry.
    """

    def __init__(self,
                 backbone: nn.Module = None,
                 neck: nn.Module = None,
                 roi_head: nn.Module = None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor: nn.Module = None,
                 init_cfg=None) -> None:
        super().__init__(data_preprocessor=data_preprocessor,
                         init_cfg=init_cfg)
        self.backbone = backbone
        self.neck = neck
        self.roi_head = roi_head
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, batch_inputs: Tensor):
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        assert self.with_bbox
        x = self.extract_feat(batch_inputs)

        # Use pre-defined proposals (set by action_recognizer)
        rpn_results_list = [
            data_sample.proposals for data_sample in batch_data_samples
        ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples


# ---------------------------------------------------------------------------
#  FastRCNN  (thin wrapper)
# ---------------------------------------------------------------------------

class FastRCNN(TwoStageDetector):
    """Fast R-CNN — identical to TwoStageDetector for inference."""
    pass
