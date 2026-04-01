# Extracted from mmaction2 — BBoxHeadAVA (inference-only).
# Training-only methods (loss_and_target, topk_accuracy, etc.) are removed.

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .structures import InstanceData


class BBoxHeadAVA(nn.Module):
    """AVA bounding-box classification head.

    Performs temporal + spatial pooling on RoI features, then linear
    classification into ``num_classes`` action categories (multi-label).
    """

    def __init__(self,
                 background_class: bool = True,
                 temporal_pool_type: str = 'avg',
                 spatial_pool_type: str = 'max',
                 in_channels: int = 2048,
                 num_classes: int = 81,
                 dropout_ratio: float = 0,
                 dropout_before_pool: bool = True,
                 multilabel: bool = True,
                 mlp_head: bool = False,
                 **kwargs) -> None:
        super().__init__()
        assert temporal_pool_type in ('max', 'avg')
        assert spatial_pool_type in ('max', 'avg')
        self.temporal_pool_type = temporal_pool_type
        self.spatial_pool_type = spatial_pool_type
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.dropout_before_pool = dropout_before_pool
        self.multilabel = multilabel
        self.background_class = background_class

        if temporal_pool_type == 'avg':
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            self.temporal_pool = nn.AdaptiveMaxPool3d((1, None, None))
        if spatial_pool_type == 'avg':
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        else:
            self.spatial_pool = nn.AdaptiveMaxPool3d((None, 1, 1))

        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)

        if mlp_head:
            self.fc_cls = nn.Sequential(
                nn.Linear(in_channels, in_channels), nn.ReLU(),
                nn.Linear(in_channels, num_classes))
        else:
            self.fc_cls = nn.Linear(in_channels, num_classes)

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        if self.dropout_before_pool and self.dropout_ratio > 0:
            x = self.dropout(x)
        x = self.temporal_pool(x)
        x = self.spatial_pool(x)
        if not self.dropout_before_pool and self.dropout_ratio > 0:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)
        return cls_score

    # ── Inference ─────────────────────────────────────────────────────

    def predict_by_feat(self,
                        rois: Tuple[Tensor],
                        cls_scores: Tuple[Tensor],
                        batch_img_metas: List[dict],
                        rcnn_test_cfg=None,
                        **kwargs) -> List[InstanceData]:
        result_list = []
        for img_id in range(len(batch_img_metas)):
            results = self._predict_single(
                roi=rois[img_id],
                cls_score=cls_scores[img_id],
                img_meta=batch_img_metas[img_id])
            result_list.append(results)
        return result_list

    def _predict_single(self, roi: Tensor, cls_score: Tensor,
                        img_meta: dict) -> InstanceData:
        results = InstanceData()
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        if cls_score is not None:
            scores = cls_score.sigmoid() if self.multilabel else cls_score.softmax(dim=-1)
        else:
            scores = None

        bboxes = roi[:, 1:]
        assert bboxes.shape[-1] == 4

        img_h, img_w = img_meta['img_shape']
        if img_meta.get('flip', False):
            bboxes_ = bboxes.clone()
            bboxes_[:, 0] = img_w - 1 - bboxes[:, 2]
            bboxes_[:, 2] = img_w - 1 - bboxes[:, 0]
            bboxes = bboxes_

        bboxes[:, 0::2] /= img_w
        bboxes[:, 1::2] /= img_h

        crop_quadruple = img_meta.get('crop_quadruple',
                                       np.array([0, 0, 1, 1]))
        if crop_quadruple is not None:
            x1, y1, tw, th = crop_quadruple
            bboxes[:, 0::2] = bboxes[:, 0::2] * tw + x1
            bboxes[:, 1::2] = bboxes[:, 1::2] * th + y1

        results.bboxes = bboxes
        results.scores = scores
        return results
