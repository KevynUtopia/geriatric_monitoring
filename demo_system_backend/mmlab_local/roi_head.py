# Extracted from mmdet + mmaction2 — RoI heads (inference-only).
# BaseRoIHead / StandardRoIHead from mmdet, AVARoIHead from mmaction2.
# Training methods (loss, assigner/sampler init) are stubbed out.

from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Union

import torch
from torch import Tensor

from .base_module import BaseModule
from .structures import InstanceData


# Type aliases
InstanceList = List[InstanceData]


def bbox2roi(bbox_list: list) -> Tensor:
    """Convert a list of per-image bbox tensors to a single (K, 5) ROI tensor
    with a leading batch-index column."""
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
        rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        rois_list.append(rois)
    if len(rois_list) == 0:
        return bbox_list[0].new_zeros((0, 5))
    return torch.cat(rois_list, 0)


# ---------------------------------------------------------------------------
#  BaseRoIHead  (from mmdet)
# ---------------------------------------------------------------------------

class BaseRoIHead(BaseModule, metaclass=ABCMeta):

    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=None,
                 mask_head=None,
                 shared_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_bbox(self) -> bool:
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self) -> bool:
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @property
    def with_shared_head(self) -> bool:
        return hasattr(self, 'shared_head') and self.shared_head is not None

    def predict(self, x, rpn_results_list, batch_data_samples,
                rescale: bool = False):
        assert self.with_bbox
        batch_img_metas = [ds.metainfo for ds in batch_data_samples]
        results_list = self.predict_bbox(
            x, batch_img_metas, rpn_results_list,
            rcnn_test_cfg=self.test_cfg, rescale=rescale)
        return results_list

    @abstractmethod
    def predict_bbox(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
#  StandardRoIHead  (from mmdet, inference-only)
# ---------------------------------------------------------------------------

class StandardRoIHead(BaseRoIHead):
    """Simplified StandardRoIHead — only predict path is implemented."""

    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor,
                      **kwargs) -> dict:
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor_num_inputs], rois)
        # bbox_feats is (roi_feats, global_feat) tuple from SingleRoIExtractor3D
        if isinstance(bbox_feats, tuple):
            bbox_feats = bbox_feats[0]
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score = self.bbox_head(bbox_feats)
        return dict(cls_score=cls_score, bbox_feats=bbox_feats)

    def predict_bbox(self, x, batch_img_metas, rpn_results_list,
                     rcnn_test_cfg=None, rescale=False):
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)

        bbox_results = self._bbox_forward(x, rois)
        cls_scores = bbox_results['cls_score']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        return self.bbox_head.predict_by_feat(
            rois=rois, cls_scores=cls_scores,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg)


# ---------------------------------------------------------------------------
#  AVARoIHead  (from mmaction2, inference-only)
# ---------------------------------------------------------------------------

class AVARoIHead(StandardRoIHead):
    """AVA-specific RoI head.

    Overrides ``_bbox_forward`` to pass ``batch_img_metas`` to a shared head
    (if present) and uses the ``(roi_feats, global_feat)`` tuple directly
    from ``SingleRoIExtractor3D``.
    """

    def _bbox_forward(self, x: Union[Tensor, Tuple[Tensor]], rois: Tensor,
                      batch_img_metas: List[dict] = None, **kwargs) -> dict:
        bbox_feats, global_feat = self.bbox_roi_extractor(x, rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(
                bbox_feats, feat=global_feat, rois=rois,
                img_metas=batch_img_metas)
        cls_score = self.bbox_head(bbox_feats)
        return dict(cls_score=cls_score, bbox_feats=bbox_feats)

    def predict(self, x, rpn_results_list, batch_data_samples, **kwargs):
        assert self.with_bbox
        batch_img_metas = [ds.metainfo for ds in batch_data_samples]
        if isinstance(x, tuple):
            x_shape = x[0].shape
        else:
            x_shape = x.shape
        assert x_shape[0] == 1, 'AVARoIHead only accepts batch_size=1 at test'
        assert x_shape[0] == len(batch_img_metas) == len(rpn_results_list)

        return self.predict_bbox(
            x, batch_img_metas, rpn_results_list,
            rcnn_test_cfg=self.test_cfg)

    def predict_bbox(self, x, batch_img_metas, rpn_results_list,
                     rcnn_test_cfg=None, **kwargs):
        proposals = [res.bboxes for res in rpn_results_list]
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois, batch_img_metas)

        cls_scores = bbox_results['cls_score']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_scores = cls_scores.split(num_proposals_per_img, 0)

        return self.bbox_head.predict_by_feat(
            rois=rois, cls_scores=cls_scores,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg)
