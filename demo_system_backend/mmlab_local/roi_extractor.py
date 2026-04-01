# Extracted from mmaction2 — SingleRoIExtractor3D.
# Uses torchvision.ops.RoIAlign instead of mmcv.ops (no compiled extensions).

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import RoIAlign


class SingleRoIExtractor3D(nn.Module):
    """Extract RoI features from a single-level 5-D feature map (N,C,T,H,W).

    Args:
        roi_layer_type: 'RoIAlign' or 'RoIPool'.
        featmap_stride: Stride of the backbone feature map.
        output_size: Spatial output size for RoI pooling.
        sampling_ratio: Sampling ratio for RoIAlign.
        pool_mode: 'avg' or 'max' — mapped to ``aligned_mode`` of
            torchvision RoIAlign (only 'avg' is supported by torchvision).
        aligned: Whether to use sub-pixel alignment.
        with_temporal_pool: Pool the temporal dimension before RoI extraction.
        temporal_pool_mode: 'avg' or 'max'.
        with_global: Concatenate global-average-pooled features.
    """

    def __init__(self,
                 roi_layer_type: str = 'RoIAlign',
                 featmap_stride: int = 16,
                 output_size: int = 16,
                 sampling_ratio: int = 0,
                 pool_mode: str = 'avg',
                 aligned: bool = True,
                 with_temporal_pool: bool = True,
                 temporal_pool_mode: str = 'avg',
                 with_global: bool = False) -> None:
        super().__init__()
        assert roi_layer_type in ('RoIPool', 'RoIAlign')
        self.spatial_scale = 1.0 / featmap_stride
        self.output_size = output_size
        self.with_temporal_pool = with_temporal_pool
        self.temporal_pool_mode = temporal_pool_mode
        self.with_global = with_global

        self.roi_layer = RoIAlign(
            output_size=(output_size, output_size),
            spatial_scale=self.spatial_scale,
            sampling_ratio=sampling_ratio,
            aligned=aligned)
        self.global_pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, feat: Union[Tensor, Tuple[Tensor]],
                rois: Tensor) -> Tuple[Tensor, Tensor]:
        if not isinstance(feat, tuple):
            feat = (feat,)

        if len(feat) >= 2:
            maxT = max(x.shape[2] for x in feat)
            max_shape = (maxT,) + feat[0].shape[3:]
            feat = [F.interpolate(x, max_shape).contiguous() for x in feat]

        if self.with_temporal_pool:
            if self.temporal_pool_mode == 'avg':
                feat = [torch.mean(x, 2, keepdim=True) for x in feat]
            elif self.temporal_pool_mode == 'max':
                feat = [torch.max(x, 2, keepdim=True)[0] for x in feat]
            else:
                raise NotImplementedError

        feat = torch.cat(feat, axis=1).contiguous()

        roi_feats = []
        for t in range(feat.size(2)):
            frame_feat = feat[:, :, t].contiguous()
            roi_feat = self.roi_layer(frame_feat, rois)
            if self.with_global:
                global_feat = self.global_pool(frame_feat.contiguous())
                inds = rois[:, 0].type(torch.int64)
                global_feat = global_feat[inds]
                roi_feat = torch.cat([roi_feat, global_feat], dim=1).contiguous()
            roi_feats.append(roi_feat)

        roi_feats = torch.stack(roi_feats, dim=2)
        return roi_feats, feat
