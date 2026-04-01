# Extracted from mmcv.cnn.bricks — build_norm_layer, DropPath, FFN, PatchEmbed.
# No mmengine registry dependency; norm types are resolved by a simple dict.

import math
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_module import BaseModule

# ── Norm-layer factory ────────────────────────────────────────────────────

NORM_LAYERS: Dict[str, type] = {
    'BN': nn.BatchNorm2d,
    'BN1d': nn.BatchNorm1d,
    'BN2d': nn.BatchNorm2d,
    'BN3d': nn.BatchNorm3d,
    'SyncBN': nn.SyncBatchNorm,
    'GN': nn.GroupNorm,
    'LN': nn.LayerNorm,
    'IN': nn.InstanceNorm2d,
    'IN1d': nn.InstanceNorm1d,
    'IN2d': nn.InstanceNorm2d,
    'IN3d': nn.InstanceNorm3d,
}

_ABBR = {
    'BatchNorm': 'bn',
    'GroupNorm': 'gn',
    'LayerNorm': 'ln',
    'InstanceNorm': 'in',
    'SyncBatchNorm': 'bn',
}


def _infer_abbr(cls: type) -> str:
    for full, short in _ABBR.items():
        if full in cls.__name__:
            return short
    return cls.__name__.lower()


def build_norm_layer(cfg: dict, num_features: int,
                     postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    """Build a normalisation layer from a config dict.

    Supports ``type`` in {'LN', 'BN', 'GN', …} — see ``NORM_LAYERS``.
    """
    cfg_ = cfg.copy()
    layer_type = cfg_.pop('type')
    norm_cls = NORM_LAYERS.get(layer_type)
    if norm_cls is None:
        raise KeyError(f'Unknown norm type {layer_type}')

    abbr = _infer_abbr(norm_cls)
    name = abbr + str(postfix)

    if norm_cls is nn.GroupNorm:
        assert 'num_groups' in cfg_
        layer = norm_cls(num_channels=num_features, **cfg_)
    else:
        layer = norm_cls(num_features, **cfg_)

    return name, layer


# ── Activation factory (minimal) ─────────────────────────────────────────

ACT_LAYERS: Dict[str, type] = {
    'ReLU': nn.ReLU,
    'GELU': nn.GELU,
    'SiLU': nn.SiLU,
    'Sigmoid': nn.Sigmoid,
    'LeakyReLU': nn.LeakyReLU,
    'ELU': nn.ELU,
}


def build_activation_layer(cfg: dict) -> nn.Module:
    cfg_ = cfg.copy()
    act_type = cfg_.pop('type')
    act_cls = ACT_LAYERS.get(act_type)
    if act_cls is None:
        raise KeyError(f'Unknown activation type {act_type}')
    return act_cls(**cfg_)


# ── DropPath (stochastic depth) ──────────────────────────────────────────

def drop_path(x: Tensor, drop_prob: float = 0.,
              training: bool = False) -> Tensor:
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


# ── FFN (feed-forward network) ───────────────────────────────────────────

class FFN(BaseModule):
    """Feed-forward network with optional identity shortcut."""

    def __init__(self,
                 embed_dims: int = 256,
                 feedforward_channels: int = 1024,
                 num_fcs: int = 2,
                 act_cfg: dict = dict(type='ReLU', inplace=True),
                 ffn_drop: float = 0.,
                 add_identity: bool = True,
                 init_cfg=None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.add_identity = add_identity

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(nn.Linear(in_channels, feedforward_channels))
            layers.append(build_activation_layer(act_cfg))
            layers.append(nn.Dropout(ffn_drop))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor, identity: Optional[Tensor] = None) -> Tensor:
        out = self.layers(x)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out


# ── AdaptivePadding ──────────────────────────────────────────────────────

class AdaptivePadding(nn.Module):
    """Pads input so the output size is ceil(input_size / stride)."""

    def __init__(self, kernel_size=1, stride=1, dilation=1,
                 padding='corner'):
        super().__init__()
        self.padding = padding
        self.kernel_size = self._to_tuple(kernel_size)
        self.stride = self._to_tuple(stride)
        self.dilation = self._to_tuple(dilation)

    @staticmethod
    def _to_tuple(x):
        if isinstance(x, int):
            return (x, x)
        return tuple(x)

    def get_pad_shape(self, input_shape):
        pads = []
        for i_dim, k, s, d in zip(
                input_shape, self.kernel_size, self.stride, self.dilation):
            pad = max((math.ceil(i_dim / s) - 1) * s +
                      (k - 1) * d + 1 - i_dim, 0)
            pads.append(pad)
        return pads

    def forward(self, x: Tensor) -> Tensor:
        pad_h, pad_w = self.get_pad_shape(x.shape[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2,
                    pad_h // 2, pad_h - pad_h // 2,
                ])
        return x


# ── Conv-layer factory (minimal, for PatchEmbed) ─────────────────────────

CONV_LAYERS: Dict[str, type] = {
    'Conv1d': nn.Conv1d,
    'Conv2d': nn.Conv2d,
    'Conv3d': nn.Conv3d,
}


def build_conv_layer(cfg: Optional[dict], *args, **kwargs) -> nn.Module:
    if cfg is None:
        return nn.Conv2d(*args, **kwargs)
    cfg_ = cfg.copy()
    conv_type = cfg_.pop('type')
    conv_cls = CONV_LAYERS.get(conv_type)
    if conv_cls is None:
        raise KeyError(f'Unknown conv type {conv_type}')
    return conv_cls(*args, **kwargs)


# ── PatchEmbed ───────────────────────────────────────────────────────────

def _to_pair(x):
    if isinstance(x, int):
        return (x, x)
    return tuple(x)


class PatchEmbed(BaseModule):
    """Image / video to patch embedding via a single convolution."""

    def __init__(self,
                 in_channels: int = 3,
                 embed_dims: int = 768,
                 conv_type: str = 'Conv2d',
                 kernel_size: Union[int, Tuple] = 16,
                 stride: Union[int, Tuple] = 16,
                 padding: Union[int, Tuple] = 0,
                 dilation: Union[int, Tuple] = 1,
                 bias: bool = True,
                 norm_cfg: Optional[dict] = None,
                 input_size: Optional[Union[int, Tuple]] = None,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, ...]]:
        x = self.projection(x)
        # out_size depends on whether input is 2D or 3D conv
        if x.ndim == 5:
            out_size = tuple(x.shape[2:])          # (T, H, W)
        else:
            out_size = (x.shape[2], x.shape[3])    # (H, W)
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size
