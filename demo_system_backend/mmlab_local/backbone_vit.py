# Extracted from mmaction2 — VisionTransformer (VideoMAE) backbone.
# Dependencies: local cnn_bricks and base_module only.

from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .base_module import BaseModule, ModuleList
from .cnn_bricks import DropPath, FFN, PatchEmbed, build_norm_layer


class Attention(BaseModule):
    """Multi-head self-attention."""

    def __init__(self,
                 embed_dims: int,
                 num_heads: int = 8,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 attn_drop_rate: float = 0.,
                 drop_rate: float = 0.,
                 init_cfg=None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims ** -0.5

        if qkv_bias:
            self._init_qv_bias()

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(drop_rate)

    def _init_qv_bias(self) -> None:
        self.q_bias = nn.Parameter(torch.zeros(self.embed_dims))
        self.v_bias = nn.Parameter(torch.zeros(self.embed_dims))

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        if hasattr(self, 'q_bias'):
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(BaseModule):
    """Transformer block: Attention + FFN with optional gamma scaling."""

    def __init__(self,
                 embed_dims: int,
                 num_heads: int,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 init_values: float = 0.0,
                 act_cfg: dict = dict(type='GELU'),
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 init_cfg=None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = Attention(
            embed_dims, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate)

        self.drop_path = nn.Identity()
        if drop_path_rate > 0.:
            self.drop_path = DropPath(drop_path_rate)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = FFN(
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_cfg=act_cfg,
            ffn_drop=drop_rate,
            add_identity=False)

        if type(init_values) is float and init_values > 0:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones(embed_dims), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones(embed_dims), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        if hasattr(self, 'gamma_1'):
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def get_sinusoid_encoding(n_position: int, embed_dims: int) -> Tensor:
    vec = torch.arange(embed_dims, dtype=torch.float64)
    vec = (vec - vec % 2) / embed_dims
    vec = torch.pow(10000, -vec).view(1, -1)

    sinusoid_table = torch.arange(n_position).view(-1, 1) * vec
    sinusoid_table[:, 0::2].sin_()
    sinusoid_table[:, 1::2].cos_()
    return sinusoid_table.to(torch.float32).unsqueeze(0)


class VisionTransformer(BaseModule):
    """VideoMAE Vision Transformer backbone.

    Produces a 5-D feature map ``(B, C, T, H, W)`` when
    ``return_feat_map=True``.
    """

    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 3,
                 embed_dims: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale: Optional[float] = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_cfg: dict = dict(type='LN', eps=1e-6),
                 init_values: float = 0.,
                 use_learnable_pos_emb: bool = False,
                 num_frames: int = 16,
                 tubelet_size: int = 2,
                 use_mean_pooling: bool = True,
                 pretrained: Optional[str] = None,
                 return_feat_map: bool = False,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        if pretrained:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv3d',
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            padding=(0, 0, 0),
            dilation=(1, 1, 1))

        grid_size = img_size // patch_size
        num_patches = grid_size ** 2 * (num_frames // tubelet_size)
        self.grid_size = (grid_size, grid_size)

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dims))
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        else:
            pos_embed = get_sinusoid_encoding(num_patches, embed_dims)
            self.register_buffer('pos_embed', pos_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = ModuleList([
            Block(embed_dims=embed_dims, num_heads=num_heads,
                  mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop_rate=drop_rate,
                  attn_drop_rate=attn_drop_rate, drop_path_rate=dpr[i],
                  norm_cfg=norm_cfg, init_values=init_values)
            for i in range(depth)
        ])

        if use_mean_pooling:
            self.norm = nn.Identity()
            self.fc_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
            self.fc_norm = None

        self.return_feat_map = return_feat_map

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, h, w = x.shape
        h //= self.patch_size
        w //= self.patch_size
        x = self.patch_embed(x)[0]

        if (h, w) != self.grid_size:
            pos_embed = self.pos_embed.reshape(
                -1, *self.grid_size, self.embed_dims)
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(
                pos_embed, size=(h, w), mode='bicubic', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = pos_embed.reshape(1, -1, self.embed_dims)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        if self.return_feat_map:
            x = x.reshape(b, -1, h, w, self.embed_dims)
            x = x.permute(0, 4, 1, 2, 3)
            return x

        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))

        return x[:, 0]
