# Hardcoded model architecture parameters extracted from the MMAction2 config:
# configs/detection/videomae/vit-large-p16_videomae-k400-pre_8xb8-16x4x1-20e-adamw_ava-kinetics-rgb.py
#
# These replace Config.fromfile() + MODELS.build().

PRETRAINED_URL = (
    'https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/'
    'vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth'
)

BACKBONE_CFG = dict(
    img_size=224,
    patch_size=16,
    embed_dims=1024,
    depth=24,
    num_heads=16,
    mlp_ratio=4,
    qkv_bias=True,
    num_frames=16,
    norm_cfg=dict(type='LN', eps=1e-6),
    drop_path_rate=0.2,
    use_mean_pooling=False,
    return_feat_map=True,
)

ROI_EXTRACTOR_CFG = dict(
    roi_layer_type='RoIAlign',
    output_size=8,
    with_temporal_pool=True,
)

BBOX_HEAD_CFG = dict(
    background_class=True,
    in_channels=1024,
    num_classes=81,
    multilabel=True,
    dropout_ratio=0.5,
)

DATA_PREPROCESSOR_CFG = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    format_shape='NCTHW',
)

TEST_CFG = dict(rcnn=None)
