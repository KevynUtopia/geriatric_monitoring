# Build the FastRCNN action-detection model by direct class instantiation.
# Replaces Config.fromfile() + MODELS.build().

from typing import Optional

import torch

from .backbone_vit import VisionTransformer
from .bbox_head import BBoxHeadAVA
from .detector import FastRCNN
from .model_config import (
    BACKBONE_CFG,
    BBOX_HEAD_CFG,
    DATA_PREPROCESSOR_CFG,
    PRETRAINED_URL,
    ROI_EXTRACTOR_CFG,
    TEST_CFG,
)
from .roi_extractor import SingleRoIExtractor3D
from .roi_head import AVARoIHead


def build_action_model(
    checkpoint: Optional[str] = None,
    device: str = 'cuda:0',
) -> FastRCNN:
    """Construct and return the FastRCNN model with all sub-modules.

    Args:
        checkpoint: Path to a ``.pth`` checkpoint. If *None* the model is
            returned with random weights (useful for architecture inspection).
        device: Target device string.

    Returns:
        A ``FastRCNN`` model in eval mode on ``device``.
    """
    # 1. Backbone
    backbone = VisionTransformer(**BACKBONE_CFG)

    # 2. RoI extractor
    roi_extractor = SingleRoIExtractor3D(**ROI_EXTRACTOR_CFG)

    # 3. Bbox head
    bbox_head = BBoxHeadAVA(**BBOX_HEAD_CFG)
    bbox_head.init_weights()

    # 4. RoI head — wire extractor + head together
    roi_head = AVARoIHead(test_cfg=TEST_CFG.get('rcnn'))
    roi_head.bbox_roi_extractor = roi_extractor
    roi_head.bbox_head = bbox_head

    # 5. Assemble the full detector
    model = FastRCNN(
        backbone=backbone,
        roi_head=roi_head,
        train_cfg=None,
        test_cfg=TEST_CFG,
    )

    # 6. Load checkpoint if provided
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location='cpu')
        if 'state_dict' in state:
            state = state['state_dict']

        model_keys = set(model.state_dict().keys())

        # Detect backbone-only pretrain: if most checkpoint keys lack the
        # "backbone." prefix but the model expects it, remap automatically.
        needs_prefix = (
            not any(k.startswith('backbone.') for k in state)
            and any(k.startswith('backbone.') for k in model_keys)
        )
        if needs_prefix:
            print('[model_builder] Detected backbone-only checkpoint, '
                  'adding "backbone." prefix')
            state = {f'backbone.{k}': v for k, v in state.items()}

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f'[model_builder] missing keys: {len(missing)}')
            for k in missing[:5]:
                print(f'  {k}')
            if len(missing) > 5:
                print(f'  ... and {len(missing) - 5} more')
        if unexpected:
            print(f'[model_builder] unexpected keys: {len(unexpected)}')
            for k in unexpected[:5]:
                print(f'  {k}')
            if len(unexpected) > 5:
                print(f'  ... and {len(unexpected) - 5} more')

    model.to(device)
    model.eval()
    return model
