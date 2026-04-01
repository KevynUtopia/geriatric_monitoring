# Minimal BaseModule extracted from mmengine.model.base_module.
# Thin nn.Module wrapper — init_cfg is accepted but ignored at inference time
# (weights are loaded via state_dict, not through mmengine's init machinery).

from typing import Any, Dict, List, Optional, Union

import torch.nn as nn


class BaseModule(nn.Module):
    """Lightweight stand-in for mmengine.model.BaseModule.

    Accepts ``init_cfg`` so that class signatures remain compatible with
    the original code, but does *not* implement the mmengine weight-init
    pipeline (unnecessary for inference-only usage).
    """

    def __init__(self,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None):
        super().__init__()
        self.init_cfg = init_cfg

    def init_weights(self):
        """No-op — weights are loaded via ``load_state_dict``."""
        pass


# Alias so that ``from mmengine.model import ModuleList`` still works
ModuleList = nn.ModuleList
