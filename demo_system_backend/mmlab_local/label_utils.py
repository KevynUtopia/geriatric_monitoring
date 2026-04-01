# AVA action label utilities.

import os
from typing import Dict, List, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_LABEL_FILE = os.path.join(_THIS_DIR, "ava_label_map.txt")


def load_ava_label_map(path: str = None) -> Dict[int, str]:
    """Load AVA label map from a ``label_map.txt`` file.

    Format per line: ``<class_id>: <action_name>``

    Returns:
        dict mapping 1-based class_id → action name string.
    """
    path = path or _DEFAULT_LABEL_FILE
    label_map: Dict[int, str] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            idx_str, name = line.split(":", 1)
            label_map[int(idx_str.strip())] = name.strip()
    return label_map


def top_k_actions(
    scores, label_map: Dict[int, str], k: int = 3, threshold: float = 0.1
) -> List[Tuple[str, float]]:
    """Return the top-k actions above ``threshold`` from a score vector.

    Args:
        scores: 1-D tensor or ndarray of shape (num_classes,).
                Index 0 is the background class and is skipped.
        label_map: mapping from 1-based class_id to action name.
        k: number of top actions to return.
        threshold: minimum score to include.

    Returns:
        List of (action_name, score) tuples, sorted by score descending.
    """
    import torch
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()

    results: List[Tuple[str, float]] = []
    for cls_id, name in label_map.items():
        idx = cls_id - 1  # scores is 0-indexed, label_map is 1-indexed
        if 0 <= idx < len(scores):
            s = float(scores[idx])
            if s >= threshold:
                results.append((name, s))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]
