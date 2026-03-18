"""
Base detector class - copied from demo_system.models.base_detector
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any


class BaseDetector(ABC):
    """Base class for all detection models."""

    def __init__(self, mode: str = "balanced", device: str = "cpu") -> None:
        self.mode = mode
        self.device = device
        self.conf_threshold = 0.5
        self.is_initialized = False

    @abstractmethod
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Main detection method."""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess input frame."""
        raise NotImplementedError

    @abstractmethod
    def postprocess(
        self, raw_results: Any, frame_shape: Tuple[int, int, int]
    ) -> Dict[str, Any]:
        """Postprocess raw model results."""
        raise NotImplementedError

    def set_confidence_threshold(self, threshold: float) -> None:
        """Set confidence threshold for detections."""
        self.conf_threshold = max(0.0, min(1.0, threshold))

    def cleanup(self) -> None:
        """Clean up model resources."""
        return

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "name": self.__class__.__name__,
            "mode": self.mode,
            "device": self.device,
            "confidence_threshold": self.conf_threshold,
            "is_initialized": self.is_initialized,
        }

