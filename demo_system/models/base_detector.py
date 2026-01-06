"""
Base detector class - provides common interface for all detection models
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any

class BaseDetector(ABC):
    """Base class for all detection models"""
    
    def __init__(self, mode='balanced', device='cpu'):
        self.mode = mode
        self.device = device
        self.conf_threshold = 0.5
        self.is_initialized = False
        
    @abstractmethod
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Main detection method
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary containing detection results
        """
        pass
    
    @abstractmethod
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess input frame
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        pass
    
    @abstractmethod
    def postprocess(self, raw_results: Any, frame_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """
        Postprocess raw model results
        
        Args:
            raw_results: Raw results from model
            frame_shape: Original frame shape
            
        Returns:
            Processed results dictionary
        """
        pass
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for detections"""
        self.conf_threshold = max(0.0, min(1.0, threshold))
    
    def cleanup(self):
        """Clean up model resources"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': self.__class__.__name__,
            'mode': self.mode,
            'device': self.device,
            'confidence_threshold': self.conf_threshold,
            'is_initialized': self.is_initialized
        }
