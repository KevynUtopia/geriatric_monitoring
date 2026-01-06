"""
Self-training utilities for keypoint detection.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import cv2
import json
from tqdm import tqdm
from ultralytics.utils import LOGGER


class PseudoLabelGenerator:
    """Generate high-quality pseudo labels for self-training."""
    
    def __init__(self, model, config: Dict):
        self.model = model
        self.config = config
        self.confidence_threshold = config.get('self_training', {}).get('pseudo_label_threshold', 0.9)
        self.consistency_threshold = config.get('self_training', {}).get('consistency_threshold', 0.8)
        
    def generate_pseudo_labels(
        self, 
        unlabeled_images: List[str],
        use_tta: bool = True,
        use_consistency: bool = True
    ) -> List[Dict]:
        """
        Generate pseudo labels with Test-Time Augmentation and consistency checking.
        
        Args:
            unlabeled_images: List of paths to unlabeled images
            use_tta: Whether to use Test-Time Augmentation
            use_consistency: Whether to use consistency checking across augmentations
            
        Returns:
            List of pseudo label dictionaries
        """
        pseudo_labels = []
        
        LOGGER.info(f"Generating pseudo labels for {len(unlabeled_images)} images")
        
        for img_path in tqdm(unlabeled_images, desc="Generating pseudo labels"):
            img_path = Path(img_path)
            
            if use_tta and use_consistency:
                # Use TTA with consistency checking
                pseudo_label = self._generate_with_tta_consistency(img_path)
            elif use_tta:
                # Use TTA without consistency checking
                pseudo_label = self._generate_with_tta(img_path)
            else:
                # Standard prediction
                pseudo_label = self._generate_standard(img_path)
            
            if pseudo_label is not None:
                pseudo_labels.append(pseudo_label)
        
        LOGGER.info(f"Generated {len(pseudo_labels)} high-quality pseudo labels")
        return pseudo_labels
    
    def _generate_standard(self, img_path: Path) -> Optional[Dict]:
        """Generate pseudo label using standard prediction."""
        results = self.model.predict(
            str(img_path),
            conf=self.confidence_threshold,
            verbose=False
        )
        
        if len(results) > 0 and results[0].keypoints is not None:
            result = results[0]
            boxes = result.boxes
            keypoints = result.keypoints
            
            if boxes is not None and keypoints is not None and len(boxes) > 0:
                # Take the highest confidence detection
                best_idx = torch.argmax(boxes.conf)
                
                return {
                    'image_path': str(img_path),
                    'bbox': boxes.xyxy[best_idx].cpu().numpy().tolist(),
                    'keypoints': keypoints.xy[best_idx].cpu().numpy().tolist(),
                    'keypoints_conf': keypoints.conf[best_idx].cpu().numpy().tolist(),
                    'confidence': boxes.conf[best_idx].item(),
                    'method': 'standard'
                }
        
        return None
    
    def _generate_with_tta(self, img_path: Path) -> Optional[Dict]:
        """Generate pseudo label using Test-Time Augmentation."""
        # Define augmentations
        augmentations = [
            {'flip': False, 'rotate': 0},
            {'flip': True, 'rotate': 0},
            {'flip': False, 'rotate': 5},
            {'flip': False, 'rotate': -5},
        ]
        
        all_predictions = []
        
        for aug in augmentations:
            # Apply augmentation and predict
            aug_results = self._predict_with_augmentation(img_path, aug)
            if aug_results:
                all_predictions.append(aug_results)
        
        if len(all_predictions) == 0:
            return None
        
        # Ensemble predictions
        return self._ensemble_predictions(all_predictions, str(img_path))
    
    def _generate_with_tta_consistency(self, img_path: Path) -> Optional[Dict]:
        """Generate pseudo label using TTA with consistency checking."""
        # Generate predictions with TTA
        tta_result = self._generate_with_tta(img_path)
        
        if tta_result is None:
            return None
        
        # Check consistency by measuring variance across augmentations
        consistency_score = self._calculate_consistency_score(img_path)
        
        if consistency_score >= self.consistency_threshold:
            tta_result['consistency_score'] = consistency_score
            tta_result['method'] = 'tta_consistency'
            return tta_result
        
        return None
    
    def _predict_with_augmentation(self, img_path: Path, augmentation: Dict) -> Optional[Dict]:
        """Apply augmentation and make prediction."""
        # Load image
        image = cv2.imread(str(img_path))
        original_image = image.copy()
        
        # Apply augmentation
        if augmentation['flip']:
            image = cv2.flip(image, 1)
        
        if augmentation['rotate'] != 0:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, augmentation['rotate'], 1.0)
            image = cv2.warpAffine(image, M, (w, h))
        
        # Save augmented image temporarily
        temp_path = img_path.parent / f"temp_aug_{img_path.name}"
        cv2.imwrite(str(temp_path), image)
        
        try:
            # Make prediction
            results = self.model.predict(
                str(temp_path),
                conf=self.confidence_threshold,
                verbose=False
            )
            
            # Clean up
            temp_path.unlink()
            
            if len(results) > 0 and results[0].keypoints is not None:
                result = results[0]
                boxes = result.boxes
                keypoints = result.keypoints
                
                if boxes is not None and keypoints is not None and len(boxes) > 0:
                    # Take the highest confidence detection
                    best_idx = torch.argmax(boxes.conf)
                    
                    # Reverse augmentation for keypoints
                    kpts = keypoints.xy[best_idx].cpu().numpy()
                    kpts_reversed = self._reverse_augmentation(
                        kpts, 
                        augmentation, 
                        original_image.shape[:2]
                    )
                    
                    return {
                        'bbox': boxes.xyxy[best_idx].cpu().numpy().tolist(),
                        'keypoints': kpts_reversed.tolist(),
                        'keypoints_conf': keypoints.conf[best_idx].cpu().numpy().tolist(),
                        'confidence': boxes.conf[best_idx].item(),
                        'augmentation': augmentation
                    }
        
        except Exception as e:
            # Clean up on error
            if temp_path.exists():
                temp_path.unlink()
            LOGGER.warning(f"Error in augmented prediction: {e}")
        
        return None
    
    def _reverse_augmentation(
        self, 
        keypoints: np.ndarray, 
        augmentation: Dict, 
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Reverse the augmentation applied to keypoints."""
        h, w = image_shape
        kpts = keypoints.copy()
        
        # Reverse rotation
        if augmentation['rotate'] != 0:
            center = np.array([w // 2, h // 2])
            angle = -augmentation['rotate'] * np.pi / 180  # Reverse rotation
            
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            # Translate to origin, rotate, translate back
            kpts_centered = kpts - center
            kpts_rotated = np.dot(kpts_centered, rotation_matrix.T)
            kpts = kpts_rotated + center
        
        # Reverse flip
        if augmentation['flip']:
            kpts[:, 0] = w - kpts[:, 0]
            
            # Also swap left-right keypoints for human pose
            if len(kpts) == 17:  # COCO keypoints
                # Swap pairs: left_eye<->right_eye, left_ear<->right_ear, etc.
                swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
                for left_idx, right_idx in swap_pairs:
                    kpts[[left_idx, right_idx]] = kpts[[right_idx, left_idx]]
        
        return kpts
    
    def _ensemble_predictions(self, predictions: List[Dict], image_path: str) -> Dict:
        """Ensemble multiple predictions into a single pseudo label."""
        if len(predictions) == 1:
            result = predictions[0].copy()
            result['image_path'] = image_path
            result['method'] = 'tta'
            return result
        
        # Calculate weighted average based on confidence
        confidences = np.array([pred['confidence'] for pred in predictions])
        weights = F.softmax(torch.tensor(confidences), dim=0).numpy()
        
        # Ensemble keypoints
        keypoints_stack = np.stack([np.array(pred['keypoints']) for pred in predictions])
        keypoints_conf_stack = np.stack([np.array(pred['keypoints_conf']) for pred in predictions])
        
        # Weighted average
        ensemble_keypoints = np.average(keypoints_stack, axis=0, weights=weights)
        ensemble_keypoints_conf = np.average(keypoints_conf_stack, axis=0, weights=weights)
        
        # Ensemble bounding box
        bboxes_stack = np.stack([np.array(pred['bbox']) for pred in predictions])
        ensemble_bbox = np.average(bboxes_stack, axis=0, weights=weights)
        
        # Average confidence
        ensemble_confidence = np.average(confidences, weights=weights)
        
        return {
            'image_path': image_path,
            'bbox': ensemble_bbox.tolist(),
            'keypoints': ensemble_keypoints.tolist(),
            'keypoints_conf': ensemble_keypoints_conf.tolist(),
            'confidence': float(ensemble_confidence),
            'method': 'tta_ensemble',
            'num_predictions': len(predictions)
        }
    
    def _calculate_consistency_score(self, img_path: Path) -> float:
        """Calculate consistency score across different augmentations."""
        augmentations = [
            {'flip': False, 'rotate': 0},
            {'flip': True, 'rotate': 0},
            {'flip': False, 'rotate': 3},
            {'flip': False, 'rotate': -3},
        ]
        
        predictions = []
        for aug in augmentations:
            pred = self._predict_with_augmentation(img_path, aug)
            if pred:
                predictions.append(np.array(pred['keypoints']))
        
        if len(predictions) < 2:
            return 0.0
        
        # Calculate pairwise distances between keypoint predictions
        predictions = np.stack(predictions)
        mean_pred = np.mean(predictions, axis=0)
        
        # Calculate normalized variance
        variances = np.var(predictions, axis=0)
        mean_variance = np.mean(variances)
        
        # Convert to consistency score (higher is better)
        consistency_score = 1.0 / (1.0 + mean_variance / 100.0)  # Normalize by image size
        
        return consistency_score


class CurriculumLearning:
    """Implement curriculum learning for self-training."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.easy_threshold = config.get('self_training', {}).get('easy_threshold', 0.95)
        self.hard_threshold = config.get('self_training', {}).get('hard_threshold', 0.7)
        
    def filter_pseudo_labels_by_difficulty(
        self, 
        pseudo_labels: List[Dict], 
        iteration: int, 
        max_iterations: int
    ) -> List[Dict]:
        """Filter pseudo labels based on curriculum learning strategy."""
        # Start with easy examples and gradually include harder ones
        progress = iteration / max_iterations
        current_threshold = self.easy_threshold - (self.easy_threshold - self.hard_threshold) * progress
        
        filtered_labels = []
        for label in pseudo_labels:
            # Calculate difficulty score (higher confidence = easier)
            difficulty_score = label['confidence']
            
            # Add consistency score if available
            if 'consistency_score' in label:
                difficulty_score = 0.7 * difficulty_score + 0.3 * label['consistency_score']
            
            if difficulty_score >= current_threshold:
                filtered_labels.append(label)
        
        LOGGER.info(f"Curriculum learning: Using threshold {current_threshold:.3f}, "
                   f"selected {len(filtered_labels)}/{len(pseudo_labels)} pseudo labels")
        
        return filtered_labels


class SelfTrainingMetrics:
    """Track and analyze self-training performance."""
    
    def __init__(self):
        self.iteration_metrics = []
        self.pseudo_label_stats = []
    
    def log_iteration(self, iteration: int, metrics: Dict):
        """Log metrics for a self-training iteration."""
        metrics['iteration'] = iteration
        self.iteration_metrics.append(metrics)
    
    def log_pseudo_labels(self, iteration: int, pseudo_labels: List[Dict]):
        """Log statistics about generated pseudo labels."""
        if not pseudo_labels:
            return
        
        confidences = [label['confidence'] for label in pseudo_labels]
        consistency_scores = [label.get('consistency_score', 0) for label in pseudo_labels]
        
        stats = {
            'iteration': iteration,
            'num_pseudo_labels': len(pseudo_labels),
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'mean_consistency': np.mean(consistency_scores) if any(consistency_scores) else 0,
        }
        
        self.pseudo_label_stats.append(stats)
        
        LOGGER.info(f"Pseudo label stats - Iteration {iteration}: "
                   f"Count: {stats['num_pseudo_labels']}, "
                   f"Mean conf: {stats['mean_confidence']:.3f}, "
                   f"Mean consistency: {stats['mean_consistency']:.3f}")
    
    def get_summary(self) -> Dict:
        """Get a summary of self-training performance."""
        if not self.iteration_metrics:
            return {}
        
        # Extract mAP values across iterations
        maps = [m.get('mAP', 0) for m in self.iteration_metrics]
        
        return {
            'total_iterations': len(self.iteration_metrics),
            'initial_map': maps[0] if maps else 0,
            'final_map': maps[-1] if maps else 0,
            'best_map': max(maps) if maps else 0,
            'improvement': maps[-1] - maps[0] if len(maps) > 1 else 0,
            'pseudo_label_stats': self.pseudo_label_stats,
            'iteration_metrics': self.iteration_metrics
        } 