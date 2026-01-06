"""
YOLO Pose detection model implementation
"""

import numpy as np
import cv2
import os
from typing import Dict, List, Optional, Tuple, Any
from .base_detector import BaseDetector

class PoseDetector(BaseDetector):
    """YOLO Pose detection model implementation"""
    
    def __init__(self, mode='balanced', device='cpu', model_path=None):
        super().__init__(mode, device)
        self.model_path = model_path or "backend_weights/yolo11n-pose.pt"
        self.model = None
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        self.init_model()
    
    def init_model(self):
        """Initialize YOLO pose detection model"""
        try:
            from ultralytics import YOLO as YOLO_ultralytics
            
            if not os.path.exists(self.model_path):
                print(f"Model file not found: {self.model_path}")
                self.is_initialized = False
                return
            
            print(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO_ultralytics(self.model_path, verbose=False)
            print(f"YOLO pose detection model initialized (mode: {self.mode}, device: {self.device})")
            self.is_initialized = True
            
        except ImportError as e:
            print(f"Failed to import ultralytics: {e}")
            print("Please install ultralytics: pip install ultralytics")
            self.is_initialized = False
        except Exception as e:
            print(f"Failed to initialize YOLO pose detection model: {e}")
            self.is_initialized = False
    
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect poses in frame using YOLO"""
        if not self.is_initialized or self.model is None:
            return {'keypoints': None, 'scores': None, 'count': 0}
        
        try:
            # Run YOLO inference with tracking
            results = self.model.track(frame, tracker="botsort.yaml", 
            conf=self.conf_threshold, verbose=False)

            # Process results
            keypoints_list = []
            scores_list = []
            boxes_list = []
            box_scores_list = []
            track_ids_list = []
            
            for result in results:
                if result.keypoints is not None and len(result.keypoints.data) > 0:
                    # Extract keypoints and confidence scores
                    keypoints_data = result.keypoints.data.cpu().numpy()
                    
                    # Extract track IDs if available
                    if hasattr(result, 'boxes') and result.boxes is not None and result.boxes.id is not None:
                        track_ids = result.boxes.id.int().cpu().tolist()
                        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                        boxes_conf = result.boxes.conf.cpu().numpy()
                    else:
                        track_ids = None
                        boxes_xyxy = None
                        boxes_conf = None
                    
                    for idx, person_keypoints in enumerate(keypoints_data):
                        # keypoints_data shape: [num_people, num_keypoints, 3] (x, y, confidence)
                        if len(person_keypoints) > 0:
                            # Extract x, y coordinates and confidence scores
                            kpts_xy = person_keypoints[:, :2]  # x, y coordinates
                            kpts_conf = person_keypoints[:, 2]  # confidence scores
                            
                            # Filter keypoints by confidence
                            valid_mask = kpts_conf > self.conf_threshold
                            kpts_xy[~valid_mask] = [0, 0]  # Set low confidence points to (0,0)
                            
                            keypoints_list.append(kpts_xy)
                            scores_list.append(kpts_conf)
                            
                            # Get track ID for this person
                            track_id = track_ids[idx] if track_ids is not None and idx < len(track_ids) else None
                            track_ids_list.append(track_id)
                            
                            # Align a box to this person if available
                            if boxes_xyxy is not None and len(boxes_xyxy) > idx:
                                boxes_list.append(boxes_xyxy[idx])
                                box_scores_list.append(float(boxes_conf[idx]) if boxes_conf is not None and len(boxes_conf) > idx else float(np.mean(kpts_conf)))
            
            # Convert to numpy arrays
            if keypoints_list:
                keypoints = np.array(keypoints_list)
                scores = np.array(scores_list)
            else:
                keypoints = None
                scores = None
            if boxes_list:
                boxes = np.array(boxes_list)
                box_scores = np.array(box_scores_list)
            else:
                boxes = None
                box_scores = None
            
            # Postprocess results - now includes track_ids
            results_dict = self.postprocess((keypoints, scores, boxes, box_scores, track_ids_list), frame.shape)
            
            return results_dict
            
        except Exception as e:
            print(f"Error in YOLO pose detection: {e}")
            return {'keypoints': None, 'scores': None, 'count': 0}
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLO pose detection"""
        # YOLO handles preprocessing internally, so we return the frame as-is
        return frame
    
    def postprocess(self, raw_results: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List], 
                   frame_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """Postprocess YOLO pose detection results - keep top-3 identities"""
        keypoints, scores, boxes, box_scores, track_ids = raw_results
        
        if keypoints is None:
            return {'keypoints': None, 'scores': None, 'count': 0}
        
        # Filter by confidence threshold and collect all valid detections
        valid_detections = []
        
        for i, person_scores in enumerate(scores):
            # Calculate average confidence for this person
            avg_confidence = np.mean(person_scores)
            
            # Check if this person meets the confidence threshold
            if avg_confidence > self.conf_threshold:
                detection_data = {
                    'keypoints': keypoints[i],
                    'scores': person_scores,
                    'confidence': avg_confidence,
                    'track_id': track_ids[i] if i < len(track_ids) else None,
                    'box': boxes[i] if boxes is not None and len(boxes) > i else None,
                    'box_confidence': float(box_scores[i]) if box_scores is not None and len(box_scores) > i else avg_confidence
                }
                valid_detections.append(detection_data)
        
        # Sort by confidence and keep top-3
        valid_detections.sort(key=lambda x: x['confidence'], reverse=True)
        top_detections = valid_detections[:1]  # Keep top-3 identities
        
        if top_detections:
            # Extract data for top detections
            top_keypoints = [det['keypoints'] for det in top_detections]
            top_scores = [det['scores'] for det in top_detections]
            top_track_ids = [det['track_id'] for det in top_detections]
            top_boxes = [det['box'] for det in top_detections]
            top_box_confidences = [det['box_confidence'] for det in top_detections]
            
            return {
                'keypoints': np.array(top_keypoints),
                'scores': np.array(top_scores),
                'count': len(top_detections),
                'keypoint_names': self.keypoint_names,
                'confidence': top_detections[0]['confidence'],  # Best confidence
                'track_ids': top_track_ids,  # Track IDs for all top detections
                'boxes': top_boxes,  # Boxes for all top detections
                'box_confidences': top_box_confidences  # Box confidences for all top detections
            }
        else:
            return {
                'keypoints': None,
                'scores': None,
                'count': 0,
                'keypoint_names': self.keypoint_names,
                'confidence': 0.0,
                'track_ids': None,
                'boxes': None,
                'box_confidences': None
            }
    
    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, 
                     scores: np.ndarray = None) -> np.ndarray:
        """Draw skeleton on frame"""
        if keypoints is None or len(keypoints) == 0:
            return frame
        
        # Define skeleton connections (COCO format)
        skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        frame_copy = frame.copy()
        
        for person_idx, person_keypoints in enumerate(keypoints):
            # Draw keypoints
            for i in range(len(person_keypoints)):
                try:
                    kp = person_keypoints[i]
                    # Extract x, y coordinates
                    x = float(kp[0])
                    y = float(kp[1])
                    
                    # Skip zero coordinates (invalid/low confidence keypoints)
                    if x == 0.0 and y == 0.0:
                        continue
                    
                    if scores is not None and scores[person_idx, i] > self.conf_threshold:
                        # Ensure coordinates are valid numbers and convert to int
                        if not np.isnan(x) and not np.isnan(y) and np.isfinite(x) and np.isfinite(y):
                            center = (int(x), int(y))
                            try:
                                cv2.circle(frame_copy, center, 5, (0, 255, 0), -1)
                            except Exception:
                                # Silently skip OpenCV drawing errors
                                pass
                except (ValueError, TypeError, OverflowError, IndexError):
                    # Skip invalid keypoint
                    continue
            
            # Draw skeleton connections
            for start_idx, end_idx in skeleton_connections:
                try:
                    if start_idx >= len(person_keypoints) or end_idx >= len(person_keypoints):
                        continue
                    
                    # Extract coordinates
                    start_x = float(person_keypoints[start_idx][0])
                    start_y = float(person_keypoints[start_idx][1])
                    end_x = float(person_keypoints[end_idx][0])
                    end_y = float(person_keypoints[end_idx][1])
                    
                    # Skip zero coordinates (invalid/low confidence keypoints)
                    if (start_x == 0.0 and start_y == 0.0) or (end_x == 0.0 and end_y == 0.0):
                        continue
                    
                    # Check scores if available
                    if scores is not None:
                        if scores[person_idx, start_idx] <= self.conf_threshold or scores[person_idx, end_idx] <= self.conf_threshold:
                            continue
                    
                    # Ensure coordinates are valid numbers
                    if (not np.isnan(start_x) and not np.isnan(start_y) and np.isfinite(start_x) and np.isfinite(start_y) and
                        not np.isnan(end_x) and not np.isnan(end_y) and np.isfinite(end_x) and np.isfinite(end_y)):
                        start_point = (int(start_x), int(start_y))
                        end_point = (int(end_x), int(end_y))
                        try:
                            cv2.line(frame_copy, start_point, end_point, (255, 0, 0), 2)
                        except Exception:
                            # Silently skip OpenCV drawing errors
                            pass
                except (ValueError, TypeError, OverflowError, IndexError):
                    # Skip invalid skeleton connection
                    continue
        
        return frame_copy
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get YOLO model information"""
        info = super().get_model_info()
        info.update({
            'model_path': self.model_path,
            'model_type': 'YOLO Pose',
            'keypoint_count': len(self.keypoint_names)
        })
        return info
