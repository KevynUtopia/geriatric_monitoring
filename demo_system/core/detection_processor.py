"""
Main detection processor - coordinates multiple detection models
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from PyQt5.QtCore import QObject, pyqtSignal
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

class DetectionProcessor(QObject):
    """Main detection processor that coordinates multiple detection models"""
    
    # Define signal for anomaly detection
    anomaly_detected = pyqtSignal(bool)
    
    def __init__(self, mode='balanced', device='cuda:0'):
        super().__init__()
        self.mode = mode
        self.device = device
        self.show_skeleton = True
        self.conf_threshold = 0.5
        
        # Initialize detection models
        self.models = {}
        self.init_models()
        
        # Detection results storage
        self.current_results = {}
        self.detection_history = []
        
        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()

        self.clip = deque(maxlen=5)
        self.boxes = deque(maxlen=5)
        self.action_results = None
        self.action_every_n_frames = 5  # run action head when we have N frames

        self.ids = deque(maxlen=5)
        self.sampling_count = 0
        # Thread pool for async action recognition
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    def init_models(self):
        """Initialize all detection models"""
        try:
            # Initialize YOLO pose detection model
            from models.pose_detector import PoseDetector
            self.models['pose'] = PoseDetector(mode=self.mode, device=self.device)
            print("YOLO pose detection model initialized")

            from models.action_recognizer import ActionRecognizer
            self._action_model = ActionRecognizer(device=self.device, num_actions=10)
            
            # For now, only initialize pose detection
            # Other models can be added later as needed
            
        except ImportError as e:
            print(f"Warning: YOLO model could not be initialized: {e}")
            print("Please ensure ultralytics is installed: pip install ultralytics")
        except Exception as e:
            print(f"Error initializing YOLO model: {e}")
    
    def process_frame(self, frame: np.ndarray, detection_types: List[str] = None) -> Dict[str, Any]:
        """Process single frame with specified detection types"""
        if detection_types is None:
            detection_types = ['pose']  # Focus on pose detection for now
        
        results = {}
        
        try:
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)

            # Run detection models
            for detection_type in detection_types:
                if detection_type in self.models:
                    try:
                        model_results = self.models[detection_type].detect(processed_frame)
                        results[detection_type] = model_results
                    except Exception as e:
                        print(f"Error in {detection_type} detection: {e}")
                        results[detection_type] = None
            
            # Post-process results
            results = self.postprocess_results(results, frame.shape)

            self.sampling_count += 1
            if self.sampling_count % 5 != 0:
                return results
            # Schedule async action recognition every 5th sample
            pose_res = results.get('pose') or {}
            pose_box = pose_res.get('boxes')
            pose_ids = pose_res.get('track_ids')
            self._schedule_action_recognition(processed_frame, pose_box, pose_ids)
            
            # Update FPS counter
            self.update_fps()
            
        except Exception as e:
            print(f"Error in frame processing: {e}")
            results = {}
        
        return results

    def _schedule_action_recognition(self, processed_frame: np.ndarray, pose_box, pose_ids):
        """Wrap and schedule action recognition asynchronously.
        Submits a worker to a thread pool every 5th frame and discards
        results that arrive after 2 seconds.
        """
        try:
            # Update clips and boxes queues
            self.clip.append(processed_frame.astype(np.float32))
            if pose_box is not None:
                self.boxes.append(np.array(pose_box, dtype=np.float32))
                self.ids.append(np.array(pose_ids, dtype=np.int32))

            # Update current results snapshot timing
            self.current_results_timestamp = time.time()

            # Only run when we have enough frames accumulated
            if len(self.clip) != self.clip.maxlen or len(self.boxes) != self.boxes.maxlen:
                return

            # Prepare immutable copies for the worker
            clip_list = list(self.clip)
            boxes_list = list(self.boxes)
            ids_list = list(self.ids)
            submit_time = time.time()

            def _worker(clip_arg, boxes_arg, ids_arg):
                return self._action_model.recognize(clip_arg, boxes_arg, ids_arg)

            future = self._executor.submit(_worker, clip_list, boxes_list, ids_list)

            def _on_done(fut):
                try:
                    result = fut.result()
                except Exception as e:
                    print(f"Action recognition failed: {e}")
                    return
                # Discard late results (>2s)
                if time.time() - submit_time > 2.0:
                    print("Action recognition result discarded due to >2s delay")
                    return

                self.action_results = result

                # Emit anomaly signal if anomaly is detected
                if self.action_results.get('anomaly_detected', False):
                    self.anomaly_detected.emit(True)
                else:
                    self.anomaly_detected.emit(False)

                # Print recognized action logits to console/terminal
                try:
                    logits_print = np.array(self.action_results.get('logits')).round(4)
                    print(f"Action logits (T={self.action_results.get('num_frames')}): {logits_print}")
                except Exception:
                    pass

            future.add_done_callback(_on_done)
        except Exception as e:
            print(f"Error scheduling action recognition: {e}")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for detection"""
        # Resize if frame is too large
        h, w = frame.shape[:2]
        if w > 640 or h > 640:
            scale = min(640/w, 640/h)
            new_w, new_h = int(w*scale), int(h*scale)
            frame = cv2.resize(frame, (new_w, new_h))
        
        return frame
    
    def postprocess_results(self, results: Dict[str, Any], original_shape: Tuple[int, int, int]) -> Dict[str, Any]:
        """Post-process detection results"""
        # Scale results back to original frame size if needed
        # Add any additional post-processing logic here
        
        # Store results in history
        self.detection_history.append({
            'timestamp': time.time(),
            'results': results.copy()
        })
        
        # Keep only recent history (last 100 frames)
        if len(self.detection_history) > 100:
            self.detection_history = self.detection_history[-100:]
        
        return results
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get summary of current detection results"""
        summary = {
            'timestamp': time.time(),
            'fps': getattr(self, 'fps', 0),
            'active_models': list(self.models.keys()),
            'results': self.current_results
        }
        return summary
    
    def set_skeleton_visibility(self, show: bool):
        """Set skeleton display state"""
        self.show_skeleton = show
        print(f"Skeleton display: {'On' if show else 'Off'}")
    
    def update_model(self, mode: str):
        """Update detection models"""
        print(f"Updating detection models to mode: {mode}")
        self.mode = mode
        self.init_models()
        print(f"Detection processor updated to mode: {mode}")
    
    def cleanup(self):
        """Clean up resources"""
        for model_name, model in self.models.items():
            if hasattr(model, 'cleanup'):
                model.cleanup()
        try:
            if hasattr(self, '_executor') and self._executor is not None:
                self._executor.shutdown(wait=False)
        except Exception:
            pass
        print("Detection processor cleaned up")

