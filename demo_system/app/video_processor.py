"""
Video processor - handles video processing and detection coordination
"""

from PyQt5.QtCore import QObject, pyqtSignal
import cv2
import numpy as np
import time
import random
from datetime import datetime, timedelta

class VideoProcessor(QObject):
    """Handles video processing and detection coordination"""
    
    # Signals
    detection_results_ready = pyqtSignal(dict)
    fps_updated = pyqtSignal(float)
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.current_detection_types = ['pose', 'object', 'action']
        self.show_skeleton = True
        self.mirror_mode = True
        self.rotate_mode = True
        self.show_boxes = True
        
        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        
        # Persistence of last valid detection
        # If the current frame has no detection, we will keep displaying the last
        # valid detection for a limited number of consecutive frames
        self.last_valid_detection_results = None
        self.no_detection_streak = 0
        self.no_detection_streak_limit = 20  # After 20 consecutive empty frames, clear
    
    def update_image(self, frame, fps=0):
        """Update image display and process detection"""
        try:
            # Update FPS
            self.current_fps = fps
            self.fps_updated.emit(fps)

            # --- Live dummy telemetry updates ---
            try:
                # Generate a random value in [0, 1] per frame
                dummy_value = random.random()
                if hasattr(self.main_window, 'temporal_curve') and self.main_window.temporal_curve is not None:
                    self.main_window.temporal_curve.update_with_value(dummy_value)

                # Threshold and append anomaly message in HKT (+8) when > 0.95
                if dummy_value > 0.95 and hasattr(self.main_window, 'text_log') and self.main_window.text_log is not None:
                    # Current HKT time: UTC + 8 hours
                    hkt_dt = datetime.utcnow() + timedelta(hours=8)
                    msg = hkt_dt.strftime('%y-%m-%d %H:%M:%S') + ' anomaly'
                    self.main_window.text_log.append_message(msg)
            except Exception:
                # Non-fatal; continue with normal processing
                pass
            
            # Get detection results from the latest inference frame
            detection_results = {}
            if hasattr(self.main_window, 'latest_detection_results'):
                detection_results = self.main_window.latest_detection_results
                
                # Scale keypoints from inference frame to display frame
                if 'pose' in detection_results and detection_results['pose']:
                    detection_results = self.scale_detection_results(detection_results, frame)
            
            # Update video display with detection overlays
            if hasattr(self.main_window, 'video_display'):
                # Pass show_boxes and show_skeleton state to renderer through results dict
                if isinstance(detection_results, dict):
                    detection_results = {**detection_results}
                    detection_results['show_boxes'] = getattr(self, 'show_boxes', True)
                    detection_results['show_skeleton'] = getattr(self, 'show_skeleton', True)
                self.main_window.video_display.update_image(frame, detection_results)
            else:
                self.main_window.video_display.update_image(frame)
            
            # Update control panel
            if hasattr(self.main_window, 'control_panel'):
                self.update_control_panel(detection_results)
            
        except Exception as e:
            print(f"Error in video processing: {e}")
    
    def process_inference_frame(self, inference_frame):
        """Process inference frame asynchronously"""
        try:
            # Process detection on the inference frame
            if hasattr(self.main_window, 'detection_processor'):
                detection_results = self.main_window.detection_processor.process_frame(
                    inference_frame, self.current_detection_types
                )
                
                # Determine if there is any detection in current results
                total_detections = 0
                if isinstance(detection_results, dict):
                    for _, res in detection_results.items():
                        if isinstance(res, dict):
                            total_detections += int(res.get('count', 0))
                
                if total_detections > 0:
                    # We have valid detections - reset streak and remember last valid
                    self.no_detection_streak = 0
                    self.last_valid_detection_results = detection_results
                    # Store current results for scaling in update_image
                    self.main_window.latest_detection_results = detection_results
                else:
                    # No detections in this frame
                    self.no_detection_streak += 1
                    if (
                        self.last_valid_detection_results is not None and
                        self.no_detection_streak < self.no_detection_streak_limit
                    ):
                        # Use the last valid detection for display until limit reached
                        self.main_window.latest_detection_results = self.last_valid_detection_results
                    else:
                        # If limit reached, clear the last valid results
                        if self.no_detection_streak >= self.no_detection_streak_limit:
                            self.last_valid_detection_results = None
                        # Propagate empty results
                        self.main_window.latest_detection_results = detection_results
                
        except Exception as e:
            print(f"Error processing inference frame: {e}")
    
    def scale_detection_results(self, detection_results, display_frame):
        """Scale detection results from inference frame to display frame"""
        try:
            if not hasattr(self.main_window, 'video_thread'):
                return detection_results
                
            # Get frame dimensions
            display_h, display_w = display_frame.shape[:2]
            inference_h = self.main_window.video_thread.inference_height
            inference_w = self.main_window.video_thread.inference_width
            
            # Calculate scaling factors
            scale_x = display_w / inference_w
            scale_y = display_h / inference_h
            
            # Scale pose keypoints
            if 'pose' in detection_results and detection_results['pose']:
                pose_results = detection_results['pose'].copy()
                if pose_results.get('keypoints') is not None:
                    keypoints = pose_results['keypoints'].copy()
                    
                    # Scale each person's keypoints
                    for person_idx in range(len(keypoints)):
                        for kp_idx in range(len(keypoints[person_idx])):
                            # Scale x and y coordinates
                            keypoints[person_idx][kp_idx][0] *= scale_x
                            keypoints[person_idx][kp_idx][1] *= scale_y
                    
                    pose_results['keypoints'] = keypoints
                    detection_results['pose'] = pose_results
                
                # Scale bounding boxes if present (now supports multiple boxes)
                if pose_results.get('boxes') is not None:
                    boxes = pose_results['boxes'].copy()
                    scaled_boxes = []
                    for box in boxes:
                        if box is not None:
                            box_array = np.array(box).astype(float)
                            # box = [x1, y1, x2, y2]
                            box_array[0] *= scale_x
                            box_array[2] *= scale_x
                            box_array[1] *= scale_y
                            box_array[3] *= scale_y
                            scaled_boxes.append(box_array)
                        else:
                            scaled_boxes.append(None)
                    pose_results['boxes'] = scaled_boxes
                    detection_results['pose'] = pose_results
            
            return detection_results
            
        except Exception as e:
            print(f"Error scaling detection results: {e}")
            return detection_results
    
    def update_control_panel(self, detection_results):
        """Update control panel with detection results"""
        try:
            if not hasattr(self.main_window, 'control_panel'):
                return
            
            # Count total detections and get confidence
            total_detections = 0
            confidence = 0.0
            
            for detection_type, results in detection_results.items():
                if results and isinstance(results, dict):
                    count = results.get('count', 0)
                    total_detections += count
                    
                    # Get confidence score from pose detection
                    if detection_type == 'pose' and count > 0:
                        confidence = results.get('confidence', 0.0)
            
            # Update control panel
            self.main_window.control_panel.update_detection_count(total_detections)
            self.main_window.control_panel.update_fps(self.current_fps)
            self.main_window.control_panel.update_confidence(confidence)
            
            # Update status
            if total_detections > 0:
                self.main_window.control_panel.update_status("Detecting", True)
            else:
                self.main_window.control_panel.update_status("Ready", False)
                
        except Exception as e:
            print(f"Error updating control panel: {e}")
    
    def change_camera(self, camera_id):
        """Change camera source"""
        try:
            if hasattr(self.main_window, 'video_thread'):
                self.main_window.video_thread.set_camera(camera_id)
                print(f"Switched to camera {camera_id}")
        except Exception as e:
            print(f"Error changing camera: {e}")
    
    def toggle_rotation(self, rotate):
        """Toggle video rotation"""
        try:
            self.rotate_mode = rotate
            if hasattr(self.main_window, 'video_thread'):
                self.main_window.video_thread.set_rotation(rotate)
                print(f"Video rotation: {'On' if rotate else 'Off'}")
        except Exception as e:
            print(f"Error toggling rotation: {e}")
    
    def toggle_skeleton(self, show):
        """Toggle skeleton display"""
        try:
            self.show_skeleton = show
            if hasattr(self.main_window, 'detection_processor'):
                self.main_window.detection_processor.set_skeleton_visibility(show)
                print(f"Skeleton display: {'On' if show else 'Off'}")
        except Exception as e:
            print(f"Error toggling skeleton: {e}")
    
    def toggle_boxes(self, show):
        """Toggle bounding boxes display"""
        try:
            self.show_boxes = show
            print(f"Boxes display: {'On' if show else 'Off'}")
        except Exception as e:
            print(f"Error toggling boxes: {e}")
    
    def toggle_mirror(self, mirror):
        """Toggle mirror mode"""
        try:
            self.mirror_mode = mirror
            if hasattr(self.main_window, 'video_thread'):
                self.main_window.video_thread.set_mirror(mirror)
                print(f"Mirror mode: {'On' if mirror else 'Off'}")
        except Exception as e:
            print(f"Error toggling mirror: {e}")
    
    def open_video_file(self):
        """Open video file"""
        try:
            from PyQt5.QtWidgets import QFileDialog
            from core.translations import Translations as T
            
            file_path, _ = QFileDialog.getOpenFileName(
                self.main_window,
                T.get("open_video"),
                "",
                T.get("video_files")
            )
            
            if file_path:
                if hasattr(self.main_window, 'video_thread'):
                    self.main_window.video_thread.set_video_file(file_path)
                    print(f"Video file loaded: {file_path}")
                else:
                    print("Video thread not available")
        except Exception as e:
            print(f"Error opening video file: {e}")
    
    def switch_to_camera_mode(self):
        """Switch back to camera mode"""
        try:
            if hasattr(self.main_window, 'video_thread'):
                self.main_window.video_thread.set_camera(0)
                print("Switched to camera mode")
        except Exception as e:
            print(f"Error switching to camera mode: {e}")
    
    def change_model(self, model_mode):
        """Change detection model"""
        try:
            if hasattr(self.main_window, 'detection_processor'):
                self.main_window.detection_processor.update_model(model_mode)
                print(f"Model changed to: {model_mode}")
        except Exception as e:
            print(f"Error changing model: {e}")
    
    def update_model_mode(self, model_mode):
        """Update model mode for async processing"""
        try:
            # This can be used for async model updates
            self.change_model(model_mode)
        except Exception as e:
            print(f"Error updating model mode: {e}")
    
    def cleanup(self):
        """Clean up video processor resources"""
        try:
            print("Video processor cleaned up")
        except Exception as e:
            print(f"Error cleaning up video processor: {e}")

