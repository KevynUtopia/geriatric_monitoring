"""
Video display component - adapted for YOLO detection results
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy, QFrame
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter
import cv2
import numpy as np

class VideoDisplay(QWidget):
    """Video display component for YOLO detection results"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # Transparent dark container; actual frame uses parent frame
        self.setStyleSheet("background-color: rgba(0,0,0,0);")
        
        # Layout setup - use center alignment
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.layout.setAlignment(Qt.AlignCenter)
        
        # Create image label for video display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: none; background-color: rgba(0,0,0,0);")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Auto-expand to fit container
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.layout.addWidget(self.image_label, 0, Qt.AlignCenter)
        
        # Default settings
        self.is_portrait = True
        self.aspect_ratio = 9/16  # Default portrait ratio
        self.current_frame_size = None  # Track current frame size
        self.set_orientation(self.is_portrait)
    
    def update_image(self, frame, detection_results=None):
        """Update image display with optional YOLO detection results overlay"""
        try:
            # Apply detection overlays if results are provided
            if detection_results:
                frame = self.draw_detection_overlays(frame, detection_results)
            
            # Convert BGR to RGB for proper color display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert OpenCV format to QImage
            h, w, ch = rgb_frame.shape
            
            # Update current frame size
            self.current_frame_size = (w, h)
            
            bytes_per_line = ch * w
            convert_to_qt_format = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Detect frame aspect ratio and update settings
            frame_aspect_ratio = w / h
            self.update_aspect_ratio(frame_aspect_ratio)
            
            # Calculate display area size
            label_width = self.width()
            label_height = self.height()
            
            if label_width > 0 and label_height > 0:
                # Use high-quality scaling, maintain aspect ratio
                qt_img = convert_to_qt_format.scaled(
                    label_width,
                    label_height,
                    Qt.KeepAspectRatio,  # Maintain aspect ratio, avoid distortion
                    Qt.SmoothTransformation  # Use smooth transformation for better quality
                )
            else:
                qt_img = convert_to_qt_format
            
            # Create high-quality pixmap
            pixmap = QPixmap.fromImage(qt_img)
            
            # Set high-quality rendering
            self.image_label.setPixmap(pixmap)
            
            # Don't use setScaledContents, let Qt handle scaling automatically
            self.image_label.setScaledContents(False)
            
        except Exception as e:
            print(f"Error updating image: {e}")
    
    def draw_detection_overlays(self, frame, detection_results):
        """Draw YOLO detection results overlays on frame"""
        try:
            frame_copy = frame.copy()
            
            # Draw pose detection results
            if 'pose' in detection_results and detection_results['pose']:
                pose_results = detection_results['pose']
                if pose_results.get('keypoints') is not None and detection_results.get('show_skeleton', True):
                    frame_copy = self.draw_pose_skeleton(frame_copy, pose_results)
                
                # Draw bounding boxes for all top detections if boxes are enabled
                if detection_results.get('show_boxes', True):
                    boxes = pose_results.get('boxes')
                    box_confidences = pose_results.get('box_confidences')
                    track_ids = pose_results.get('track_ids')
                    
                    if boxes is not None:
                        for i, box in enumerate(boxes):
                            if box is not None:
                                try:
                                    box_array = np.array(box, dtype=float).reshape(-1)
                                    if box_array.size >= 4 and np.all(np.isfinite(box_array[:4])):
                                        x1f, y1f, x2f, y2f = box_array[:4]
                                        # Ensure proper ordering
                                        x1f, x2f = (x1f, x2f) if x1f <= x2f else (x2f, x1f)
                                        y1f, y2f = (y1f, y2f) if y1f <= y2f else (y2f, y1f)
                                        # Clip to frame bounds
                                        h_, w_ = frame_copy.shape[:2]
                                        x1 = int(max(0, min(w_-1, x1f)))
                                        y1 = int(max(0, min(h_-1, y1f)))
                                        x2 = int(max(0, min(w_-1, x2f)))
                                        y2 = int(max(0, min(h_-1, y2f)))
                                        # Valid rectangle check
                                        if x2 > x1 and y2 > y1:
                                            # Different colors for different identities
                                            colors = [(0, 200, 255), (0, 255, 0), (255, 0, 255)]  # Blue, Green, Magenta
                                            color = colors[i % len(colors)]
                                            
                                            try:
                                                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                                            except Exception:
                                                # Silently skip OpenCV rectangle drawing errors
                                                pass
                                            
                                            # Prepare label with confidence and track ID
                                            conf = float(box_confidences[i]) if box_confidences and i < len(box_confidences) else 0.0
                                            track_id = track_ids[i] if track_ids and i < len(track_ids) else None
                                            
                                            if track_id is not None:
                                                label = f"ID:{track_id} {conf*100:.1f}%"
                                            else:
                                                label = f"{conf*100:.1f}%"
                                            
                                            try:
                                                cv2.putText(frame_copy, label, (x1, max(0, y1-6)), 
                                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                                            except Exception:
                                                # Silently skip OpenCV text drawing errors
                                                pass
                                except Exception as e:
                                    # Skip box drawing if anything goes wrong
                                    pass
            
            return frame_copy
            
        except Exception:
            # Silently skip any drawing errors
            return frame
    
    def draw_pose_skeleton(self, frame, pose_results):
        """Draw YOLO pose skeleton on frame"""
        try:
            keypoints = pose_results['keypoints']
            scores = pose_results.get('scores')
            
            if keypoints is None or len(keypoints) == 0:
                return frame
            
            # Define skeleton connections (COCO format)
            skeleton_connections = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                (5, 11), (6, 12), (11, 12),  # Torso
                (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
            ]
            
            # Different colors for different identities
            colors = [(0, 200, 255), (0, 255, 0), (255, 0, 255)]  # Blue, Green, Magenta
            keypoint_colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255)]  # Green, Yellow, Magenta
            
            for person_idx, person_keypoints in enumerate(keypoints):
                # Get color for this person
                person_color = colors[person_idx % len(colors)]
                keypoint_color = keypoint_colors[person_idx % len(keypoint_colors)]
                
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
                        
                        if scores is None or scores[person_idx, i] > 0.5:
                            # Validate coordinates before drawing
                            if not np.isnan(x) and not np.isnan(y) and np.isfinite(x) and np.isfinite(y):
                                center = (int(x), int(y))
                                try:
                                    cv2.circle(frame, center, 5, keypoint_color, -1)
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
                            if scores[person_idx, start_idx] <= 0.5 or scores[person_idx, end_idx] <= 0.5:
                                continue
                        
                        # Validate coordinates before drawing
                        if (not np.isnan(start_x) and not np.isnan(start_y) and np.isfinite(start_x) and np.isfinite(start_y) and
                            not np.isnan(end_x) and not np.isnan(end_y) and np.isfinite(end_x) and np.isfinite(end_y)):
                            start_point = (int(start_x), int(start_y))
                            end_point = (int(end_x), int(end_y))
                            try:
                                cv2.line(frame, start_point, end_point, person_color, 2)
                            except Exception:
                                # Silently skip OpenCV drawing errors
                                pass
                    except (ValueError, TypeError, OverflowError, IndexError):
                        # Skip invalid skeleton connection
                        continue
            
            return frame
            
        except Exception:
            # Silently skip any drawing errors
            return frame
    
    def resizeEvent(self, event):
        """Maintain aspect ratio when component is resized"""
        super().resizeEvent(event)
        # When component size changes, adjust internal label size
        self.adjust_size()
        
    def adjust_size(self):
        """Adjust internal label size to fit container size while maintaining aspect ratio"""
        width = self.width()
        height = self.height()
        
        # Calculate target height or width
        target_height = width / self.aspect_ratio
        target_width = height * self.aspect_ratio
        
        # Decide which dimension to use based on container size
        if target_height <= height:
            # Use width as base
            new_size = (width, int(target_height))
            margin_h = 0
            margin_v = (height - int(target_height)) // 2
        else:
            # Use height as base
            new_size = (int(target_width), height)
            margin_h = (width - int(target_width)) // 2
            margin_v = 0
            
        # Set label size and margins
        self.layout.setContentsMargins(margin_h, margin_v, margin_h, margin_v)
        
    def update_aspect_ratio(self, frame_aspect_ratio):
        """Update display settings based on actual frame aspect ratio
        
        Args:
            frame_aspect_ratio (float): Frame aspect ratio
        """
        # Determine if it's a vertical video
        is_vertical = frame_aspect_ratio < 1
        
        # Use actual frame aspect ratio
        self.aspect_ratio = frame_aspect_ratio
        
        # Update orientation flag
        self.is_portrait = is_vertical
        
    def set_orientation(self, portrait_mode=True):
        """Set video display orientation
        
        Args:
            portrait_mode (bool): True for portrait mode (9:16), False for landscape mode (16:9)
        """
        self.is_portrait = portrait_mode
        
        if portrait_mode:
            # Portrait mode - 9:16 ratio
            self.aspect_ratio = 9/16
            self.image_label.setMinimumSize(360, 640)
        else:
            # Landscape mode - 16:9 ratio
            self.aspect_ratio = 16/9
            self.image_label.setMinimumSize(640, 360)
        
        # Adjust size to fit new aspect ratio
        self.adjust_size()
