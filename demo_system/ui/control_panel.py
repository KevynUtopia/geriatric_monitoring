"""
Control panel component - adapted for YOLO detection
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QComboBox, QGroupBox, QCheckBox, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
from .styles import AppStyles
from .custom_widgets import SwitchControl
from core.translations import Translations as T
import cv2

class ControlPanel(QWidget):
    """Control panel component for YOLO detection"""
    
    # Define signals
    detection_changed = pyqtSignal(str)
    camera_changed = pyqtSignal(int)
    rotation_toggled = pyqtSignal(bool)
    skeleton_toggled = pyqtSignal(bool)
    boxes_toggled = pyqtSignal(bool)
    model_changed = pyqtSignal(str)
    mirror_toggled = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.detection_colors = AppStyles.DETECTION_COLORS
        
        # Initialize detection type mappings
        self.detection_display_map = {
            "pose_detection": T.get("pose_detection"),
            "object_detection": T.get("object_detection"),
            "action_recognition": T.get("action_recognition"),
            "custom_detection": T.get("custom_detection")
        }
        
        # Initialize model type mappings
        self.model_display_map = {
            "lightweight": T.get("lightweight"),
            "balanced": T.get("balanced"),
            "performance": T.get("performance")
        }
        
        # Initialize reverse mappings
        self.detection_code_map = {v: k for k, v in self.detection_display_map.items()}
        self.current_detection = "pose_detection"
        
        # Anomaly detection state
        self.anomaly_detected = False
        
        # Detect available cameras first
        print("Detecting available cameras...")
        self.available_cameras = self.detect_available_cameras()
        print(f"Detected cameras: {self.available_cameras}")
        
        # Setup layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(12)
        self.setup_ui()
        
        # Anomaly detection is now handled by real detection system
        # No need for random testing timer
    
    def setup_ui(self):
        """Setup control panel UI"""
        # Add detection info group
        self.setup_info_group()
        
        # Add control options group
        self.setup_controls_group()
        
        # Add detection status group
        self.setup_status_group()
        
        # Flexible vertical sharing without extra stretch that distorts alignment
    
    def setup_info_group(self):
        """Setup detection info group"""
        self.info_group = QGroupBox(T.get("detection_data"))
        self.info_group.setStyleSheet(AppStyles.get_group_box_style())
        info_layout = QVBoxLayout(self.info_group)
        info_layout.setSpacing(5)  # Reduce spacing between items
        
        # Create a grid layout for better alignment
        grid_layout = QGridLayout()
        grid_layout.setSpacing(8)
        
        # People Detected
        self.count_label = QLabel("People Detected:")
        self.count_label.setStyleSheet("color: #e6f1ff; font-size: 13pt; font-weight: 700;")
        self.count_label.setMinimumHeight(25)
        
        self.count_value = QLabel("0")
        self.count_value.setStyleSheet(AppStyles.get_counter_value_style("#7fb3ff"))
        self.count_value.setAlignment(Qt.AlignCenter)
        self.count_value.setFixedSize(90, 35)
        
        grid_layout.addWidget(self.count_label, 0, 0)
        grid_layout.addWidget(self.count_value, 0, 1)
        
        # FPS
        self.fps_label = QLabel("FPS:")
        self.fps_label.setStyleSheet("color: #e6f1ff; font-size: 13pt; font-weight: 700;")
        self.fps_label.setMinimumHeight(25)
        
        self.fps_value = QLabel("0")
        self.fps_value.setStyleSheet(AppStyles.get_counter_value_style("#7fe3a2"))
        self.fps_value.setAlignment(Qt.AlignCenter)
        self.fps_value.setFixedSize(90, 35)
        
        grid_layout.addWidget(self.fps_label, 1, 0)
        grid_layout.addWidget(self.fps_value, 1, 1)
        
        # Confidence
        self.conf_label = QLabel("Confidence:")
        self.conf_label.setStyleSheet("color: #e6f1ff; font-size: 13pt; font-weight: 700;")
        self.conf_label.setMinimumHeight(25)
        
        self.conf_value = QLabel("0%")
        self.conf_value.setStyleSheet(AppStyles.get_counter_value_style("#ff9f9f"))
        self.conf_value.setAlignment(Qt.AlignCenter)
        self.conf_value.setFixedSize(90, 35)
        
        grid_layout.addWidget(self.conf_label, 2, 0)
        grid_layout.addWidget(self.conf_value, 2, 1)
        
        info_layout.addLayout(grid_layout)
        self.layout.addWidget(self.info_group)
    
    def setup_controls_group(self):
        """Setup control options group"""
        self.controls_group = QGroupBox(T.get("control_options"))
        self.controls_group.setStyleSheet(AppStyles.get_group_box_style())
        controls_layout = QVBoxLayout(self.controls_group)
        controls_layout.setSpacing(12)
        
        # Detection type selection removed as requested
        # Model selection removed as requested
        
        # Camera selection
        camera_layout = QHBoxLayout()
        self.camera_label = QLabel("Camera:")
        self.camera_label.setStyleSheet("color: #f0f4ff; font-size: 14pt; font-weight: 700;")
        
        self.camera_combo = QComboBox()
        self.populate_camera_combo()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        self.camera_combo.setStyleSheet(AppStyles.get_camera_combo_style())
        
        # Add refresh button for cameras
        self.refresh_camera_button = QPushButton("↻")
        self.refresh_camera_button.setFixedSize(30, 30)
        self.refresh_camera_button.setToolTip("Refresh camera list")
        self.refresh_camera_button.clicked.connect(self.refresh_cameras)
        self.refresh_camera_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 15px;
                font-size: 14pt;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        
        camera_layout.addWidget(self.camera_label)
        camera_layout.addWidget(self.camera_combo, 1)
        camera_layout.addWidget(self.refresh_camera_button)
        
        # Add spacing
        spacer = QWidget()
        spacer.setMinimumHeight(5)
        controls_layout.addWidget(spacer)
        
        controls_layout.addLayout(camera_layout)
        
        # Portrait mode toggle
        self.rotation_switch = SwitchControl("Vertical Mode")
        self.rotation_switch.switched.connect(self._on_rotation_toggled)
        controls_layout.addWidget(self.rotation_switch)
        
        # Skeleton display toggle
        self.skeleton_switch = SwitchControl("Show Skeleton")
        self.skeleton_switch.switched.connect(self._on_skeleton_toggled)
        controls_layout.addWidget(self.skeleton_switch)
        
        # Boxes display toggle
        self.boxes_switch = SwitchControl("Show Boxes")
        self.boxes_switch.switched.connect(self._on_boxes_toggled)
        controls_layout.addWidget(self.boxes_switch)
        
        # Mirror mode toggle
        self.mirror_switch = SwitchControl("Mirror Mode")
        self.mirror_switch.switched.connect(self._on_mirror_toggled)
        controls_layout.addWidget(self.mirror_switch)
        
        # Add spacing
        spacer = QWidget()
        spacer.setMinimumHeight(5)
        controls_layout.addWidget(spacer)
        
        # Detection operation button row
        detection_buttons_layout = QHBoxLayout()
        detection_buttons_layout.setSpacing(10)  # Add spacing between buttons
        
        # Reset detection button
        self.reset_button = QPushButton("Reset")
        self.reset_button.setFixedSize(120, 45)
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14pt;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
            QPushButton:pressed {
                background-color: #6c7b7d;
            }
        """)
        self.reset_button.clicked.connect(self._on_reset_detection)
        detection_buttons_layout.addWidget(self.reset_button)

        # Save results button
        self.save_button = QPushButton("Save")
        self.save_button.setFixedSize(120, 45)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14pt;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        self.save_button.clicked.connect(self._on_save_results)
        detection_buttons_layout.addWidget(self.save_button)

        controls_layout.addLayout(detection_buttons_layout)
        
        self.layout.addWidget(self.controls_group)
    
    def setup_status_group(self):
        """Setup detection status group"""
        self.status_group = QGroupBox("Detection Status")
        self.status_group.setStyleSheet(AppStyles.get_group_box_style())
        self.status_group.setMinimumHeight(140)  # Increased height for better spacing
        status_layout = QVBoxLayout(self.status_group)
        status_layout.setSpacing(8)  # Increase spacing for better readability
        
        # Current detection status label
        status_label_layout = QHBoxLayout()
        self.status_title = QLabel("Current Status:")
        self.status_title.setStyleSheet("color: #e6f1ff; font-size: 14pt; font-weight: 700;")
        
        status_label_layout.addWidget(self.status_title)
        status_layout.addLayout(status_label_layout)
        
        # Create status indicators - two buttons horizontally aligned and evenly distributed
        status_indicator = QHBoxLayout()
        status_indicator.setSpacing(20)  # Spacing between indicators
        
        # First indicator - Human Detection (Blue when active)
        self.active_indicator = QLabel("●")
        self.active_indicator.setStyleSheet(AppStyles.get_status_indicator_style(False, color="#6ea8fe"))
        self.active_indicator.setAlignment(Qt.AlignCenter)
        self.active_indicator.setFixedSize(60, 60)
        
        self.inactive_indicator = QLabel("○")
        self.inactive_indicator.setStyleSheet(AppStyles.get_status_indicator_style(True, color="#6ea8fe"))
        self.inactive_indicator.setAlignment(Qt.AlignCenter)
        self.inactive_indicator.setFixedSize(60, 60)
        
        # Second indicator - Anomaly Detection (Red when active)
        self.anomaly_active_indicator = QLabel("●")
        self.anomaly_active_indicator.setStyleSheet(AppStyles.get_status_indicator_style(True, color="#ff6b6b"))
        self.anomaly_active_indicator.setAlignment(Qt.AlignCenter)
        self.anomaly_active_indicator.setFixedSize(60, 60)
        self.anomaly_active_indicator.hide()  # Initially hidden
        
        self.anomaly_inactive_indicator = QLabel("○")
        self.anomaly_inactive_indicator.setStyleSheet(AppStyles.get_status_indicator_style(False, color="#ff6b6b"))
        self.anomaly_inactive_indicator.setAlignment(Qt.AlignCenter)
        self.anomaly_inactive_indicator.setFixedSize(60, 60)
        
        # Add both indicators to layout with even distribution
        status_indicator.addStretch()
        status_indicator.addWidget(self.active_indicator)
        status_indicator.addWidget(self.inactive_indicator)
        status_indicator.addStretch()
        status_indicator.addWidget(self.anomaly_active_indicator)
        status_indicator.addWidget(self.anomaly_inactive_indicator)
        status_indicator.addStretch()
        
        # Add to layout
        status_layout.addLayout(status_indicator)
        
        # Add captions under each button
        captions_layout = QHBoxLayout()
        captions_layout.setSpacing(20)
        
        # Caption for first button (Detecting/Ready)
        self.status_value = QLabel("Ready")
        self.status_value.setStyleSheet("color: #8ab4ff; font-size: 14pt; font-weight: 700;")
        self.status_value.setAlignment(Qt.AlignCenter)
        
        # Caption for second button (Anomaly)
        self.anomaly_caption = QLabel("Anomaly")
        self.anomaly_caption.setStyleSheet("color: #ff7f7f; font-size: 14pt; font-weight: 700;")
        self.anomaly_caption.setAlignment(Qt.AlignCenter)
        
        # Add captions with even distribution matching buttons above
        captions_layout.addStretch()
        captions_layout.addWidget(self.status_value)
        captions_layout.addStretch()
        captions_layout.addWidget(self.anomaly_caption)
        captions_layout.addStretch()
        
        status_layout.addLayout(captions_layout)
        
        self.layout.addWidget(self.status_group)
        # Ensure equal widths visually
        self.info_group.setMinimumWidth(320)
        self.controls_group.setMinimumWidth(320)
        self.status_group.setMinimumWidth(320)
    
    def _on_detection_changed(self, detection_display):
        """Detection type change handler"""
        # Check if detection_display is empty or not in mapping
        if not detection_display or detection_display not in self.detection_code_map:
            return
            
        detection_code = self.detection_code_map[detection_display]
        self.current_detection = detection_code
        self.detection_changed.emit(detection_code)
        self.update_detection_style()
    
    def _on_reset_detection(self):
        """Reset detection handler"""
        self.count_value.setText("0")
        self.fps_value.setText("0")
        self.status_value.setText("Ready")
        self.update_status_indicators(False)
    
    def _on_save_results(self):
        """Save results handler"""
        # TODO: Implement save functionality
        self.save_button.setStyleSheet(AppStyles.get_success_button_style())
        QTimer.singleShot(1500, lambda: self.save_button.setStyleSheet(
            AppStyles.get_save_button_style()
        ))
    
    def _on_camera_changed(self, index):
        """Camera change handler"""
        if index >= 0 and index < len(self.available_cameras):
            camera_id = self.available_cameras[index]
            self.camera_changed.emit(camera_id)
            print(f"Camera selection changed to: Camera {camera_id}")
    
    def _on_rotation_toggled(self, checked):
        """Rotation mode toggle handler"""
        self.rotation_toggled.emit(checked)
    
    def _on_skeleton_toggled(self, checked):
        """Skeleton display toggle handler"""
        self.skeleton_toggled.emit(checked)

    def _on_boxes_toggled(self, checked):
        """Bounding boxes display toggle handler"""
        self.boxes_toggled.emit(checked)
    
    def _on_model_changed(self, index):
        """Model mode change handler"""
        model_mode = self.model_combo.currentData()
        self.model_changed.emit(model_mode)
    
    def _on_mirror_toggled(self, checked):
        """Mirror mode toggle handler"""
        self.mirror_toggled.emit(checked)
    
    def update_detection_count(self, count):
        """Update detection count display"""
        self.count_value.setText(str(count))
        if count > 0:
            self.show_detection_animation()
    
    def update_fps(self, fps):
        """Update FPS display"""
        self.fps_value.setText(f"{fps:.1f}")
    
    def update_confidence(self, confidence):
        """Update confidence display"""
        if confidence > 0:
            self.conf_value.setText(f"{confidence*100:.1f}%")
        else:
            self.conf_value.setText("0%")
    
    def update_status(self, status, is_active=False):
        """Update detection status display"""
        self.status_value.setText(status)
        self.update_status_indicators(is_active)
    
    def update_status_indicators(self, is_active):
        """Update status indicators"""
        if is_active:
            self.active_indicator.setStyleSheet(AppStyles.get_status_indicator_style(True))
            self.inactive_indicator.setStyleSheet(AppStyles.get_status_indicator_style(False))
        else:
            self.active_indicator.setStyleSheet(AppStyles.get_status_indicator_style(False))
            self.inactive_indicator.setStyleSheet(AppStyles.get_status_indicator_style(True))
    
    def show_detection_animation(self):
        """Show detection animation"""
        self.count_value.setStyleSheet(AppStyles.get_success_counter_style())
        QTimer.singleShot(1000, self.reset_detection_style)
    
    def update_detection_style(self):
        """Update detection style to current detection color"""
        try:
            current_detection = self.detection_display_map.get(self.current_detection, "")
            
            if current_detection in AppStyles.DETECTION_COLORS:
                current_color = AppStyles.DETECTION_COLORS[current_detection]
            else:
                current_color = "#3498db"  # Default use blue
                
            self.count_value.setStyleSheet(AppStyles.get_counter_value_style(current_color))
        except Exception as e:
            print(f"Error in update_detection_style: {e}")
            self.count_value.setStyleSheet(AppStyles.get_counter_value_style("#3498db"))
    
    def reset_detection_style(self):
        """Reset detection style"""
        self.update_detection_style()
    
    def update_anomaly_indicator(self, is_anomaly):
        """Update anomaly detection indicator
        
        Args:
            is_anomaly (bool): True if anomaly detected, False otherwise
        """
        if is_anomaly:
            self.anomaly_active_indicator.show()
            self.anomaly_inactive_indicator.hide()
        else:
            self.anomaly_active_indicator.hide()
            self.anomaly_inactive_indicator.show()
    
    
    def set_anomaly_status(self, is_anomaly):
        """Set anomaly status (to be called by external detection system)
        
        Args:
            is_anomaly (bool): True if anomaly detected, False otherwise
        """
        self.anomaly_detected = is_anomaly
        self.update_anomaly_indicator(is_anomaly)
    
    def detect_available_cameras(self):
        """Detect available cameras in the system"""
        available_cameras = []
        
        # Test cameras from 0 to 9
        for camera_id in range(10):
            try:
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    # Try to read a frame to confirm camera is working
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available_cameras.append(camera_id)
                        print(f"Camera {camera_id} detected and working")
                    cap.release()
                else:
                    cap.release()
            except Exception as e:
                print(f"Error testing camera {camera_id}: {e}")
                continue
        
        if not available_cameras:
            print("No cameras detected, defaulting to camera 0")
            available_cameras = [0]
        
        return available_cameras
    
    def populate_camera_combo(self):
        """Populate camera combo box with available cameras"""
        self.camera_combo.clear()
        
        for camera_id in self.available_cameras:
            self.camera_combo.addItem(f"Camera {camera_id}", camera_id)
        
        # Set default to first available camera
        if self.available_cameras:
            self.camera_combo.setCurrentIndex(0)
    
    def refresh_cameras(self):
        """Refresh the list of available cameras"""
        self.available_cameras = self.detect_available_cameras()
        self.populate_camera_combo()
        print(f"Refreshed cameras: {self.available_cameras}")
