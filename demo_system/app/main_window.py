"""
Main window class - responsible for basic UI setup and signal connections
"""

import sys
import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QStatusBar, QMessageBox, QAction, QActionGroup, QMenu, QFileDialog, QLabel)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont

from core.video_thread import VideoThread
from core.detection_processor import DetectionProcessor
from core.sound_manager import SoundManager
from core.data_manager import DataManager
from core.translations import Translations as T
from ui.video_display import VideoDisplay
from ui.control_panel import ControlPanel
from ui.stats_panel import StatsPanel
from ui.styles import AppStyles
from ui.console_widget import ConsoleWidget, OutputRedirector

from .mode_manager import ModeManager
from .menu_manager import MenuManager
from .stats_manager import StatsManager
from .video_processor import VideoProcessor
from .detection_manager import DetectionManager

class DetectionApp(QMainWindow):
    """AI Detection Template Main Window Class"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(T.get("app_title"))
        self.setMinimumSize(1200, 800)
        
        # Initialize core components
        self._init_core_components()
        
        # Initialize managers
        self._init_managers()
        
        # Create UI
        self.setup_ui()
        
        # Setup console output redirection
        self._setup_console_redirection()
        
        # Initialize video thread
        self.setup_video_thread()
        
        # Initialize panels
        self._init_panels()
        
        # Start video processing
        self.start_video()
        
        # Initialize state variables
        self._init_state_variables()
        
        # Show welcome message
        self.statusBar.showMessage(f"{T.get('welcome')} - Personalized Geriatric Wellness Monitoring")
    
    def _init_core_components(self):
        """Initialize core components"""
        # Device settings
        self.device = 'cuda:0'
        self.model_mode = 'balanced'
        
        # Initialize detection processor
        print(f"Initializing detection processor (mode: {self.model_mode}, device: {self.device})")
        self.detection_processor = DetectionProcessor(
            mode=self.model_mode,
            device=self.device
        )
        
        # Set default detection type
        self.detection_type = "pose_detection"
        
        # Create sound manager
        self.sound_manager = SoundManager()
        
        # Create data manager
        self.data_manager = DataManager()
    
    def _init_managers(self):
        """Initialize managers"""
        # Mode manager
        self.mode_manager = ModeManager(self)
        
        # Menu manager
        self.menu_manager = MenuManager(self)
        
        # Statistics manager
        self.stats_manager = StatsManager(self)
        
        # Video processor
        self.video_processor = VideoProcessor(self)
        
        # Detection manager
        self.detection_manager = DetectionManager(self)
    
    def _init_panels(self):
        """Initialize panels"""
        # Initialize statistics panel
        self.stats_manager.init_stats_panel()
        # Assign stats panel to main window for easy access
        self.stats_panel = self.stats_manager.stats_panel
    
    def _init_state_variables(self):
        """Initialize state variables"""
        # Current detection results
        self.current_results = {}
        self.latest_detection_results = {}
        
        # Detection history
        self.detection_history = []
        
        # Default stats panel visibility
        self.stats_panel.setVisible(False)
        
        # Mirror mode
        self.mirror_mode = True
    
    def setup_ui(self):
        """Setup user interface"""
        # Apply styles
        self.setPalette(AppStyles.get_window_palette())
        self.setStyleSheet(AppStyles.get_global_stylesheet())
        
        # Create main window layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # R1: Title (top middle)
        self.title_label = QLabel("HKUST: Personalized Geriatric Wellness Monitoring through Biomarker Stability Assessment")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(
            """
            QLabel {
                font-size: 20pt;
                font-weight: 700;
                color: #e6f1ff;
                letter-spacing: 0.5px;
                padding: 10px 14px;
                background-color: rgba(0, 0, 0, 0.2);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 12px;
            }
            """
        )
        main_layout.addWidget(self.title_label)

        # R2 + R3: Main content area (left: info/controls/status; right: video+console)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        # R2: Left column
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # Control panel already contains: detection data, control options, detection status
        self.control_panel = ControlPanel()
        if hasattr(self.control_panel, 'boxes_toggled'):
            self.control_panel.boxes_toggled.connect(self.toggle_boxes)
        # Force uniform widths via size policy
        self.control_panel.setStyleSheet("QGroupBox{min-width: 320px;} ")
        left_layout.addWidget(self.control_panel)

        # R3: Right column (video on top, console at bottom), aligned same width
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        # Video window container with subtle frame (placeholder then filled with frames)
        self.video_display = VideoDisplay()
        self.video_display.setStyleSheet(
            """
            QWidget {
                background-color: rgba(0,0,0,0.6);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 12px;
            }
            """
        )
        right_layout.addWidget(self.video_display, stretch=6)

        # Console area aligned with video width (outer frame handled inside ConsoleWidget)
        self.console_widget = ConsoleWidget()
        right_layout.addWidget(self.console_widget, stretch=4)

        # Ensure width ratio: R2 35% (left), R3 65% (right)
        content_layout.addWidget(left_widget, 35)
        content_layout.addWidget(right_widget, 65)

        main_layout.addLayout(content_layout)
        
        # Add status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage(T.get("ready"))
        
        # Setup menu bar
        self.menu_manager.setup_menu_bar()
        
        # Connect control panel signals
        self.connect_signals()

    
    def _setup_console_redirection(self):
        """Setup console output redirection to GUI console"""
        # Create output redirectors
        self.stdout_redirector = OutputRedirector('stdout')
        self.stderr_redirector = OutputRedirector('stderr')
        
        # Connect redirector signals to console widget
        self.stdout_redirector.output_signal.connect(self.console_widget.append_output)
        self.stderr_redirector.output_signal.connect(self.console_widget.append_output)
        
        # Redirect sys.stdout and sys.stderr
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stderr_redirector
        
        print("Console output redirection initialized - all print statements will appear here")
    
    def connect_signals(self):
        """Connect signals and slots"""
        # Connect control panel signals
        self.control_panel.detection_changed.connect(self.change_detection)
        self.control_panel.camera_changed.connect(self.change_camera)
        self.control_panel.rotation_toggled.connect(self.toggle_rotation)
        self.control_panel.skeleton_toggled.connect(self.toggle_skeleton)
        self.control_panel.model_changed.connect(self.change_model)
        self.control_panel.mirror_toggled.connect(self.toggle_mirror)
        
        # Connect stats panel signals
        if hasattr(self, 'stats_panel'):
            self.stats_panel.data_updated.connect(self.update_stats)
        
        # Connect anomaly detection signal to control panel
        self.detection_processor.anomaly_detected.connect(self.control_panel.set_anomaly_status)
    
    def setup_video_thread(self):
        """Setup video processing thread"""
        # Get the first available camera from control panel
        initial_camera_id = 0
        try:
            if hasattr(self, 'control_panel') and hasattr(self.control_panel, 'available_cameras'):
                if self.control_panel.available_cameras:
                    initial_camera_id = self.control_panel.available_cameras[0]
                    print(f"Using camera {initial_camera_id} from detected cameras: {self.control_panel.available_cameras}")
                else:
                    print("No cameras detected, using default camera 0")
            else:
                print("Control panel not ready, using default camera 0")
        except Exception as e:
            print(f"Error getting camera from control panel: {e}, using default camera 0")
        
        # Set dual resolution: high resolution for UI display, low resolution for model inference
        self.video_thread = VideoThread(
            camera_id=initial_camera_id,
            rotate=True,
            display_width=640,
            display_height=360,
            inference_width=112,
            inference_height=63)
        
        # Set main window reference for storing inference frames
        self.video_thread.main_window = self
        
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        
        # Initialize FPS values and inference frames
        self.current_fps = 0
        self.current_inference_frame = None
    
    def set_inference_frame(self, inference_frame):
        """Set inference frame and trigger async processing"""
        self.current_inference_frame = inference_frame
        # Send to async detection processor
        if hasattr(self.video_processor, 'process_inference_frame'):
            self.video_processor.process_inference_frame(inference_frame)
    
    def start_video(self):
        """Start video processing"""
        self.video_thread.start()
    
    def update_image(self, frame, fps=0):
        """Update image display and process detection"""
        self.video_processor.update_image(frame, fps)
    
    def change_detection(self, detection_type):
        """Change detection type"""
        self.detection_manager.change_detection(detection_type)
    
    def change_camera(self, index):
        """Switch camera"""
        self.video_processor.change_camera(index)
    
    def toggle_rotation(self, rotate):
        """Toggle video rotation mode"""
        self.video_processor.toggle_rotation(rotate)
    
    def toggle_skeleton(self, show):
        """Toggle skeleton display"""
        self.video_processor.toggle_skeleton(show)
    
    def toggle_boxes(self, show):
        """Toggle bounding boxes display"""
        if hasattr(self.video_processor, 'toggle_boxes'):
            self.video_processor.toggle_boxes(show)
        
    def toggle_mirror(self, mirror):
        """Toggle mirror mode"""
        self.video_processor.toggle_mirror(mirror)
    
    def open_video_file(self):
        """Open video file"""
        self.video_processor.open_video_file()
    
    def switch_to_camera_mode(self):
        """Switch back to camera mode"""
        self.video_processor.switch_to_camera_mode()
    
    def change_model(self, model_mode):
        """Switch detection model mode"""
        # Update model mode
        self.model_mode = model_mode
        
        # Update async detector model
        if hasattr(self.video_processor, 'update_model_mode'):
            self.video_processor.update_model_mode(model_mode)
        
        # Update sync processor model
        self.video_processor.change_model(model_mode)
    
    def switch_to_detection_mode(self):
        """Switch to detection mode"""
        self.mode_manager.switch_to_detection_mode()
    
    def switch_to_stats_mode(self):
        """Switch to statistics management mode"""
        self.mode_manager.switch_to_stats_mode()
    
    def show_about(self):
        """Show about dialog"""
        self.menu_manager.show_about()
    
    def change_language(self, language):
        """Change interface language"""
        self.menu_manager.change_language(language)
    
    def update_stats(self):
        """Update detection statistics"""
        self.stats_manager.update_stats()
    
    def closeEvent(self, event):
        """Clean up resources when closing window"""
        # Restore original stdout/stderr
        if hasattr(self, 'stdout_redirector'):
            sys.stdout = self.stdout_redirector.original_stream
        if hasattr(self, 'stderr_redirector'):
            sys.stderr = self.stderr_redirector.original_stream
        
        # Stop video thread
        if self.video_thread.isRunning():
            self.video_thread.stop()
        
        # Clean up async detection processor
        if hasattr(self.video_processor, 'cleanup'):
            self.video_processor.cleanup()
        
        event.accept()

