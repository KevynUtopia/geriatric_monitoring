"""
Detection manager - handles detection type switching and coordination
"""

from PyQt5.QtCore import QObject, pyqtSignal

class DetectionManager(QObject):
    """Manages detection types and coordination"""
    
    # Signals
    detection_type_changed = pyqtSignal(str)
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.current_detection_type = "pose_detection"
        self.available_detection_types = [
            "pose_detection",
            "object_detection", 
            "action_recognition",
            "custom_detection"
        ]
    
    def change_detection(self, detection_type):
        """Change detection type"""
        try:
            if detection_type in self.available_detection_types:
                self.current_detection_type = detection_type
                self._update_detection_settings()
                self.detection_type_changed.emit(detection_type)
                print(f"Detection type changed to: {detection_type}")
            else:
                print(f"Invalid detection type: {detection_type}")
        except Exception as e:
            print(f"Error changing detection type: {e}")
    
    def _update_detection_settings(self):
        """Update detection settings based on current type"""
        try:
            # Update video processor detection types
            if hasattr(self.main_window, 'video_processor'):
                if self.current_detection_type == "pose_detection":
                    self.main_window.video_processor.current_detection_types = ['pose']
                elif self.current_detection_type == "object_detection":
                    self.main_window.video_processor.current_detection_types = ['object']
                elif self.current_detection_type == "action_recognition":
                    self.main_window.video_processor.current_detection_types = ['action']
                elif self.current_detection_type == "custom_detection":
                    self.main_window.video_processor.current_detection_types = ['custom']
                else:
                    # Multiple detection types
                    self.main_window.video_processor.current_detection_types = ['pose', 'object', 'action', 'custom']
            
            # Update control panel
            if hasattr(self.main_window, 'control_panel'):
                self.main_window.control_panel.current_detection = self.current_detection_type
                self.main_window.control_panel.update_detection_style()
                
        except Exception as e:
            print(f"Error updating detection settings: {e}")
    
    def get_current_detection_type(self):
        """Get current detection type"""
        return self.current_detection_type
    
    def get_available_detection_types(self):
        """Get list of available detection types"""
        return self.available_detection_types.copy()
    
    def add_custom_detection_type(self, detection_type):
        """Add a custom detection type"""
        try:
            if detection_type not in self.available_detection_types:
                self.available_detection_types.append(detection_type)
                print(f"Added custom detection type: {detection_type}")
            else:
                print(f"Detection type already exists: {detection_type}")
        except Exception as e:
            print(f"Error adding custom detection type: {e}")
    
    def remove_detection_type(self, detection_type):
        """Remove a detection type"""
        try:
            if detection_type in self.available_detection_types and detection_type != "pose_detection":
                self.available_detection_types.remove(detection_type)
                if self.current_detection_type == detection_type:
                    self.change_detection("pose_detection")
                print(f"Removed detection type: {detection_type}")
            else:
                print(f"Cannot remove detection type: {detection_type}")
        except Exception as e:
            print(f"Error removing detection type: {e}")

