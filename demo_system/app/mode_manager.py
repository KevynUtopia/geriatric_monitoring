"""
Mode manager - handles switching between different application modes
"""

from PyQt5.QtCore import QObject, pyqtSignal

class ModeManager(QObject):
    """Manages different application modes"""
    
    # Signals
    mode_changed = pyqtSignal(str)
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.current_mode = "detection"
        self.available_modes = ["detection", "stats", "settings"]
    
    def switch_to_detection_mode(self):
        """Switch to detection mode"""
        if self.current_mode != "detection":
            self.current_mode = "detection"
            self._update_ui_for_mode()
            self.mode_changed.emit("detection")
            print("Switched to detection mode")
    
    def switch_to_stats_mode(self):
        """Switch to statistics mode"""
        if self.current_mode != "stats":
            self.current_mode = "stats"
            self._update_ui_for_mode()
            self.mode_changed.emit("stats")
            print("Switched to statistics mode")
    
    def switch_to_settings_mode(self):
        """Switch to settings mode"""
        if self.current_mode != "settings":
            self.current_mode = "settings"
            self._update_ui_for_mode()
            self.mode_changed.emit("settings")
            print("Switched to settings mode")
    
    def _update_ui_for_mode(self):
        """Update UI based on current mode"""
        try:
            if self.current_mode == "detection":
                # Show detection-related UI elements
                if hasattr(self.main_window, 'stats_panel'):
                    self.main_window.stats_panel.setVisible(False)
                # Show video display and control panel
                if hasattr(self.main_window, 'video_display'):
                    self.main_window.video_display.setVisible(True)
                if hasattr(self.main_window, 'control_panel'):
                    self.main_window.control_panel.setVisible(True)
                    
            elif self.current_mode == "stats":
                # Show statistics panel
                if hasattr(self.main_window, 'stats_panel'):
                    self.main_window.stats_panel.setVisible(True)
                # Hide or minimize other panels
                if hasattr(self.main_window, 'video_display'):
                    self.main_window.video_display.setVisible(False)
                    
            elif self.current_mode == "settings":
                # Show settings panel (if implemented)
                # For now, just show a message
                print("Settings mode - implement settings panel here")
                
        except Exception as e:
            print(f"Error updating UI for mode {self.current_mode}: {e}")
    
    def get_current_mode(self):
        """Get current mode"""
        return self.current_mode
    
    def get_available_modes(self):
        """Get list of available modes"""
        return self.available_modes.copy()

