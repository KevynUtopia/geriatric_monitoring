"""
Statistics manager - handles detection statistics and data visualization
"""

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTabWidget
from ui.stats_panel import StatsPanel

class StatsManager(QObject):
    """Manages detection statistics and data visualization"""
    
    # Signals
    stats_updated = pyqtSignal(dict)
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.stats_panel = None
    
    def init_stats_panel(self):
        """Initialize statistics panel"""
        try:
            self.stats_panel = StatsPanel(self.main_window)
            self.stats_panel.data_updated.connect(self.update_stats)
            print("Statistics panel initialized")
        except Exception as e:
            print(f"Error initializing stats panel: {e}")
    
    def update_stats(self):
        """Update detection statistics"""
        try:
            if hasattr(self.main_window, 'data_manager'):
                stats = self.main_window.data_manager.get_overall_stats()
                daily_stats = self.main_window.data_manager.get_daily_stats(7)
                
                # Update stats panel if it exists
                if self.stats_panel:
                    self.stats_panel.update_display(stats, daily_stats)
                
                # Emit signal
                self.stats_updated.emit(stats)
                
        except Exception as e:
            print(f"Error updating stats: {e}")
    
    def get_detection_summary(self):
        """Get summary of detection statistics"""
        try:
            if hasattr(self.main_window, 'data_manager'):
                return self.main_window.data_manager.get_overall_stats()
            return {}
        except Exception as e:
            print(f"Error getting detection summary: {e}")
            return {}
    
    def export_stats(self, file_path):
        """Export statistics to file"""
        try:
            if hasattr(self.main_window, 'data_manager'):
                return self.main_window.data_manager.export_results(file_path)
            return False
        except Exception as e:
            print(f"Error exporting stats: {e}")
            return False

