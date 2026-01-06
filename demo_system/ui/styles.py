"""
Application style definitions - adapted for detection models
"""

from PyQt5.QtGui import QColor, QPalette, QFont
from PyQt5.QtCore import Qt

class AppStyles:
    """Application style definitions for detection models"""
    
    # Detection type color mapping
    DETECTION_COLORS = {
        # Detection types
        "Pose Detection": "#3498db",      # Blue
        "Object Detection": "#e74c3c",    # Red
        "Action Recognition": "#2ecc71",  # Green
        "Custom Detection": "#f39c12",    # Orange
        
        # Chinese names
        "姿态检测": "#3498db",      # Blue
        "目标检测": "#e74c3c",    # Red
        "动作识别": "#2ecc71",  # Green
        "自定义检测": "#f39c12", # Orange
        
        # English names
        "Pose Detection": "#3498db",      # Blue
        "Object Detection": "#e74c3c",    # Red
        "Action Recognition": "#2ecc71",  # Green
        "Custom Detection": "#f39c12"     # Orange
    }
    
    @staticmethod
    def get_window_palette():
        """Get window palette"""
        palette = QPalette()
        # Dark theme base
        palette.setColor(QPalette.Window, QColor(18, 18, 18))
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.Base, QColor(30, 30, 30))
        palette.setColor(QPalette.AlternateBase, QColor(40, 40, 40))
        palette.setColor(QPalette.ToolTipBase, QColor(30, 30, 30))
        palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
        palette.setColor(QPalette.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.Button, QColor(28, 28, 28))
        palette.setColor(QPalette.ButtonText, QColor(230, 230, 230))
        palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
        palette.setColor(QPalette.Highlight, QColor(41, 128, 185))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        return palette
    
    @staticmethod
    def get_global_stylesheet():
        """Get global stylesheet"""
        return """
            QWidget {
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 10pt;
                color: #dddddd;
                background-color: transparent;
            }
            QComboBox {
                border: 1px solid #3c3c3c;
                border-radius: 6px;
                padding: 6px 8px;
                min-width: 6em;
                background-color: rgba(40, 40, 40, 0.8);
                color: #e0e0e0;
            }
            QComboBox:hover {
                border: 1px solid #2980b9;
            }
            QLabel {
                color: #e0e0e0;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QGroupBox {
                padding-top: 15px;
            }
        """
    
    @staticmethod
    def get_detection_combo_style():
        """Get detection selection dropdown style"""
        return """
            QComboBox {
                font-size: 12pt;
                padding: 4px 10px;
                min-height: 28px;
                max-height: 32px;
                border: 1px solid #3c3c3c;
                border-radius: 8px;
                background-color: rgba(40, 40, 40, 0.8);
                color: #e0e0e0;
            }
            QComboBox:hover {
                border-color: #2980b9;
            }
            QComboBox::drop-down {
                border: 0px;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #3c3c3c;
                border-radius: 6px;
                selection-background-color: #2980b9;
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-size: 12pt;
            }
        """
    
    @staticmethod
    def get_counter_value_style(color="#27ae60"):
        """Get counter style"""
        return f"""
            color: {color};
            background-color: rgba(255, 255, 255, 0.04);
            border-radius: 8px;
            padding: 6px 10px;
            border: 1px solid #3c3c3c;
            font-size: 12pt;
            font-weight: bold;
        """
    
    @staticmethod
    def get_fps_value_style(color="#34495e"):
        """Get FPS value style"""
        return f"""
            color: {color};
            background-color: rgba(255, 255, 255, 0.04);
            border-radius: 10px;
            padding: 6px 10px;
            border: 1px solid #3c3c3c;
            min-width: 100px;
            text-align: center;
            font-size: 28pt;
        """
    
    @staticmethod
    def get_success_counter_style():
        """Get success counter style"""
        return """
            color: #2ecc71;
            background-color: rgba(46, 204, 113, 0.08);
            border-radius: 8px;
            padding: 6px 10px;
            border: 1px solid #2ecc71;
            font-size: 12pt;
            font-weight: bold;
        """
    
    @staticmethod
    def get_status_indicator_style(active=False, color="#3498db"):
        """Get status indicator style"""
        bg_color = color if active else "#2c2c2c"
        text_color = "white" if active else "#9e9e9e"
        
        return f"""
            font-size: 34pt;
            color: {text_color};
            background-color: {bg_color};
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            min-width: 56px;
            min-height: 56px;
            padding: 6px;
            text-align: center;
        """
    
    @staticmethod
    def get_group_box_style():
        """Get group box style"""
        return """
            QGroupBox {
                font-weight: bold;
                color: #e0e0e0;
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 12px;
                margin-top: 10px;
                background-color: rgba(40,40,40,0.75);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 2px 8px;
                background-color: rgba(22,22,22,0.7);
                font-size: 17pt;
                color: #eaf2ff;
            }
        """
    
    @staticmethod
    def get_camera_combo_style():
        """Get camera selection dropdown style"""
        return """
            QComboBox {
                font-size: 12pt;
                padding: 4px 10px;
                min-height: 28px;
                max-height: 32px;
                border: 1px solid #3c3c3c;
                border-radius: 8px;
                background-color: rgba(40, 40, 40, 0.8);
                color: #e0e0e0;
            }
            QComboBox:hover {
                border-color: #2980b9;
            }
            QComboBox::drop-down {
                border: 0px;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #3c3c3c;
                border-radius: 6px;
                selection-background-color: #2980b9;
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-size: 12pt;
            }
        """
        
    @staticmethod
    def get_reset_button_style():
        """Get reset button style"""
        return """
            QPushButton {
                background-color: #6c757d;
                color: white;
                border-radius: 8px;
                padding: 10px 14px;
                font-weight: bold;
                border: 1px solid rgba(255,255,255,0.08);
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:pressed {
                background-color: #545b62;
            }
        """
    
    @staticmethod
    def get_save_button_style():
        """Get save results button style - Blue"""
        return """
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 8px;
                padding: 10px 14px;
                font-weight: bold;
                border: 1px solid rgba(255,255,255,0.08);
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1c6ea4;
            }
        """
        
    @staticmethod
    def get_success_button_style():
        """Get success button style - Green"""
        return """
            QPushButton {
                background-color: #27ae60;
                color: white;
                border-radius: 8px;
                padding: 10px 14px;
                font-weight: bold;
                border: 1px solid rgba(255,255,255,0.08);
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #1f8a4c;
            }
        """
    
    @staticmethod
    def get_toggle_button_style(checked=False):
        """Get toggle button style"""
        # Set different styles for on and off states
        if checked:
            # On state - green background
            return """
                QPushButton {
                    background-color: #2ecc71;
                    border-radius: 15px;
                    color: white;
                    border: none;
                    min-width: 80px;
                    max-width: 80px;
                    min-height: 30px;
                    max-height: 30px;
                    font-weight: bold;
                    font-size: 12pt;
                    padding-right: 25px; /* Leave space for indicator */
                    text-align: center;
                }
                QPushButton:hover {
                    background-color: #27ae60;
                }
            """
        else:
            # Off state - gray background
            return """
                QPushButton {
                    background-color: #4a4a4a;
                    border-radius: 15px;
                    color: white;
                    border: none;
                    min-width: 80px;
                    max-width: 80px;
                    min-height: 30px;
                    max-height: 30px;
                    font-weight: bold;
                    font-size: 12pt;
                    padding-left: 25px; /* Leave space for indicator */
                    text-align: center;
                }
                QPushButton:hover {
                    background-color: #5a5a5a;
                }
            """

