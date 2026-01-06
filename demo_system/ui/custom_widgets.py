"""
Custom widget components - adapted from Good-GYM
"""

from PyQt5.QtWidgets import QPushButton, QWidget, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen, QFont

class ToggleSwitch(QPushButton):
    """Custom sliding switch button component"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(True)  # Default on state
        self.setMinimumWidth(90)
        self.setMaximumWidth(90)
        self.setMinimumHeight(34)
        self.setMaximumHeight(34)
        
        # Set colors
        # Use coherent palette with the dark theme
        self.on_color = QColor("#2ecc71")  # Green accent
        self.off_color = QColor("#4a4a4a")  # Dark gray
        self.thumb_color = QColor("#ffffff")  # White
        self.text_color = QColor("#eaf2ff")  # Bright label for readability
        
        # Connect state change signal to update text method
        self.toggled.connect(self.update_text)
        self.update_text(self.isChecked())
    
    def update_text(self, checked):
        """Update button text based on state"""
        self.setText("ON" if checked else "OFF")
        self.update()  # Force redraw
    
    def paintEvent(self, event):
        """Custom paint event"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set font
        font = QFont("Arial", 11, QFont.Bold)
        painter.setFont(font)
        
        # Determine background color
        bg_color = self.on_color if self.isChecked() else self.off_color
        
        # Draw background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 15, 15)
        
        # Draw thumb
        thumb_radius = self.height() - 6
        thumb_x = self.width() - thumb_radius - 3 if self.isChecked() else 3
        painter.setBrush(QBrush(self.thumb_color))
        painter.drawEllipse(thumb_x, 3, thumb_radius, thumb_radius)
        
        # Draw text
        painter.setPen(QPen(self.text_color))
        text_x = 5 if self.isChecked() else self.width() - 30
        painter.drawText(text_x, 0, 30, self.height(), Qt.AlignCenter, self.text())
        
class SwitchControl(QWidget):
    """Sliding switch control with label"""
    
    switched = pyqtSignal(bool)
    
    def __init__(self, label_text, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Create sliding switch
        self.toggle = ToggleSwitch()
        self.toggle.toggled.connect(self.switched.emit)
        
        # Set label
        self.label = QLabel(label_text)
        # Match brighter R2 font color and increase size
        self.label.setStyleSheet("color: #eaf2ff; font-size: 16pt; font-weight: 600;")
        
        # Add to layout
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.toggle, 0, Qt.AlignRight)
    
    def isChecked(self):
        """Return current switch state"""
        return self.toggle.isChecked()
    
    def setChecked(self, checked):
        """Set switch state"""
        self.toggle.setChecked(checked)
        
    def setText(self, text):
        """Set label text"""
        self.label.setText(text)

