"""
Console widget for displaying real-time output from print statements
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QFont, QTextCursor
import sys
from datetime import datetime


class ConsoleWidget(QWidget):
    """Widget to display console output"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the console UI"""
        layout = QVBoxLayout(self)
        # Remove outer margins so the console container width equals the video container
        layout.setContentsMargins(0, 0, 0, 0)
        # Match video container frame (background, border, radius)
        self.setStyleSheet(
            """
            QWidget {
                background-color: rgba(0,0,0,0.6);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 12px;
            }
            """
        )
        
        # Create text display
        self.text_display = QTextEdit(self)
        self.text_display.setReadOnly(True)
        self.text_display.setMinimumHeight(200)  # Increased minimum height
        self.text_display.setMaximumHeight(300)  # Increased maximum height
        
        # Set font with emoji support - try multiple fonts for cross-platform compatibility
        font = QFont()
        font.setPointSize(10)  # Slightly larger for better readability
        # Try to use fonts with good emoji support
        font.setFamilies(["Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", "DejaVu Sans", "Arial", "Liberation Sans"])
        self.text_display.setFont(font)
        
        # Set style
        # Inner text area should be borderless and transparent so outer frame defines the width
        self.text_display.setStyleSheet("""
            QTextEdit {
                background-color: rgba(0,0,0,0);
                color: #d4d4d4;
                border: none;
                padding: 0px;
                font-size: 10pt;
            }
        """)
        
        layout.addWidget(self.text_display)
    
    def append_output(self, text, stream_type='stdout'):
        """Append text to console output"""
        if not text.strip():
            return
            
        # Color code based on stream type
        if stream_type == 'stderr':
            color = '#f48771'  # Red for errors
        else:
            color = '#d4d4d4'  # Normal color
        
        # Get current timestamp
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        # Escape HTML special characters but preserve Unicode emojis
        import html
        escaped_text = html.escape(text)
        
        # Enhanced warning detection and formatting
        if '⚠️' in text or 'warning' in text.lower() or 'error' in text.lower():
            # Make warnings more visually significant
            if '⚠️' in text:
                # Replace plain warning emoji with enhanced version
                escaped_text = escaped_text.replace('⚠️', '<span style="color: #ffaa00; font-size: 14pt; font-weight: bold;">⚠️</span>')
            elif 'warning' in text.lower():
                escaped_text = escaped_text.replace('warning', '<span style="color: #ffaa00; font-weight: bold; text-transform: uppercase;">WARNING</span>')
            elif 'error' in text.lower():
                escaped_text = escaped_text.replace('error', '<span style="color: #ff4444; font-weight: bold; text-transform: uppercase;">ERROR</span>')
            
            # Add background highlight for warnings/errors
            formatted_text = f'<span style="color: #808080; font-family: monospace;">[{timestamp}]</span> <span style="color: {color}; background-color: rgba(255, 170, 0, 0.1); padding: 2px 4px; border-radius: 3px;">{escaped_text}</span>'
        else:
            # Normal formatting
            formatted_text = f'<span style="color: #808080; font-family: monospace;">[{timestamp}]</span> <span style="color: {color};">{escaped_text}</span>'
        
        # Append to display with HTML formatting
        cursor = self.text_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.text_display.setTextCursor(cursor)
        self.text_display.insertHtml(formatted_text + '<br>')
        
        # Auto-scroll to bottom
        cursor = self.text_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.text_display.setTextCursor(cursor)
    
    def clear(self):
        """Clear console output"""
        self.text_display.clear()


class OutputRedirector(QObject):
    """Redirects stdout/stderr to Qt signal"""
    
    output_signal = pyqtSignal(str, str)  # text, stream_type
    
    def __init__(self, stream_type='stdout'):
        super().__init__()
        self.stream_type = stream_type
        self.original_stream = sys.stdout if stream_type == 'stdout' else sys.stderr
    
    def write(self, text):
        """Write method called by print()"""
        # Also write to original stream
        self.original_stream.write(text)
        self.original_stream.flush()
        
        # Emit signal for GUI
        if text.strip():
            self.output_signal.emit(text.rstrip(), self.stream_type)
    
    def flush(self):
        """Flush method required for file-like objects"""
        self.original_stream.flush()
    
    def isatty(self):
        """Check if stream is a TTY"""
        return False

