"""
Statistics panel - displays detection statistics and data visualization
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTabWidget, 
                             QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
                             QProgressBar, QPushButton, QFileDialog, QTextEdit)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from .styles import AppStyles
from core.translations import Translations as T

class StatsPanel(QWidget):
    """Statistics panel for detection data"""
    
    # Signals
    data_updated = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Setup statistics panel UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel(T.get("detection_statistics"))
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 15px;")
        layout.addWidget(title_label)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Overview tab
        self.overview_tab = self.create_overview_tab()
        self.tab_widget.addTab(self.overview_tab, T.get("overview"))
        
        # Daily stats tab
        self.daily_tab = self.create_daily_tab()
        self.tab_widget.addTab(self.daily_tab, T.get("daily_stats"))
        
        # Detection types tab
        self.types_tab = self.create_types_tab()
        self.tab_widget.addTab(self.types_tab, T.get("detection_types"))
        
        # Export tab
        self.export_tab = self.create_export_tab()
        self.tab_widget.addTab(self.export_tab, T.get("export_data"))
        
        layout.addWidget(self.tab_widget)
    
    def create_overview_tab(self):
        """Create overview statistics tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Overall statistics group
        stats_group = QGroupBox(T.get("overall_statistics"))
        stats_group.setStyleSheet(AppStyles.get_group_box_style())
        stats_layout = QVBoxLayout(stats_group)
        
        # Total detections
        total_layout = QHBoxLayout()
        total_label = QLabel(T.get("total_detections"))
        total_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        self.total_detections_label = QLabel("0")
        self.total_detections_label.setStyleSheet("font-size: 24pt; color: #3498db; font-weight: bold;")
        total_layout.addWidget(total_label)
        total_layout.addWidget(self.total_detections_label)
        total_layout.addStretch()
        stats_layout.addLayout(total_layout)
        
        # Simple text display for now
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        self.stats_text.setPlainText("Detection statistics will be displayed here...")
        stats_layout.addWidget(self.stats_text)
        
        layout.addWidget(stats_group)
        layout.addStretch()
        
        return widget
    
    def create_daily_tab(self):
        """Create daily statistics tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Daily statistics group
        daily_group = QGroupBox(T.get("daily_statistics"))
        daily_group.setStyleSheet(AppStyles.get_group_box_style())
        daily_layout = QVBoxLayout(daily_group)
        
        # Simple text display for daily stats
        self.daily_text = QTextEdit()
        self.daily_text.setReadOnly(True)
        self.daily_text.setPlainText("Daily statistics will be displayed here...")
        daily_layout.addWidget(self.daily_text)
        
        layout.addWidget(daily_group)
        layout.addStretch()
        
        return widget
    
    def create_types_tab(self):
        """Create detection types tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Detection types group
        types_group = QGroupBox(T.get("detection_types_breakdown"))
        types_group.setStyleSheet(AppStyles.get_group_box_style())
        types_layout = QVBoxLayout(types_group)
        
        # Simple text display for types
        self.types_text = QTextEdit()
        self.types_text.setReadOnly(True)
        self.types_text.setPlainText("Detection types breakdown will be displayed here...")
        types_layout.addWidget(self.types_text)
        
        layout.addWidget(types_group)
        layout.addStretch()
        
        return widget
    
    def create_export_tab(self):
        """Create export data tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Export group
        export_group = QGroupBox(T.get("export_data"))
        export_group.setStyleSheet(AppStyles.get_group_box_style())
        export_layout = QVBoxLayout(export_group)
        
        # Export description
        desc_label = QLabel(T.get("export_description"))
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #7f8c8d; margin-bottom: 15px;")
        export_layout.addWidget(desc_label)
        
        # Export buttons
        buttons_layout = QHBoxLayout()
        
        # Export JSON button
        self.export_json_button = QPushButton(T.get("export_json"))
        self.export_json_button.setStyleSheet(AppStyles.get_save_button_style())
        self.export_json_button.clicked.connect(self.export_json)
        buttons_layout.addWidget(self.export_json_button)
        
        # Export CSV button
        self.export_csv_button = QPushButton(T.get("export_csv"))
        self.export_csv_button.setStyleSheet(AppStyles.get_save_button_style())
        self.export_csv_button.clicked.connect(self.export_csv)
        buttons_layout.addWidget(self.export_csv_button)
        
        # Clear data button
        self.clear_data_button = QPushButton(T.get("clear_data"))
        self.clear_data_button.setStyleSheet(AppStyles.get_reset_button_style())
        self.clear_data_button.clicked.connect(self.clear_data)
        buttons_layout.addWidget(self.clear_data_button)
        
        export_layout.addLayout(buttons_layout)
        
        layout.addWidget(export_group)
        layout.addStretch()
        
        return widget
    
    def update_display(self, stats, daily_stats):
        """Update statistics display"""
        try:
            # Update overview tab
            self.update_overview_tab(stats)
            
            # Update daily tab
            self.update_daily_tab(daily_stats)
            
            # Update types tab
            self.update_types_tab(stats)
            
        except Exception as e:
            print(f"Error updating stats display: {e}")
    
    def update_overview_tab(self, stats):
        """Update overview tab"""
        try:
            # Update total detections
            total = stats.get('total_detections', 0)
            self.total_detections_label.setText(str(total))
            
            # Update stats text
            detection_types = stats.get('detection_types', {})
            stats_text = f"Total Detections: {total}\n\nDetection Types:\n"
            for det_type, count in detection_types.items():
                stats_text += f"- {det_type}: {count}\n"
            
            self.stats_text.setPlainText(stats_text)
                
        except Exception as e:
            print(f"Error updating overview tab: {e}")
    
    def update_daily_tab(self, daily_stats):
        """Update daily tab"""
        try:
            daily_text = "Daily Statistics:\n\n"
            for date, day_stats in daily_stats.items():
                total = day_stats.get('total_detections', 0)
                daily_text += f"{date}: {total} detections\n"
                
                types = day_stats.get('detection_types', {})
                for det_type, count in types.items():
                    daily_text += f"  - {det_type}: {count}\n"
                daily_text += "\n"
            
            self.daily_text.setPlainText(daily_text)
                
        except Exception as e:
            print(f"Error updating daily tab: {e}")
    
    def update_types_tab(self, stats):
        """Update types tab"""
        try:
            detection_types = stats.get('detection_types', {})
            types_text = "Detection Types Breakdown:\n\n"
            
            for det_type, count in detection_types.items():
                daily_avg = count / 7 if count > 0 else 0
                types_text += f"{det_type}:\n"
                types_text += f"  Total: {count}\n"
                types_text += f"  Daily Average: {daily_avg:.1f}\n"
                types_text += f"  Last Detection: Recent\n\n"
            
            self.types_text.setPlainText(types_text)
                
        except Exception as e:
            print(f"Error updating types tab: {e}")
    
    def export_json(self):
        """Export data as JSON"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, T.get("save_file"), "detection_data.json", "JSON Files (*.json)"
            )
            
            if file_path:
                # TODO: Implement JSON export
                print(f"Exporting to JSON: {file_path}")
                
        except Exception as e:
            print(f"Error exporting JSON: {e}")
    
    def export_csv(self):
        """Export data as CSV"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, T.get("save_file"), "detection_data.csv", "CSV Files (*.csv)"
            )
            
            if file_path:
                # TODO: Implement CSV export
                print(f"Exporting to CSV: {file_path}")
                
        except Exception as e:
            print(f"Error exporting CSV: {e}")
    
    def clear_data(self):
        """Clear all data"""
        try:
            from PyQt5.QtWidgets import QMessageBox
            
            reply = QMessageBox.question(
                self, T.get("confirm_clear"), T.get("clear_data_warning"),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # TODO: Implement data clearing
                print("Data cleared")
                
        except Exception as e:
            print(f"Error clearing data: {e}")
