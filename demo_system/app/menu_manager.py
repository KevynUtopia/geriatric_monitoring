"""
Menu manager - handles menu bar and dialog management
"""

from PyQt5.QtWidgets import (QMenuBar, QMenu, QAction, QActionGroup, QMessageBox, 
                             QFileDialog, QDialog, QVBoxLayout, QLabel, QTextEdit, QPushButton)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from core.translations import Translations as T

class MenuManager:
    """Manages menu bar and dialogs"""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.language_group = None
    
    def setup_menu_bar(self):
        """Setup menu bar"""
        try:
            menubar = self.main_window.menuBar()
            
            # File menu
            file_menu = menubar.addMenu(T.get("file_menu"))
            
            # Open video file action
            open_video_action = QAction(T.get("open_video"), self.main_window)
            open_video_action.triggered.connect(self.main_window.open_video_file)
            file_menu.addAction(open_video_action)
            
            # Switch to camera action
            camera_action = QAction(T.get("camera_mode"), self.main_window)
            camera_action.triggered.connect(self.main_window.switch_to_camera_mode)
            file_menu.addAction(camera_action)
            
            file_menu.addSeparator()
            
            # Exit action
            exit_action = QAction(T.get("exit"), self.main_window)
            exit_action.triggered.connect(self.main_window.close)
            file_menu.addAction(exit_action)
            
            # Tools menu
            tools_menu = menubar.addMenu(T.get("tools_menu"))
            
            # Skeleton display toggle
            skeleton_action = QAction(T.get("skeleton_display"), self.main_window)
            skeleton_action.setCheckable(True)
            skeleton_action.setChecked(True)
            skeleton_action.triggered.connect(lambda checked: self.main_window.toggle_skeleton(checked))
            tools_menu.addAction(skeleton_action)
            
            # Mode menu
            mode_menu = menubar.addMenu(T.get("mode_menu"))
            
            # Detection mode
            detection_mode_action = QAction(T.get("detection_mode"), self.main_window)
            detection_mode_action.triggered.connect(self.main_window.switch_to_detection_mode)
            mode_menu.addAction(detection_mode_action)
            
            # Statistics mode
            stats_mode_action = QAction(T.get("stats_mode"), self.main_window)
            stats_mode_action.triggered.connect(self.main_window.switch_to_stats_mode)
            mode_menu.addAction(stats_mode_action)
            
            # Language menu
            language_menu = menubar.addMenu(T.get("language_menu"))
            self.setup_language_menu(language_menu)
            
            # Help menu
            help_menu = menubar.addMenu(T.get("help_menu"))
            
            # About action
            about_action = QAction(T.get("about"), self.main_window)
            about_action.triggered.connect(self.show_about)
            help_menu.addAction(about_action)
            
            print("Menu bar setup completed")
            
        except Exception as e:
            print(f"Error setting up menu bar: {e}")
    
    def setup_language_menu(self, language_menu):
        """Setup language selection menu"""
        try:
            self.language_group = QActionGroup(self.main_window)
            
            # English
            english_action = QAction(T.get("english"), self.main_window)
            english_action.setCheckable(True)
            english_action.setChecked(T.current_language == "en")
            english_action.triggered.connect(lambda: self.change_language("en"))
            self.language_group.addAction(english_action)
            language_menu.addAction(english_action)
            
            # Chinese
            chinese_action = QAction(T.get("chinese"), self.main_window)
            chinese_action.setCheckable(True)
            chinese_action.setChecked(T.current_language == "zh")
            chinese_action.triggered.connect(lambda: self.change_language("zh"))
            self.language_group.addAction(chinese_action)
            language_menu.addAction(chinese_action)
            
            # Spanish
            spanish_action = QAction(T.get("spanish"), self.main_window)
            spanish_action.setCheckable(True)
            spanish_action.setChecked(T.current_language == "es")
            spanish_action.triggered.connect(lambda: self.change_language("es"))
            self.language_group.addAction(spanish_action)
            language_menu.addAction(spanish_action)
            
            # Hindi
            hindi_action = QAction(T.get("hindi"), self.main_window)
            hindi_action.setCheckable(True)
            hindi_action.setChecked(T.current_language == "hi")
            hindi_action.triggered.connect(lambda: self.change_language("hi"))
            self.language_group.addAction(hindi_action)
            language_menu.addAction(hindi_action)
            
        except Exception as e:
            print(f"Error setting up language menu: {e}")
    
    def change_language(self, language):
        """Change application language"""
        try:
            if T.set_language(language):
                # Update UI language
                self.main_window.change_language(language)
                print(f"Language changed to {language}")
            else:
                print(f"Invalid language: {language}")
                
        except Exception as e:
            print(f"Error changing language: {e}")
    
    def show_about(self):
        """Show about dialog"""
        try:
            dialog = QDialog(self.main_window)
            dialog.setWindowTitle(T.get("about_title"))
            dialog.setModal(True)
            dialog.resize(500, 400)
            
            layout = QVBoxLayout(dialog)
            
            # Title
            title_label = QLabel("AI Detection Template")
            title_label.setFont(QFont("Arial", 16, QFont.Bold))
            title_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(title_label)
            
            # Content
            content_text = QTextEdit()
            content_text.setReadOnly(True)
            content_text.setHtml(T.get("about_content"))
            layout.addWidget(content_text)
            
            # Close button
            close_button = QPushButton("Close")
            close_button.clicked.connect(dialog.accept)
            layout.addWidget(close_button)
            
            dialog.exec_()
            
        except Exception as e:
            print(f"Error showing about dialog: {e}")
    
    def show_message(self, title, message, message_type=QMessageBox.Information):
        """Show message dialog"""
        try:
            msg_box = QMessageBox(self.main_window)
            msg_box.setWindowTitle(title)
            msg_box.setText(message)
            msg_box.setIcon(message_type)
            msg_box.exec_()
        except Exception as e:
            print(f"Error showing message: {e}")
    
    def show_file_dialog(self, title, file_filter, save_mode=False):
        """Show file dialog"""
        try:
            if save_mode:
                file_path, _ = QFileDialog.getSaveFileName(
                    self.main_window, title, "", file_filter
                )
            else:
                file_path, _ = QFileDialog.getOpenFileName(
                    self.main_window, title, "", file_filter
                )
            return file_path
        except Exception as e:
            print(f"Error showing file dialog: {e}")
            return ""

