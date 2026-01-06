"""
Simple text log widget that appends timestamped messages.
"""

from typing import List
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QListWidget
from PyQt5.QtCore import Qt


class TextLogWidget(QWidget):
    def __init__(self, parent=None, max_items: int = 30):
        super().__init__(parent)
        self.max_items = max_items
        self.list = QListWidget(self)
        self.list.setAlternatingRowColors(True)
        self.list.setStyleSheet(
            """
            QListWidget { background: #f8f9fa; border: 2px solid #dee2e6; border-radius: 8px; color: #000000; }
            QListWidget::item { color: #000000; }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.list)

    def append_message(self, message: str) -> None:
        self.list.addItem(message)
        # Trim to max_items
        while self.list.count() > self.max_items:
            self.list.takeItem(0)
        # Scroll to bottom
        self.list.scrollToBottom()

    def set_window_size(self, max_items: int) -> None:
        self.max_items = int(max(1, max_items))
        # Trim immediately if needed
        while self.list.count() > self.max_items:
            self.list.takeItem(0)

    def clear(self) -> None:
        self.list.clear()



