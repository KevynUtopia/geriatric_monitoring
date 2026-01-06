"""
Temporal curve widget that maintains a rolling buffer of values in [0, 1]
and draws them as a live-updating line chart. The newest value is plotted
at the right edge; older values shift left until they leave the view.
"""

from collections import deque
from typing import Deque, List
import numpy as np

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRect


class TemporalCurveWidget(QWidget):
    def __init__(self, parent=None, max_points: int = 30):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setAutoFillBackground(True)
        self.values: Deque[float] = deque(maxlen=max_points)
        self.border_color = QColor(222, 226, 230)
        self.line_color = QColor(52, 152, 219)
        self.grid_color = QColor(230, 230, 230)

    def update_with_value(self, value: float) -> None:
        # Clamp to [0, 1]
        v = 0.0 if value is None else max(0.0, min(1.0, float(value)))
        self.values.append(v)
        self.update()

    def set_window_size(self, max_points: int) -> None:
        max_points = int(max(1, max_points))
        old_vals = list(self.values)[-max_points:]
        self.values = deque(old_vals, maxlen=max_points)
        self.update()

    def clear(self) -> None:
        self.values.clear()
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        rect: QRect = self.rect()
        w = rect.width()
        h = rect.height()

        # Background
        painter.fillRect(rect, QColor(248, 249, 250))

        # Border
        pen = QPen(self.border_color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(rect.adjusted(1, 1, -1, -1))

        if w <= 4 or h <= 4 or len(self.values) == 0:
            return

        # Draw horizontal grid lines at 0.25, 0.5, 0.75
        grid_pen = QPen(self.grid_color)
        grid_pen.setStyle(Qt.DashLine)
        painter.setPen(grid_pen)
        for frac in (0.25, 0.5, 0.75):
            y = int((1.0 - frac) * (h - 4)) + 2
            painter.drawLine(2, y, w - 2, y)

        # Prepare points mapping values -> widget coordinates
        painter.setPen(QPen(self.line_color, 2))
        vals: List[float] = list(self.values)
        n = len(vals)
        # Map entire history across the available width, newest at right
        # x positions are linearly spaced from 2 .. w-2
        if n == 1:
            x = w - 2
            y = int((1.0 - vals[0]) * (h - 4)) + 2
            painter.drawPoint(x, y)
            return

        x_positions = np.linspace(2, w - 2, num=n)
        prev_x = int(x_positions[0])
        prev_y = int((1.0 - vals[0]) * (h - 4)) + 2
        for i in range(1, n):
            x = int(x_positions[i])
            y = int((1.0 - vals[i]) * (h - 4)) + 2
            painter.drawLine(prev_x, prev_y, x, y)
            prev_x, prev_y = x, y



