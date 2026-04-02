from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QColor, QPainterPath, QPen
from PySide6.QtWidgets import QWidget


class MicIndicatorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self._level = 0.0
        self._sensitivity = 35.0

    def set_level(self, level: float):
        target = min(1.0, level * self._sensitivity)

        if target > self._level:
            self._level = target
        else:
            self._level = self._level * 0.8 + target * 0.2

        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        cap_w = 24
        cap_h = 42
        cap_x = (w - cap_w) / 2
        cap_y = (h - cap_h) / 2 - 12
        cap_rect = QRectF(cap_x, cap_y, cap_w, cap_h)

        p.setPen(QPen(
            QColor("#7f8c8d"), 2.5,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin
        ))

        arc_rect = QRectF(cap_x - 8, cap_y + 12, cap_w + 16, cap_h)
        p.drawArc(arc_rect, 180 * 16, 180 * 16)

        stand_x = w / 2
        stand_top = cap_y + cap_h + 10
        stand_bottom = stand_top + 12
        p.drawLine(stand_x, stand_top, stand_x, stand_bottom)
        p.drawLine(stand_x - 12, stand_bottom, stand_x + 12, stand_bottom)

        cap_path = QPainterPath()
        cap_path.addRoundedRect(cap_rect, cap_w / 2, cap_w / 2)
        p.setPen(QPen(QColor("#7f8c8d"), 2))
        p.setBrush(QColor("#1a1a2e"))
        p.drawPath(cap_path)

        if self._level > 0.01:
            p.save()
            p.setClipPath(cap_path)

            fill_h = cap_h * self._level
            fill_rect = QRectF(cap_x, cap_y + cap_h - fill_h, cap_w, fill_h)

            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor("#2ecc71"))
            p.drawRect(fill_rect)
            p.restore()
