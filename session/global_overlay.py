import time
from datetime import datetime
from pathlib import Path

import mss
import mss.tools
from pynput import keyboard

from PySide6.QtCore import Qt, QPoint, QRect, Signal, QObject, Slot
from PySide6.QtGui import QColor, QCursor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget


class KeyboardSignals(QObject):
    ctrl_pressed = Signal()
    ctrl_released = Signal()
    kill_app = Signal()


class GlobalOverlay(QWidget):
    screenshot_taken = Signal(str, float)

    _IDLE_DIM = QColor(0, 0, 0, 30)
    _ACTIVE_DIM = QColor(0, 0, 0, 100)

    def __init__(self, output_dir: Path, session_start: float, parent=None):
        super().__init__(parent)
        self._out_dir = output_dir
        self._session_start = session_start

        self._origin: QPoint | None = None
        self._current: QPoint | None = None
        self._dragging = False
        self._is_active = False

        self._ctrl_down = False
        self._alt_down = False

        self.setGeometry(QApplication.primaryScreen().virtualGeometry())

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

        self._signals = KeyboardSignals()
        self._signals.ctrl_pressed.connect(self._activate_capture)
        self._signals.ctrl_released.connect(self._deactivate_capture)

        self._listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release
        )
        self._listener.start()

    def _on_key_press(self, key):
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._ctrl_down = True
            if not self._is_active:
                self._signals.ctrl_pressed.emit()

        elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
            self._alt_down = True

        elif hasattr(key, 'char') and key.char in ('q', 'Q'):
            if self._ctrl_down and self._alt_down:
                self._signals.kill_app.emit()

    def _on_key_release(self, key):
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self._ctrl_down = False
            self._signals.ctrl_released.emit()
        elif key in (keyboard.Key.alt_l, keyboard.Key.alt_r):
            self._alt_down = False

    @Slot()
    def _activate_capture(self):
        self._is_active = True
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, False)
        self.show()
        self.update()

    @Slot()
    def _deactivate_capture(self):
        self._is_active = False
        self._dragging = False
        self._origin = None
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, True)
        self.show()
        self.update()

    def stop_listener(self):
        self._listener.stop()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._is_active:
            self._origin = event.position().toPoint()
            self._current = self._origin
            self._dragging = True
            self.update()

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self._current = event.position().toPoint()
            self._capture()
            self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        bg_color = self._ACTIVE_DIM if self._is_active else self._IDLE_DIM
        p.fillRect(self.rect(), bg_color)

        if self._is_active and self._origin and self._current:
            sel = QRect(self._origin, self._current).normalized()
            if sel.width() > 2 and sel.height() > 2:
                p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
                p.fillRect(sel, Qt.GlobalColor.black)
                p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
                p.setPen(QPen(QColor(52, 152, 219), 2))
                p.drawRect(sel)

    def _capture(self):
        sel = QRect(self._origin, self._current).normalized()
        if sel.width() < 10 or sel.height() < 10:
            return

        self.hide()
        QApplication.processEvents()

        elapsed = time.time() - self._session_start
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
        out_path = self._out_dir / f"screenshot_{ts_str}.png"

        dpr = QApplication.primaryScreen().devicePixelRatio()
        monitor = {
            "top": int(sel.y() * dpr),
            "left": int(sel.x() * dpr),
            "width": int(sel.width() * dpr),
            "height": int(sel.height() * dpr),
        }

        with mss.mss() as sct:
            img = sct.grab(monitor)
            mss.tools.to_png(img.rgb, img.size, output=str(out_path))

        self.screenshot_taken.emit(str(out_path), elapsed)
        self.show()

    @property
    def signals(self):
        return self._signals
