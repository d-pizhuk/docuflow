import os
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QRect, QEasingCurve
from PySide6.QtGui import QFont, QPixmap, QPainter, QCloseEvent, QCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, QApplication
)

from session.audio_recorder import AudioRecorderThread
from session.global_overlay import GlobalOverlay
from session.mic_indicator import MicIndicatorWidget


class TabWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(40)
        self.setStyleSheet(
            "background: #e74c3c; color: white; border-top-left-radius: 8px; border-bottom-left-radius: 8px;")
        self.time_str = "00:00:00"

    def set_time(self, time_str: str):
        self.time_str = time_str
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setPen(Qt.GlobalColor.white)
        p.setFont(QFont("Consolas", 11, QFont.Weight.Bold))

        p.translate(self.width() / 2 + 5, self.height() / 2 + 35)
        p.rotate(-90)
        p.drawText(0, 0, self.time_str)


def _emergency_kill():
    print("Emergency kill switch activated.")
    os._exit(1)


class SidebarPanel(QWidget):
    _EXPANDED_W = 340
    _COLLAPSED_W = 40
    _HEIGHT = 480

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config
        self._session_start = time.time()
        self._screenshots: list[dict] = []

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = Path.home() / "DocuFlow" / "sessions" / ts
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self._is_expanded = False
        self._current_screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._build_ui()
        self._snap_to_screen(animate=False)

        self._tick = QTimer(self)
        self._tick.timeout.connect(self._update_clock)
        self._tick.start(500)

        self._tracker = QTimer(self)
        self._tracker.timeout.connect(self._track_mouse_and_monitor)
        self._tracker.start(100)

        self._overlay = GlobalOverlay(self._session_dir, self._session_start)
        self._overlay.screenshot_taken.connect(self._on_screenshot_done)
        self._overlay.signals.kill_app.connect(_emergency_kill)
        self._overlay.show()

        self._start_audio()

    def _build_ui(self):
        self.resize(self._EXPANDED_W, self._HEIGHT)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._tab = TabWidget()
        root.addWidget(self._tab)

        self._main_panel = QWidget()
        self._main_panel.setStyleSheet(
            "background: #1a1a2e; border-top-left-radius: 0px; border-bottom-left-radius: 0px;")
        mp_layout = QVBoxLayout(self._main_panel)
        mp_layout.setContentsMargins(20, 20, 20, 20)
        mp_layout.setSpacing(15)
        root.addWidget(self._main_panel)

        self._timer_lbl = QLabel("00:00:00")
        self._timer_lbl.setFont(QFont("Consolas", 32, QFont.Weight.Bold))
        self._timer_lbl.setStyleSheet("color: #ecf0f1;")
        self._timer_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mp_layout.addWidget(self._timer_lbl)

        self._mic_widget = MicIndicatorWidget()
        mp_layout.addWidget(self._mic_widget)

        lbl_hint = QLabel("Hold <b>Ctrl</b> + Drag to capture screenshot")
        lbl_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_hint.setStyleSheet("color: #aab7c4; font-size: 11px;")
        mp_layout.addWidget(lbl_hint)

        self._thumb = QLabel("No captures yet")
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setFixedHeight(120)
        self._thumb.setStyleSheet("background: #0d0d1a; border: 1px solid #2a2a3e; border-radius: 5px; color: #555;")
        mp_layout.addWidget(self._thumb)

        mp_layout.addStretch()

        stop_btn = QPushButton("■   Stop Session")
        stop_btn.setFixedHeight(46)
        stop_btn.setStyleSheet("""
            QPushButton { 
                background: #922b21; 
                color: white; 
                border-radius: 7px; 
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { 
                background: #e74c3c; 
            }
            QPushButton:pressed { 
                background: #641e16; 
            }
        """)
        stop_btn.clicked.connect(self._on_stop)
        mp_layout.addWidget(stop_btn)

        self._anim = QPropertyAnimation(self, b"geometry")
        self._anim.setDuration(250)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _track_mouse_and_monitor(self):
        global_pos = QCursor.pos()

        screen = QApplication.screenAt(global_pos)
        if screen and screen != self._current_screen:
            self._current_screen = screen
            self._snap_to_screen(animate=False)

        is_hovering = self.geometry().contains(global_pos)

        if is_hovering and not self._is_expanded:
            self._is_expanded = True
            self._snap_to_screen(animate=True)
        elif not is_hovering and self._is_expanded:
            self._is_expanded = False
            self._snap_to_screen(animate=True)

    def _snap_to_screen(self, animate=True):
        geo = self._current_screen.geometry()
        y = geo.y() + (geo.height() - self._HEIGHT) // 2

        target_x = geo.right() - self._EXPANDED_W + 1 if self._is_expanded else geo.right() - self._COLLAPSED_W + 1

        if animate:
            if self._anim.endValue() != QRect(target_x, y, self._EXPANDED_W, self._HEIGHT):
                self._anim.setEndValue(QRect(target_x, y, self._EXPANDED_W, self._HEIGHT))
                self._anim.start()
        else:
            self.setGeometry(target_x, y, self._EXPANDED_W, self._HEIGHT)

    def _update_clock(self):
        elapsed = int(time.time() - self._session_start)
        h, r = divmod(elapsed, 3600)
        m, s = divmod(r, 60)
        t_str = f"{h:02d}:{m:02d}:{s:02d}"

        self._timer_lbl.setText(t_str)
        self._tab.set_time(t_str)

    def _start_audio(self):
        self._recorder = AudioRecorderThread(self._config["device_index"], self._session_dir)
        self._recorder.audio_level.connect(self._mic_widget.set_level)
        self._recorder.recording_saved.connect(self._on_recording_saved)
        self._recorder.start()

    def _on_screenshot_done(self, path: str, elapsed: float):
        self._screenshots.append({"path": path, "elapsed": elapsed})

        pix = QPixmap(path)
        if not pix.isNull():
            scaled = pix.scaled(
                self._thumb.width() - 4, self._thumb.height() - 4,
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            self._thumb.setPixmap(scaled)

    def _on_stop(self):
        self._tick.stop()
        self._tracker.stop()
        self._overlay.stop_listener()
        self._overlay.close()

        self._recorder.stop_recording()

    def _on_recording_saved(self, wav_path: str):
        QMessageBox.information(
            self, "Session Complete",
            f"Audio saved to:\n{wav_path}\n\n{len(self._screenshots)} screenshot(s) saved."
        )
        self.close()

    def closeEvent(self, event: QCloseEvent):
        if hasattr(self, '_overlay') and self._overlay:
            self._overlay.stop_listener()
        if hasattr(self, '_recorder') and self._recorder.isRunning():
            self._recorder.stop_recording()
            self._recorder.wait(4000)
        super().closeEvent(event)
