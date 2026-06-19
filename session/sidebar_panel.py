import os
import time
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QRect, QEasingCurve
from PySide6.QtGui import QFont, QPixmap, QPainter, QCloseEvent, QCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QApplication, QPlainTextEdit, QDialog, QDialogButtonBox
)

from ai.transcriber import Transcriber, TranscribedChunk
from ai.transcript_assembler import TranscriptAssembler, AnnotatedStep
from session.audio_recorder import AudioRecorderThread
from session.global_overlay import GlobalOverlay
from session.mic_indicator import MicIndicatorWidget
from session.screenshot_capture import CapturedScreenshot


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


class TranscriptPreviewDialog(QDialog):
    """Read-only transcript preview shown after a session ends."""

    def __init__(self, steps: list[AnnotatedStep], full_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Session Transcript")
        self.resize(700, 500)

        layout = QVBoxLayout(self)

        info = QLabel(f"{len(steps)} annotated step(s) · scroll to review")
        info.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(info)

        self._editor = QPlainTextEdit()
        self._editor.setReadOnly(True)
        self._editor.setFont(QFont("Consolas", 10))
        self._editor.setPlainText(self._format(steps, full_text))
        layout.addWidget(self._editor)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    @staticmethod
    def _format(steps: list[AnnotatedStep], full_text: str) -> str:
        if not steps:
            return f"[No steps assembled]\n\nFull transcript:\n{full_text}"

        lines = []
        for i, step in enumerate(steps, 1):
            ts = f"{step.start_sec:.1f}s – {step.end_sec:.1f}s"
            ss = f"  📷 {step.screenshot.path.name}" if step.screenshot else ""
            lines.append(f"── Step {i}  [{ts}]{ss}")
            lines.append(step.text)
            lines.append("")
        return "\n".join(lines)


class SidebarPanel(QWidget):
    _EXPANDED_W = 340
    _COLLAPSED_W = 40
    _HEIGHT = 480

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config = config
        self._session_start_mono = time.time()
        self._session_start_utc = datetime.now(timezone.utc)

        ts = self._session_start_utc.strftime("%Y%m%d_%H%M%S")
        self._session_dir = Path.home() / "DocuFlow" / "sessions" / ts
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self._is_expanded = False
        self._current_screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
        self._stopping = False
        self._t_stop = 0.0

        # AI pipeline
        self._assembler = TranscriptAssembler(self._session_start_utc)

        transcriber = config.get("transcriber")
        if transcriber is not None:
            transcriber._on_result = self._on_chunk_transcribed
            self._transcriber = transcriber
        else:
            self._transcriber = Transcriber(on_chunk_transcribed=self._on_chunk_transcribed)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
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

        self._overlay = GlobalOverlay(self._session_dir, self._session_start_mono)
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
        self._thumb.setStyleSheet(
            "background: #0d0d1a; border: 1px solid #2a2a3e; border-radius: 5px; color: #555;")
        mp_layout.addWidget(self._thumb)

        # Live transcript feed (last chunk text)
        self._transcript_lbl = QLabel("Transcript will appear here…")
        self._transcript_lbl.setWordWrap(True)
        self._transcript_lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._transcript_lbl.setStyleSheet(
            "color: #7f8c8d; font-size: 10px; background: #0d0d1a;"
            "border: 1px solid #2a2a3e; border-radius: 5px; padding: 6px;")
        self._transcript_lbl.setFixedHeight(60)
        mp_layout.addWidget(self._transcript_lbl)

        self._status_lbl = QLabel("")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet("color: #f39c12; font-size: 10px;")
        mp_layout.addWidget(self._status_lbl)

        mp_layout.addStretch()

        self._stop_btn = QPushButton("■   Stop Session")
        self._stop_btn.setFixedHeight(46)
        self._stop_btn.setStyleSheet("""
            QPushButton {
                background: #922b21;
                color: white;
                border-radius: 7px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background: #e74c3c; }
            QPushButton:pressed { background: #641e16; }
            QPushButton:disabled { background: #444; color: #888; }
        """)
        self._stop_btn.clicked.connect(self._on_stop)
        mp_layout.addWidget(self._stop_btn)

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
        target_x = (
            geo.right() - self._EXPANDED_W + 1
            if self._is_expanded
            else geo.right() - self._COLLAPSED_W + 1
        )
        target_rect = QRect(target_x, y, self._EXPANDED_W, self._HEIGHT)
        if animate:
            if self._anim.endValue() != target_rect:
                self._anim.setEndValue(target_rect)
                self._anim.start()
        else:
            self.setGeometry(target_rect)

    def _update_clock(self):
        elapsed = int(time.time() - self._session_start_mono)
        h, r = divmod(elapsed, 3600)
        m, s = divmod(r, 60)
        t_str = f"{h:02d}:{m:02d}:{s:02d}"
        self._timer_lbl.setText(t_str)
        self._tab.set_time(t_str)

    def _start_audio(self):
        self._recorder = AudioRecorderThread(self._config["device_index"], self._session_dir)
        self._recorder.audio_level.connect(self._mic_widget.set_level)
        self._recorder.chunk_ready.connect(self._on_chunk_ready)
        self._recorder.recording_saved.connect(self._on_recording_saved)
        self._recorder.start()

    # ------------------------------------------------------------------ #
    # Screenshot
    # ------------------------------------------------------------------ #

    def _on_screenshot_done(self, path: str, elapsed: float):
        # Convert elapsed seconds → absolute UTC timestamp for assembler alignment
        from datetime import timedelta
        ts = self._session_start_utc + timedelta(seconds=elapsed)
        screenshot = CapturedScreenshot(path=Path(path), timestamp=ts)
        self._assembler.add_screenshot(screenshot)

        pix = QPixmap(path)
        if not pix.isNull():
            scaled = pix.scaled(
                self._thumb.width() - 4, self._thumb.height() - 4,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._thumb.setPixmap(scaled)

    # ------------------------------------------------------------------ #
    # Audio chunks → transcription
    # ------------------------------------------------------------------ #

    def _on_chunk_ready(self, chunk_path: str, start_offset: float):
        self._transcriber.submit(chunk_path, start_offset)

    def _on_chunk_transcribed(self, chunk: TranscribedChunk):
        # Called from the Transcriber worker thread — post to main thread via signal-safe update
        self._assembler.add_chunk(chunk)
        preview = chunk.text[:120] + ("…" if len(chunk.text) > 120 else "")
        # Qt label updates must happen on the main thread
        QApplication.instance().postEvent(self, _TranscriptUpdateEvent(preview))

    # ------------------------------------------------------------------ #
    # Stop flow
    # ------------------------------------------------------------------ #

    def _on_stop(self):
        if self._stopping:
            return
        self._stopping = True
        self._t_stop = time.monotonic()   # mark when recording was stopped
        self._stop_btn.setEnabled(False)
        self._stop_btn.setText("⏳  Processing…")
        self._status_lbl.setText("Finishing transcription…")

        self._tick.stop()
        self._tracker.stop()
        self._overlay.stop_listener()
        self._overlay.close()

        self._recorder.stop_recording()
        # _on_recording_saved fires when the recorder finishes flushing

    def _on_recording_saved(self, wav_path: str):
        self._status_lbl.setText("Waiting for transcription to finish…")
        # finish() blocks the worker thread — run in a Qt thread so UI stays alive
        from PySide6.QtCore import QThreadPool, QRunnable

        assembler = self._assembler
        transcriber = self._transcriber

        class _Finaliser(QRunnable):
            def __init__(self, panel):
                super().__init__()
                self._panel = panel

            def run(self):
                transcriber.finish()
                steps = assembler.assemble()
                full_text = assembler.full_transcript()
                elapsed = time.monotonic() - self._panel._t_stop
                print(
                    f"[session] transcript ready {elapsed:.1f}s after recording stopped "
                    f"({len(steps)} step(s))",
                    flush=True,
                )
                # Post result back to main thread
                QApplication.instance().postEvent(
                    self._panel, _SessionDoneEvent(steps, full_text, wav_path, elapsed)
                )

        QThreadPool.globalInstance().start(_Finaliser(self))

    # ------------------------------------------------------------------ #
    # Custom Qt events (cross-thread UI updates)
    # ------------------------------------------------------------------ #

    def event(self, ev):
        if isinstance(ev, _TranscriptUpdateEvent):
            self._transcript_lbl.setText(ev.text)
            return True
        if isinstance(ev, _SessionDoneEvent):
            self._show_results(ev.steps, ev.full_text, ev.wav_path, ev.elapsed)
            return True
        return super().event(ev)

    def _show_results(self, steps: list[AnnotatedStep], full_text: str, wav_path: str, elapsed: float = 0.0):
        self._status_lbl.setText(f"Done · {len(steps)} step(s) · ready {elapsed:.1f}s after stop")
        dlg = TranscriptPreviewDialog(steps, full_text, self)
        dlg.exec()
        self.close()

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def closeEvent(self, event: QCloseEvent):
        if hasattr(self, "_overlay") and self._overlay:
            self._overlay.stop_listener()
        if hasattr(self, "_recorder") and self._recorder.isRunning():
            self._recorder.stop_recording()
            self._recorder.wait(4_000)
        super().closeEvent(event)


# ------------------------------------------------------------------ #
# Private Qt events for cross-thread UI updates
# ------------------------------------------------------------------ #

from PySide6.QtCore import QEvent

class _TranscriptUpdateEvent(QEvent):
    _TYPE = QEvent.Type(QEvent.registerEventType())

    def __init__(self, text: str):
        super().__init__(self._TYPE)
        self.text = text


class _SessionDoneEvent(QEvent):
    _TYPE = QEvent.Type(QEvent.registerEventType())

    def __init__(self, steps: list[AnnotatedStep], full_text: str, wav_path: str, elapsed: float = 0.0):
        super().__init__(self._TYPE)
        self.steps = steps
        self.full_text = full_text
        self.wav_path = wav_path
        self.elapsed = elapsed