import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QRect, QEasingCurve, QEvent
from PySide6.QtGui import QFont, QPixmap, QPainter, QCloseEvent, QCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QApplication, QPlainTextEdit, QDialog, QDialogButtonBox,
    QMessageBox,
)

from ai.transcriber import Transcriber, TranscribedChunk
from ai.transcript_assembler import TranscriptAssembler, AnnotatedStep
from ai.step_structurer import StepStructurer
from ai.screenshot_describer import ScreenshotDescriber
from ai.languages import WHISPER_LANGUAGE_CODES
from ai.doc_merger import DocMerger, MergedDoc
from ai.api_gateway import ApiGateway, ApiGatewayError
from session.audio_recorder import AudioRecorderThread
from session.global_overlay import GlobalOverlay
from session.mic_indicator import MicIndicatorWidget
from session.screenshot_capture import CapturedScreenshot
from session.review_edit import ReviewEditWindow


class TabWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(40)
        self.setStyleSheet(
            "background: #2c3e50; color: white; border-top-left-radius: 8px; border-bottom-left-radius: 8px;")
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
    sys.exit(1)


class TranscriptApprovalDialog(QDialog):
    def __init__(self, steps: list[AnnotatedStep], annotated: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review Transcript")
        self.resize(700, 520)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)

        layout = QVBoxLayout(self)

        info = QLabel(
            f"{len(steps)} segment(s) transcribed. Review the text below and fix any "
            "mistakes, then generate the documentation. The [SCREENSHOT: …] markers "
            "show where each screenshot will be placed — keep them as they are."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(info)

        self._editor = QPlainTextEdit()
        self._editor.setFont(QFont("Consolas", 10))
        self._editor.setPlainText(annotated)
        layout.addWidget(self._editor)

        buttons = QDialogButtonBox()
        buttons.addButton("Generate Documentation →", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def edited_transcript(self) -> str:
        return self._editor.toPlainText().strip()


class TranscriptionProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._programmatic_close = False
        self.setWindowTitle("Processing…")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint |
            Qt.WindowType.WindowTitleHint | Qt.WindowType.WindowMinimizeButtonHint
        )
        self.resize(380, 120)

        layout = QVBoxLayout(self)
        lbl = QLabel(
            "Transcription is in progress…\nThis can take up to 60 seconds depending on the audio length.\nPlease wait.")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

    def closeEvent(self, event):
        if not self._programmatic_close and self.parent():
            self.parent().close()
        super().closeEvent(event)


class GenerationProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._programmatic_close = False
        self.setWindowTitle("Working…")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint |
            Qt.WindowType.WindowTitleHint | Qt.WindowType.WindowMinimizeButtonHint
        )
        self.resize(380, 120)

        layout = QVBoxLayout(self)
        lbl = QLabel(
            "Generating documentation…\nStructuring the transcript and describing screenshots.\nThis can take up to ~90 seconds.")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

    def closeEvent(self, event):
        if not self._programmatic_close and self.parent():
            self.parent().close()
        super().closeEvent(event)


_MAX_TRANSCRIPT_WORDS = 4000


class SidebarPanel(QWidget):
    _EXPANDED_W = 340
    _COLLAPSED_W = 40
    _HEIGHT = 480

    def __init__(self, config: dict, settings, parent=None):
        super().__init__(parent)
        self._config = config
        self._settings = settings
        self._session_start_mono = time.time()
        self._session_start_utc = datetime.now(timezone.utc)

        ts = self._session_start_utc.strftime("%Y%m%d_%H%M%S")
        self._session_dir = Path(self._settings.output_dir) / ts
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self._is_expanded = False
        self._current_screen = QApplication.screenAt(QCursor.pos()) or QApplication.primaryScreen()
        self._stopping = False
        self._t_stop = 0.0
        self._progress_dlg: GenerationProgressDialog | None = None
        self._transcription_dlg: TranscriptionProgressDialog | None = None

        self._assembler = TranscriptAssembler(self._session_start_utc)

        transcriber = config.get("transcriber")
        if transcriber is not None:
            transcriber._on_result = self._on_chunk_transcribed
            self._transcriber = transcriber
        else:
            doc_lang_str = self._settings.documentation_language
            whisper_lang = WHISPER_LANGUAGE_CODES.get(doc_lang_str, "en")
            self._transcriber = Transcriber(
                on_chunk_transcribed=self._on_chunk_transcribed,
                language=whisper_lang
            )

        self._gateway = ApiGateway(
            base_url=self._settings.api_base_url,
            api_key=self._settings.api_key
        )

        doc_lang = self._settings.documentation_language
        self._structurer = StepStructurer(
            gateway=self._gateway,
            model=self._settings.llm_model,
            language=doc_lang
        )
        self._describer = ScreenshotDescriber(
            gateway=self._gateway,
            model=self._settings.vlm_model,
            language=doc_lang
        )
        self._merger = DocMerger()

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
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
        self._thumb.setStyleSheet("background: #0d0d1a; border: 1px solid #2a2a3e; border-radius: 5px; color: #555;")
        mp_layout.addWidget(self._thumb)

        self._transcript_lbl = QLabel("Transcript will appear here…")
        self._transcript_lbl.setWordWrap(True)
        self._transcript_lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._transcript_lbl.setStyleSheet(
            "color: #7f8c8d; font-size: 10px; background: #0d0d1a; border: 1px solid #2a2a3e; border-radius: 5px; padding: 6px;")
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
            QPushButton { background: #922b21; color: white; border-radius: 7px; font-weight: bold; font-size: 14px; }
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

    def _on_screenshot_done(self, path: str, elapsed: float):
        from datetime import timedelta
        ts = self._session_start_utc + timedelta(seconds=elapsed)
        screenshot = CapturedScreenshot(path=Path(path), timestamp=ts)
        self._assembler.add_screenshot(screenshot)

        pix = QPixmap(path)
        if not pix.isNull():
            scaled = pix.scaled(
                self._thumb.width() - 4, self._thumb.height() - 4,
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation,
            )
            self._thumb.setPixmap(scaled)

    def _on_chunk_ready(self, chunk_path: str, start_offset: float):
        self._transcriber.submit(chunk_path, start_offset)

    def _on_chunk_transcribed(self, chunk: TranscribedChunk):
        self._assembler.add_chunk(chunk)
        preview = chunk.text[:120] + ("…" if len(chunk.text) > 120 else "")
        QApplication.instance().postEvent(self, _TranscriptUpdateEvent(preview))

    def _on_stop(self):
        if self._stopping: return
        self._stopping = True
        self._t_stop = time.monotonic()
        self._stop_btn.setEnabled(False)

        self._tick.stop()
        self._tracker.stop()
        self._overlay.stop_listener()
        self._overlay.close()

        self._recorder.stop_recording()
        self.hide()

        self._transcription_dlg = TranscriptionProgressDialog(self)
        self._transcription_dlg.show()

    def _on_recording_saved(self, wav_path: str):
        from PySide6.QtCore import QThreadPool, QRunnable

        assembler = self._assembler
        transcriber = self._transcriber
        panel = self

        class _Finaliser(QRunnable):
            def run(self):
                transcriber.finish()
                steps = assembler.assemble()
                full_text = assembler.full_transcript()
                annotated = assembler.annotated_transcript()
                session_dir = panel._session_dir

                try:
                    (session_dir / "transcript.txt").write_text(annotated, encoding="utf-8")
                except Exception:
                    pass

                QApplication.instance().postEvent(panel, _TranscriptReadyEvent(steps, full_text, annotated, 0.0))

        QThreadPool.globalInstance().start(_Finaliser())

    def _review_transcript(self, steps: list[AnnotatedStep], full_text: str, annotated: str, elapsed: float):
        dlg = TranscriptApprovalDialog(steps, annotated, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            approved = dlg.edited_transcript()
            self._start_generation(approved, steps, full_text)
        else:
            self.close()

    def _start_generation(self, annotated: str, steps: list[AnnotatedStep], full_text: str):
        from PySide6.QtCore import QThreadPool, QRunnable

        word_count = len(annotated.split())
        if word_count > _MAX_TRANSCRIPT_WORDS:
            reply = QMessageBox.warning(
                self,
                "Long Recording Warning",
                f"The transcript contains {word_count:,} words, which exceeds the "
                f"recommended limit of {_MAX_TRANSCRIPT_WORDS:,} words.\n\n"
                "The LLM may fail, truncate, or produce low-quality output. "
                "Consider splitting your recording into shorter sessions.\n\n"
                "Do you want to proceed anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                self._show_results(annotated, merged=None, error=None)
                return

        self._progress_dlg = GenerationProgressDialog(self)
        self._progress_dlg.show()

        structurer = self._structurer
        describer = self._describer
        merger = self._merger
        screenshot_names = self._assembler.screenshot_names()
        session_dir = self._session_dir
        panel = self
        t0 = time.monotonic()

        class _GenerateRunnable(QRunnable):
            def run(self):
                merged = None
                error = None
                try:
                    doc = structurer.structure(annotated, valid_screenshots=screenshot_names, session_dir=session_dir)
                    (session_dir / "steps.json").write_text(json.dumps(doc.to_json(), indent=2), encoding="utf-8")

                    descriptions = describer.describe_all(doc, session_dir)
                    merged = merger.merge(doc, descriptions)
                    (session_dir / "documentation.json").write_text(json.dumps(merged.to_json(), indent=2),
                                                                    encoding="utf-8")
                except ApiGatewayError as exc:
                    error = str(exc)
                except Exception as exc:
                    error = f"Unexpected error: {exc}"

                QApplication.instance().postEvent(panel,
                                                  _SessionDoneEvent(steps, full_text, annotated, merged, error, 0.0))

        QThreadPool.globalInstance().start(_GenerateRunnable())

    def event(self, ev):
        if isinstance(ev, _TranscriptUpdateEvent):
            if self.isVisible():
                self._transcript_lbl.setText(ev.text)
            return True
        if isinstance(ev, _TranscriptReadyEvent):
            if self._transcription_dlg is not None:
                self._transcription_dlg._programmatic_close = True
                self._transcription_dlg.close()
                self._transcription_dlg = None
            self._review_transcript(ev.steps, ev.full_text, ev.annotated, ev.elapsed)
            return True
        if isinstance(ev, _SessionDoneEvent):
            self._show_results(ev.annotated, ev.merged, ev.error, ev.elapsed)
            return True
        return super().event(ev)

    def _show_results(self, annotated: str, merged: MergedDoc | None = None, error: str | None = None,
                      elapsed: float = 0.0):
        if self._progress_dlg is not None:
            self._progress_dlg._programmatic_close = True
            self._progress_dlg.close()
            self._progress_dlg = None

        win = ReviewEditWindow(merged, annotated, self._session_dir, error=error, parent=self)
        win.exec()
        self.close()

    def closeEvent(self, event: QCloseEvent):
        if hasattr(self, "_overlay") and self._overlay:
            self._overlay.stop_listener()
        if hasattr(self, "_recorder") and self._recorder.isRunning():
            self._recorder.stop_recording()
            self._recorder.wait(4_000)
        super().closeEvent(event)
        sys.exit(0)


class _TranscriptUpdateEvent(QEvent):
    _TYPE = QEvent.Type(QEvent.registerEventType())

    def __init__(self, text: str):
        super().__init__(self._TYPE)
        self.text = text


class _TranscriptReadyEvent(QEvent):
    _TYPE = QEvent.Type(QEvent.registerEventType())

    def __init__(self, steps: list[AnnotatedStep], full_text: str, annotated: str, elapsed: float = 0.0):
        super().__init__(self._TYPE)
        self.steps = steps
        self.full_text = full_text
        self.annotated = annotated
        self.elapsed = elapsed


class _SessionDoneEvent(QEvent):
    _TYPE = QEvent.Type(QEvent.registerEventType())

    def __init__(self, steps: list[AnnotatedStep], full_text: str, annotated: str, merged: MergedDoc | None = None,
                 error: str | None = None, elapsed: float = 0.0):
        super().__init__(self._TYPE)
        self.steps = steps
        self.full_text = full_text
        self.annotated = annotated
        self.merged = merged
        self.error = error
        self.elapsed = elapsed