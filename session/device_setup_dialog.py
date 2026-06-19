import threading

import numpy as np
import sounddevice as sd
from PySide6.QtCore import Qt, QTimer, QEvent, QCoreApplication
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFrame, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QSizePolicy, QVBoxLayout,
)


class _ModelReadyEvent(QEvent):
    _TYPE = QEvent.Type(QEvent.registerEventType())

    def __init__(self, success: bool, error: str = ""):
        super().__init__(self._TYPE)
        self.success = success
        self.error = error


class DeviceSetupDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DocuFlow — Device Setup")
        self.setMinimumWidth(520)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)

        self._devices: list[tuple[int, str]] = []
        self._test_stream: sd.InputStream | None = None
        self._audio_level = 0.0
        self._transcriber = None
        self._model_loaded = False

        self._level_timer = QTimer(self)
        self._level_timer.timeout.connect(self._refresh_level_bar)

        self._setup_ui()
        self._populate_devices()
        self._begin_model_load()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(14)
        root.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Set Up Your Recording Session")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        root.addWidget(title)

        info = QLabel(
            "Select your microphone. Audio is processed entirely on your machine — "
            "nothing is sent to the cloud during recording (NfReq8)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #555; font-size: 11px;")
        root.addWidget(info)

        root.addWidget(_hsep())
        root.addWidget(_bold_label("Microphone"))

        self._combo = QComboBox()
        self._combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._combo.currentIndexChanged.connect(self._on_device_changed)
        root.addWidget(self._combo)

        root.addWidget(_bold_label("Input Level"))
        level_row = QHBoxLayout()

        self._level_bar = QProgressBar()
        self._level_bar.setRange(0, 100)
        self._level_bar.setValue(0)
        self._level_bar.setTextVisible(False)
        self._level_bar.setFixedHeight(14)
        self._level_bar.setStyleSheet(_MIC_BAR_STYLE)
        level_row.addWidget(self._level_bar)

        self._test_btn = QPushButton("Test Mic")
        self._test_btn.setFixedWidth(90)
        self._test_btn.clicked.connect(self._toggle_test)
        level_row.addWidget(self._test_btn)
        root.addLayout(level_row)

        self._status_label = QLabel("Press 'Test Mic' to verify audio input.")
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")
        root.addWidget(self._status_label)

        root.addWidget(_hsep())
        root.addWidget(_bold_label("Speech Recognition Model"))

        self._model_bar = QProgressBar()
        self._model_bar.setRange(0, 0)      # pulsing / indeterminate
        self._model_bar.setTextVisible(False)
        self._model_bar.setFixedHeight(18)
        self._model_bar.setStyleSheet(_MODEL_BAR_STYLE)
        root.addWidget(self._model_bar)

        self._model_label = QLabel("⏳  Loading Whisper model from disk…")
        self._model_label.setStyleSheet("color: #888; font-size: 11px;")
        root.addWidget(self._model_label)

        root.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel = QPushButton("Cancel")
        cancel.setFixedWidth(90)
        cancel.clicked.connect(self.reject)
        btn_row.addWidget(cancel)

        self._start_btn = QPushButton("Start Session →")
        self._start_btn.setFixedWidth(160)
        self._start_btn.setEnabled(False)
        self._start_btn.setStyleSheet(_START_BTN_STYLE)
        self._start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self._start_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------ #
    # Model load
    # ------------------------------------------------------------------ #

    def _begin_model_load(self):
        dialog = self

        def _load():
            # NOTE: cross-thread UI notification MUST go through
            # QCoreApplication.postEvent (thread-safe). QTimer.singleShot posts
            # to the *calling* thread's event loop — this worker has none, so the
            # event would never fire and the dialog would hang on "Loading…".
            try:
                from ai.transcriber import Transcriber
                t = Transcriber(on_chunk_transcribed=lambda _: None)
                t.wait_until_ready()
                if t._model is not None:
                    dialog._transcriber = t
                    QCoreApplication.postEvent(dialog, _ModelReadyEvent(success=True))
                else:
                    QCoreApplication.postEvent(
                        dialog,
                        _ModelReadyEvent(success=False, error="Model failed to load (see logs)."),
                    )
            except Exception as exc:
                QCoreApplication.postEvent(
                    dialog, _ModelReadyEvent(success=False, error=str(exc))
                )

        threading.Thread(target=_load, daemon=True, name="model-load").start()

    def event(self, ev):
        if isinstance(ev, _ModelReadyEvent):
            self._on_model_ready(ev.success, ev.error)
            return True
        return super().event(ev)

    def _on_model_ready(self, success: bool, error: str):
        self._model_bar.setRange(0, 1)
        self._model_bar.setValue(1 if success else 0)

        if success:
            self._model_loaded = True
            self._model_label.setText("✅  Whisper large-v3-turbo ready")
            self._model_label.setStyleSheet("color: #27ae60; font-size: 11px;")
            self._update_start_btn()
        else:
            self._model_label.setText(f"❌  {error}")
            self._model_label.setStyleSheet("color: #e74c3c; font-size: 11px;")

    def _update_start_btn(self):
        has_mic = bool(self._devices) and self._combo.currentIndex() >= 0
        self._start_btn.setEnabled(self._model_loaded and has_mic)

    # ------------------------------------------------------------------ #
    # Mic / device
    # ------------------------------------------------------------------ #

    def _populate_devices(self):
        try:
            for i, dev in enumerate(sd.query_devices()):
                if dev["max_input_channels"] > 0:
                    self._devices.append((i, dev["name"]))
                    self._combo.addItem(dev["name"], i)
        except Exception as exc:
            self._status_label.setText(f"Error querying devices: {exc}")
            return

        if not self._devices:
            self._status_label.setText("No input devices found.")
            return

        try:
            default = sd.default.device[0]
            for pos, (idx, _) in enumerate(self._devices):
                if idx == default:
                    self._combo.setCurrentIndex(pos)
                    break
        except Exception:
            pass

        self._update_start_btn()

    def _on_device_changed(self, pos: int):
        if self._test_stream is not None:
            self._stop_test()
        self._update_start_btn()

    def _toggle_test(self):
        if self._test_stream is None:
            self._start_test()
        else:
            self._stop_test()

    def _start_test(self):
        pos = self._combo.currentIndex()
        if pos < 0 or pos >= len(self._devices):
            return
        dev_idx = self._devices[pos][0]

        def _cb(indata, frames, time_info, status):
            self._audio_level = float(np.abs(indata).mean())

        try:
            self._test_stream = sd.InputStream(
                device=dev_idx, channels=1, samplerate=16_000,
                blocksize=1024, dtype="float32", callback=_cb,
            )
            self._test_stream.start()
            self._level_timer.start(80)
            self._test_btn.setText("Stop Test")
            self._status_label.setText("Listening… speak into your microphone.")
            self._status_label.setStyleSheet("color: #27ae60; font-size: 11px;")
        except Exception as exc:
            self._status_label.setText(f"Cannot open device: {exc}")
            self._status_label.setStyleSheet("color: #e74c3c; font-size: 11px;")

    def _stop_test(self):
        self._level_timer.stop()
        if self._test_stream:
            try:
                self._test_stream.stop()
                self._test_stream.close()
            except Exception:
                pass
            self._test_stream = None
        self._level_bar.setValue(0)
        self._test_btn.setText("Test Mic")
        self._status_label.setText("Press 'Test Mic' to verify audio input.")
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")

    def _refresh_level_bar(self):
        self._level_bar.setValue(min(int(self._audio_level * 1000), 100))

    def _on_start(self):
        self._stop_test()
        pos = self._combo.currentIndex()
        if 0 <= pos < len(self._devices):
            self._sel_idx, self._sel_name = self._devices[pos]
        else:
            self._sel_idx, self._sel_name = None, "Unknown"
        self.accept()

    def get_config(self) -> dict:
        return {
            "device_index": getattr(self, "_sel_idx", self._devices[0][0] if self._devices else 0),
            "device_name": getattr(self, "_sel_name", "Unknown"),
            "transcriber": self._transcriber,
        }

    def closeEvent(self, event):
        self._stop_test()
        super().closeEvent(event)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _hsep() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.HLine)
    sep.setFrameShadow(QFrame.Shadow.Sunken)
    return sep


def _bold_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
    return lbl


_MIC_BAR_STYLE = """
    QProgressBar {
        border: 1px solid #ccc; border-radius: 6px; background: #f0f0f0;
    }
    QProgressBar::chunk {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #2ecc71, stop:0.7 #f39c12, stop:1 #e74c3c);
        border-radius: 5px;
    }
"""

_MODEL_BAR_STYLE = """
    QProgressBar {
        border: 1px solid #ccc; border-radius: 6px;
        background: #f0f0f0; text-align: center; font-size: 10px;
    }
    QProgressBar::chunk {
        background: #3498db; border-radius: 5px;
    }
"""

_START_BTN_STYLE = """
    QPushButton {
        background: #2c3e50; color: white;
        border-radius: 5px; padding: 6px 12px; font-weight: bold;
    }
    QPushButton:enabled:hover  { background: #1a252f; }
    QPushButton:disabled       { background: #95a5a6; color: #ddd; }
"""