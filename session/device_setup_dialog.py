import threading

import numpy as np
import sounddevice as sd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFrame, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QSizePolicy, QVBoxLayout,
)

from session.language_options import (
    DEFAULT_DOCUMENTATION_LANGUAGE,
    DOCUMENTATION_LANGUAGES,
)


class DeviceSetupDialog(QDialog):
    def __init__(self, parent=None, *, populate_devices: bool = True):
        super().__init__(parent)
        self.setWindowTitle("DocuFlow — Session Setup")
        self.setMinimumWidth(500)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)

        self._devices: list[tuple[int, str]] = []
        self._test_stream: sd.InputStream | None = None
        self._audio_level = 0.0

        self._level_timer = QTimer(self)
        self._level_timer.timeout.connect(self._refresh_level_bar)

        self._setup_ui()
        if populate_devices:
            self._populate_devices()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(14)
        root.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Set Up Your Recording Session")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        root.addWidget(title)

        info = QLabel(
            "Select your microphone and documentation language. Audio is "
            "processed entirely on your machine during recording."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #555; font-size: 11px;")
        root.addWidget(info)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(sep)

        root.addWidget(self._bold_label("Microphone"))
        self._combo = QComboBox()
        self._combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._combo.currentIndexChanged.connect(self._on_device_changed)
        root.addWidget(self._combo)

        root.addWidget(self._bold_label("Documentation Language"))
        self._language_combo = QComboBox()
        self._language_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        for language in DOCUMENTATION_LANGUAGES:
            self._language_combo.addItem(language, language)
        self._language_combo.setCurrentText(DEFAULT_DOCUMENTATION_LANGUAGE)
        root.addWidget(self._language_combo)

        language_hint = QLabel(
            "Whisper detects the spoken language automatically. This setting "
            "controls the generated documentation."
        )
        language_hint.setWordWrap(True)
        language_hint.setStyleSheet("color: #888; font-size: 11px;")
        root.addWidget(language_hint)

        root.addWidget(self._bold_label("Input Level"))
        level_row = QHBoxLayout()

        self._level_bar = QProgressBar()
        self._level_bar.setRange(0, 100)
        self._level_bar.setValue(0)
        self._level_bar.setTextVisible(False)
        self._level_bar.setFixedHeight(14)
        self._level_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc; border-radius: 6px; background: #f0f0f0;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #2ecc71, stop:0.7 #f39c12, stop:1 #e74c3c);
                border-radius: 5px;
            }
        """)
        level_row.addWidget(self._level_bar)

        self._test_btn = QPushButton("Test Mic")
        self._test_btn.setFixedWidth(90)
        self._test_btn.clicked.connect(self._toggle_test)
        level_row.addWidget(self._test_btn)
        root.addLayout(level_row)

        self._status_label = QLabel("Press 'Test Mic' to verify audio input.")
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")
        root.addWidget(self._status_label)

        root.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel = QPushButton("Cancel")
        cancel.setFixedWidth(90)
        cancel.clicked.connect(self.reject)
        btn_row.addWidget(cancel)

        self._start_btn = QPushButton("Start Session →")
        self._start_btn.setFixedWidth(150)
        self._start_btn.setEnabled(False)
        self._start_btn.setStyleSheet("""
            QPushButton {
                background: #2c3e50; color: white;
                border-radius: 5px; padding: 6px 12px; font-weight: bold;
            }
            QPushButton:enabled:hover  { background: #1a252f; }
            QPushButton:disabled       { background: #95a5a6; color: #ddd; }
        """)
        self._start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self._start_btn)
        root.addLayout(btn_row)

    @staticmethod
    def _bold_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        return lbl

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

        self._start_btn.setEnabled(True)

    def _on_device_changed(self, pos: int):
        if self._test_stream is not None:
            self._stop_test()
        self._start_btn.setEnabled(pos >= 0 and pos < len(self._devices))

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
            "output_language": (
                self._language_combo.currentData()
                or DEFAULT_DOCUMENTATION_LANGUAGE
            ),
        }

    def closeEvent(self, event):
        self._stop_test()
        super().closeEvent(event)
