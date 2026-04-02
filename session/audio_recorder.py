import threading
import wave
from pathlib import Path
from datetime import datetime

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QThread, Signal


class AudioRecorderThread(QThread):
    audio_level = Signal(float)
    recording_saved = Signal(str)
    error_occurred = Signal(str)

    SAMPLE_RATE = 16_000
    CHANNELS = 1
    BLOCK_SIZE = 2048
    DTYPE = "float32"

    def __init__(self, device_index: int, output_dir: Path, parent=None):
        super().__init__(parent)
        self._device_index = device_index
        self._output_dir = output_dir
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def run(self):
        try:
            self._record()
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    def _record(self):
        def _callback(indata: np.ndarray, frames: int, time_info, status):
            chunk = indata[:, 0].copy()
            with self._lock:
                self._frames.append(chunk)

            level = float(np.abs(chunk).mean())
            self.audio_level.emit(level)

        with sd.InputStream(
                device=self._device_index, channels=self.CHANNELS,
                samplerate=self.SAMPLE_RATE, blocksize=self.BLOCK_SIZE,
                dtype=self.DTYPE, callback=_callback
        ):
            self._stop_event.wait()

        self._save_wav()

    def stop_recording(self):
        self._stop_event.set()

    def _save_wav(self):
        with self._lock:
            data = np.concatenate(self._frames) if self._frames else np.array([], dtype=np.float32)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self._output_dir / f"recording_{ts}.wav"
        pcm = (data * 32_767).clip(-32_768, 32_767).astype(np.int16)

        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())

        self.recording_saved.emit(str(out))
