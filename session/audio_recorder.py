import threading
import time
import wave
from pathlib import Path
from datetime import datetime

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QThread, Signal


class AudioRecorderThread(QThread):
    audio_level = Signal(float)
    chunk_ready = Signal(str, float)
    recording_saved = Signal(str)
    error_occurred = Signal(str)

    SAMPLE_RATE = 16000
    CHANNELS = 1
    BLOCK_SIZE = 1024
    DTYPE = "float32"

    SILENCE_THRESHOLD = 0.01
    SILENCE_MIN_BLOCKS = 8
    CHUNK_MIN_BLOCKS = 125
    CHUNK_MAX_BLOCKS = 235

    def __init__(self, device_index: int, output_dir: Path, parent=None):
        super().__init__(parent)
        self._device_index = device_index
        self._output_dir = output_dir
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._current_chunk: list[np.ndarray] = []
        self._silence_count = 0
        self._block_count = 0
        self._chunk_index = 0
        self._t0 = 0.0
        self._elapsed_audio = 0.0

        self._chunk_paths: list[Path] = []

    def run(self):
        try:
            self._record()
        except Exception as exc:
            self.error_occurred.emit(str(exc))

    def _record(self):
        self._t0 = time.monotonic()

        def _callback(indata: np.ndarray, frames: int, time_info, status):
            block = indata[:, 0].copy()

            with self._lock:
                self._current_chunk.append(block)
                self._block_count += 1

                rms = float(np.sqrt(np.mean(block ** 2)))
                if rms < self.SILENCE_THRESHOLD:
                    self._silence_count += 1
                else:
                    self._silence_count = 0

                hit_max = self._block_count >= self.CHUNK_MAX_BLOCKS
                hit_silence = (
                        self._block_count >= self.CHUNK_MIN_BLOCKS
                        and self._silence_count >= self.SILENCE_MIN_BLOCKS
                )

                if hit_max or hit_silence:
                    self._flush_chunk("max-cap" if hit_max else "silence")

            self.audio_level.emit(rms)

        with sd.InputStream(
                device=self._device_index,
                channels=self.CHANNELS,
                samplerate=self.SAMPLE_RATE,
                blocksize=self.BLOCK_SIZE,
                dtype=self.DTYPE,
                callback=_callback,
        ):
            self._stop_event.wait()

        with self._lock:
            if self._current_chunk:
                self._flush_chunk("final")

        full_audio_path = self._output_dir / "full_audio.wav"
        self._write_full_wav(full_audio_path, self._chunk_paths)

        self.recording_saved.emit(str(full_audio_path))

    def stop_recording(self):
        self._stop_event.set()

    def _flush_chunk(self, reason: str = "final"):
        data = np.concatenate(self._current_chunk)
        dur = len(data) / self.SAMPLE_RATE
        elapsed = time.monotonic() - self._t0

        start_offset = self._elapsed_audio
        self._elapsed_audio += dur

        self._current_chunk = []
        self._block_count = 0
        self._silence_count = 0

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        idx = self._chunk_index
        path = self._output_dir / f"chunk_{idx:03d}_{ts}.wav"
        self._chunk_index += 1
        self._write_wav(path, data)

        self._chunk_paths.append(path)

        print(
            f"[recorder] cut chunk #{idx:03d} -> {path.name} | "
            f"{dur:4.1f}s audio | reason={reason:<7} | t+{elapsed:5.1f}s | "
            f"offset={start_offset:5.1f}s | on={threading.current_thread().name}",
            flush=True,
        )
        self.chunk_ready.emit(str(path), start_offset)

    @staticmethod
    def _write_wav(path: Path, data: np.ndarray):
        pcm = (data * 32_767).clip(-32_768, 32_767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16_000)
            wf.writeframes(pcm.tobytes())

    @staticmethod
    def _write_full_wav(out_path: Path, chunk_paths: list[Path]):
        if not chunk_paths:
            return

        print(f"[recorder] merging {len(chunk_paths)} chunks into {out_path.name}...", flush=True)

        with wave.open(str(out_path), "wb") as out_wf:
            with wave.open(str(chunk_paths[0]), "rb") as first_wf:
                out_wf.setnchannels(first_wf.getnchannels())
                out_wf.setsampwidth(first_wf.getsampwidth())
                out_wf.setframerate(first_wf.getframerate())

            for path in chunk_paths:
                with wave.open(str(path), "rb") as wf:
                    out_wf.writeframes(wf.readframes(wf.getnframes()))

        print(f"[recorder] full audio saved -> {out_path}", flush=True)