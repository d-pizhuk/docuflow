import threading
import time
import wave
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np
import sounddevice as sd
from PySide6.QtCore import QThread, Signal


class AudioRecorderThread(QThread):
    audio_level = Signal(float)
    chunk_ready = Signal(str, float)  # (path to flushed WAV chunk, start_offset secs)
    recording_saved = Signal(str)   # path to the final full WAV
    error_occurred = Signal(str)

    SAMPLE_RATE = 16_000
    CHANNELS = 1
    BLOCK_SIZE = 1_024              # ~64 ms at 16 kHz
    DTYPE = "float32"

    # VAD / chunking config
    SILENCE_THRESHOLD = 0.01        # RMS below this = silence
    SILENCE_MIN_BLOCKS = 8          # ~0.5 s of silence before a cut is allowed
    # Chunk sizing is chosen to BOUND THE POST-STOP TAIL. The only audio left to
    # transcribe at Stop is the in-progress chunk, so its hard cap is the worst
    # case the user waits on. ~15 s cap → a few seconds of transcription on any
    # reasonable CPU (turbo/int8 runs several × real-time). Silence cuts most
    # chunks earlier. Live transcription has ample headroom to keep up, so no
    # backlog forms and Stop only pays for this last short chunk.
    CHUNK_MIN_BLOCKS = 125          # ~8 s minimum before looking for a silence cut
    CHUNK_MAX_BLOCKS = 235          # ~15 s hard cap (bounds the post-Stop tail)

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
        self._elapsed_audio = 0.0   # cumulative seconds of audio cut so far

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

        # NOTE: we intentionally do NOT write a second full-session WAV here.
        # Nothing downstream reads it (only the per-chunk WAVs feed Whisper),
        # and writing hundreds of MB synchronously was the single biggest
        # contributor to post-Stop latency. The chunk WAVs are the source of
        # truth; concatenating them reproduces the full audio if ever needed.
        # Emit recording_saved with the session dir so the existing stop-flow
        # signal plumbing keeps working.
        self.recording_saved.emit(str(self._output_dir))

    def stop_recording(self):
        self._stop_event.set()

    # called with self._lock held
    def _flush_chunk(self, reason: str = "final"):
        data = np.concatenate(self._current_chunk)
        dur = len(data) / self.SAMPLE_RATE
        elapsed = time.monotonic() - self._t0

        # Each chunk's position in the recording timeline is the audio cut
        # before it. Assigned here (chunks are strictly ordered) so transcription
        # can run on multiple workers and finish out of order without scrambling
        # the timeline.
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