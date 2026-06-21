# ai/transcriber.py
import logging
import os
import queue
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

_MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"


def _resolve_model_dir() -> Path:
    turbo = _MODELS_ROOT / "whisper-large-v3-turbo"
    legacy = _MODELS_ROOT / "whisper-distil-large-v3"
    if turbo.exists():
        return turbo
    if legacy.exists():
        return legacy
    return turbo


MODEL_DIR = _resolve_model_dir()
DEVICE = "cpu"
COMPUTE_TYPE = "int8"


def _physical_cores() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    logical = os.cpu_count() or 4
    return max(1, logical // 2) if logical >= 4 else max(1, logical)


NUM_WORKERS = 1
CPU_THREADS = _physical_cores()

_REQUIRED_FILES = {"model.bin", "config.json", "tokenizer.json"}

ENABLE_WORD_TIMESTAMPS = False
_WORD_TIMESTAMP_MIN_SECONDS = 4.0


@dataclass
class TranscribedChunk:
    chunk_path: Path
    text: str
    words: list[tuple[float, float, str]] = field(default_factory=list)
    start_offset: float = 0.0
    duration: float = 0.0


@dataclass
class _Job:
    path: Path
    start_offset: float


class Transcriber:
    def __init__(self, on_chunk_transcribed: Callable[[TranscribedChunk], None], language: str):
        self._validate_model_dir()
        self._on_result = on_chunk_transcribed
        self._queue: queue.Queue[_Job | None] = queue.Queue()
        self._model: WhisperModel | None = None
        self._model_ready = threading.Event()
        self._workers: list[threading.Thread] = []
        self.language = language

        self._boot = threading.Thread(target=self._boot_and_serve, daemon=True, name="transcriber-boot")
        self._boot.start()

    @staticmethod
    def _validate_model_dir():
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Model directory not found:\n  {MODEL_DIR}\n\n"
                "Run:  python scripts/download_model.py"
            )
        existing = {f.name for f in MODEL_DIR.iterdir()}
        missing = _REQUIRED_FILES - existing
        if missing:
            raise FileNotFoundError(
                f"Model directory is incomplete. Missing: {missing}\n\n"
                "Re-run:  python scripts/download_model.py"
            )

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        return self._model_ready.wait(timeout=timeout)

    def submit(self, chunk_path: Path | str, start_offset: float = 0.0):
        self._queue.put(_Job(Path(chunk_path), float(start_offset)))
        print(
            f"[whisper] queued  {Path(chunk_path).name} | pending={self._queue.qsize()} | "
            f"on={threading.current_thread().name}",
            flush=True,
        )

    def finish(self):
        for _ in self._workers:
            self._queue.put(None)
        for w in self._workers:
            w.join()

    def _boot_and_serve(self):
        try:
            self._model = WhisperModel(
                str(MODEL_DIR),
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                cpu_threads=CPU_THREADS,
                num_workers=NUM_WORKERS,
            )
            logger.info(
                "Whisper model loaded from %s (workers=%d, cpu_threads=%d)",
                MODEL_DIR, NUM_WORKERS, CPU_THREADS,
            )
            print(f"[whisper] model loaded (cpu_threads={CPU_THREADS}, workers={NUM_WORKERS})", flush=True)
            self._warmup()
        except Exception:
            logger.exception("Failed to load Whisper model")
        finally:
            self._model_ready.set()

        if self._model is None:
            return

        for i in range(NUM_WORKERS):
            t = threading.Thread(
                target=self._worker_loop, daemon=True, name=f"transcriber-worker-{i}"
            )
            t.start()
            self._workers.append(t)

    def _warmup(self):
        try:
            t0 = time.monotonic()
            silent = np.zeros(16_000, dtype=np.float32)
            segments, _ = self._model.transcribe(
                silent, language=self.language, beam_size=1, temperature=0.0,
                vad_filter=False, without_timestamps=True,
            )
            for _ in segments:
                pass
            print(f"[whisper] warmup complete in {time.monotonic() - t0:.1f}s", flush=True)
        except Exception:
            logger.exception("Warmup failed (non-fatal)")

    def _worker_loop(self):
        while True:
            job = self._queue.get()
            if job is None:
                self._queue.task_done()
                break
            try:
                print(
                    f"[whisper] -> sending {job.path.name} to model… | "
                    f"on={threading.current_thread().name}",
                    flush=True,
                )
                t0 = time.monotonic()
                result = self._transcribe(job.path, job.start_offset)
                took = time.monotonic() - t0

                rtf = (took / result.duration) if result.duration else 0.0
                text = result.text.strip()
                preview = (text[:80] + "…") if len(text) > 80 else text
                shown = preview if text else "(no speech detected)"
                print(
                    f"[whisper] <- done    {job.path.name} | {took:4.1f}s for "
                    f"{result.duration:4.1f}s audio ({rtf:.2f}x) | {shown} | "
                    f"on={threading.current_thread().name}",
                    flush=True,
                )
                self._on_result(result)
            except Exception:
                logger.exception("Transcription failed for %s", job.path)
                print(f"[whisper] !! FAILED  {job.path.name}", flush=True)
            finally:
                self._queue.task_done()

    def _transcribe(self, path: Path, start_offset: float) -> TranscribedChunk:
        duration = self._wav_duration(path)
        word_timestamps = ENABLE_WORD_TIMESTAMPS and duration >= _WORD_TIMESTAMP_MIN_SECONDS

        audio_array = self._read_wav_to_array(path)

        segments, _ = self._model.transcribe(
            audio_array,
            language=self.language,
            task="transcribe",
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=word_timestamps,
            without_timestamps=not word_timestamps,
        )

        words: list[tuple[float, float, str]] = []
        text_parts: list[str] = []

        for segment in segments:
            text_parts.append(segment.text.strip())
            if segment.words:
                for w in segment.words:
                    words.append((w.start, w.end, w.word))

        return TranscribedChunk(
            chunk_path=path,
            text=" ".join(text_parts),
            words=words,
            start_offset=start_offset,
            duration=duration,
        )

    @classmethod
    def _read_wav_to_array(cls, path: Path) -> np.ndarray:
        try:
            with wave.open(str(path), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            return np.array([], dtype=np.float32)

    @staticmethod
    def _wav_duration(path: Path) -> float:
        try:
            with wave.open(str(path), "rb") as wf:
                return wf.getnframes() / float(wf.getframerate() or 16_000)
        except Exception:
            return 0.0