import logging
import os
import queue
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "whisper-large-v3-turbo"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "whisper-distil-large-v3"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# Up to 2 transcription workers run in parallel. They share ONE WhisperModel
# created with num_workers=NUM_WORKERS: CTranslate2 holds that many compute
# replicas (weights are shared, so no extra model RAM) and releases the GIL
# during inference, so two chunks can be transcribed at the same time. On CPU
# the two replicas still share the cores, so this mainly prevents a long chunk
# from blocking a short one behind it, rather than giving a literal 2x.
NUM_WORKERS = 2

# Split the available cores across the workers to avoid oversubscription
# (more threads than cores tends to slow CTranslate2 down rather than speed up).
CPU_THREADS = max(1, (os.cpu_count() or 4) // NUM_WORKERS)

_REQUIRED_FILES = {"model.bin", "config.json", "tokenizer.json"}

# Word-timestamp alignment (needed for fine screenshot placement) is only worth
# its cost on chunks long enough to likely contain a screenshot trigger. Short
# chunks are transcribed with word_timestamps=False to keep the post-Stop path
# fast; the assembler falls back to spreading their text across the chunk's own
# [start_offset, start_offset + duration] span, so no text is ever lost and
# screenshots still align within the chunk.
_WORD_TIMESTAMP_MIN_SECONDS = 10.0


@dataclass
class TranscribedChunk:
    chunk_path: Path
    text: str
    words: list[tuple[float, float, str]] = field(default_factory=list)
    start_offset: float = 0.0
    # Chunk length in seconds. Lets the assembler position a chunk's text on the
    # timeline even when word_timestamps were skipped (words == []).
    duration: float = 0.0


@dataclass
class _Job:
    path: Path
    start_offset: float


class Transcriber:
    """
    Loads faster-whisper from models/whisper-distil-large-v3/ and transcribes
    WAV chunks as they arrive via submit().

    A single shared model is served by NUM_WORKERS worker threads, so chunks may
    be transcribed concurrently and finish out of order. Ordering is preserved
    downstream because each chunk carries its own start_offset (assigned by the
    recorder), so transcription itself is stateless and order-independent.

    Run scripts/download_model.py once after cloning to populate the model dir.
    """

    def __init__(self, on_chunk_transcribed: Callable[[TranscribedChunk], None]):
        self._validate_model_dir()
        self._on_result = on_chunk_transcribed
        self._queue: queue.Queue[_Job | None] = queue.Queue()
        self._model: WhisperModel | None = None
        self._model_ready = threading.Event()
        self._workers: list[threading.Thread] = []
        # One boot thread loads the model, then spawns the worker pool.
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
        """Signal all workers to stop and block until the queue is drained."""
        # The model is already loaded by session start (DeviceSetupDialog calls
        # wait_until_ready), so there's no need to join the boot thread here.
        for _ in self._workers:
            self._queue.put(None)               # one sentinel per worker
        for w in self._workers:
            w.join()

    # ------------------------------------------------------------------ #
    # Boot + worker pool
    # ------------------------------------------------------------------ #

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

                text = result.text.strip()
                preview = (text[:80] + "…") if len(text) > 80 else text
                shown = preview if text else "(no speech detected)"
                print(
                    f"[whisper] <- done    {job.path.name} | {took:4.1f}s | "
                    f"{len(result.words):3d} words | {shown} | "
                    f"on={threading.current_thread().name}",
                    flush=True,
                )
                self._on_result(result)
            except Exception:
                logger.exception("Transcription failed for %s", job.path)
                print(f"[whisper] !! FAILED  {job.path.name}", flush=True)
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------ #
    # Inference (stateless: start_offset comes from the caller)
    # ------------------------------------------------------------------ #

    def _transcribe(self, path: Path, start_offset: float) -> TranscribedChunk:
        # Silence is already trimmed by the recorder's RMS-based VAD, so we run
        # Whisper's internal Silero VAD off — it's a fixed per-chunk CPU cost we
        # don't need on the (post-Stop) critical path.
        #
        # Word timestamps are needed only for fine screenshot alignment. Short
        # chunks are transcribed without the alignment pass; the assembler still
        # positions their text using start_offset + duration, so nothing is lost.
        duration = self._wav_duration(path)
        word_timestamps = duration >= _WORD_TIMESTAMP_MIN_SECONDS

        segments, _ = self._model.transcribe(
            str(path),
            language="en",
            word_timestamps=word_timestamps,
            vad_filter=False,
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

    @staticmethod
    def _wav_duration(path: Path) -> float:
        """Cheap O(1) duration lookup from the WAV header."""
        try:
            with wave.open(str(path), "rb") as wf:
                return wf.getnframes() / float(wf.getframerate() or 16_000)
        except Exception:
            return 0.0