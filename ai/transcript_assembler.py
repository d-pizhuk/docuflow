import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ai.transcriber import TranscribedChunk
from session.screenshot_capture import CapturedScreenshot


@dataclass
class AnnotatedStep:
    """A transcript segment optionally paired with a screenshot."""
    text: str
    start_sec: float
    end_sec: float
    screenshot: CapturedScreenshot | None = None


class TranscriptAssembler:
    """
    Merges incoming TranscribedChunks with CapturedScreenshots into an
    ordered list of AnnotatedSteps aligned by timestamp.

    Thread-safe: chunks and screenshots may arrive from different threads.
    """

    def __init__(self, recording_start: datetime):
        self._recording_start = recording_start.replace(tzinfo=timezone.utc)
        self._chunks: list[TranscribedChunk] = []
        self._screenshots: list[CapturedScreenshot] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #

    def add_chunk(self, chunk: TranscribedChunk):
        with self._lock:
            self._chunks.append(chunk)

    def add_screenshot(self, screenshot: CapturedScreenshot):
        with self._lock:
            self._screenshots.append(screenshot)

    # ------------------------------------------------------------------ #
    # Assembly
    # ------------------------------------------------------------------ #

    def assemble(self) -> list[AnnotatedStep]:
        """
        Call once recording and all transcription are complete.
        Returns steps in chronological order; each step carries the nearest
        screenshot whose timestamp falls at or before the step boundary.
        """
        with self._lock:
            chunks = list(self._chunks)
            screenshots = list(self._screenshots)

        words: list[tuple[float, float, str]] = []
        for chunk in sorted(chunks, key=lambda c: c.start_offset):
            for start, end, word in chunk.words:
                words.append((chunk.start_offset + start, chunk.start_offset + end, word))

        if not words:
            return []

        ss_offsets: list[tuple[float, CapturedScreenshot]] = []
        for ss in screenshots:
            delta = (ss.timestamp - self._recording_start).total_seconds()
            ss_offsets.append((delta, ss))
        ss_offsets.sort(key=lambda x: x[0])

        steps: list[AnnotatedStep] = []
        ss_iter = iter(ss_offsets)
        current_ss: tuple[float, CapturedScreenshot] | None = next(ss_iter, None)
        pending_ss: CapturedScreenshot | None = None
        segment_words: list[str] = []
        seg_start = words[0][0]
        seg_end = words[0][1]

        for w_start, w_end, word in words:
            while current_ss and current_ss[0] <= w_start:
                if segment_words:
                    steps.append(AnnotatedStep(
                        text=" ".join(segment_words).strip(),
                        start_sec=seg_start,
                        end_sec=seg_end,
                        screenshot=pending_ss,
                    ))
                    segment_words = []
                    seg_start = w_start
                pending_ss = current_ss[1]
                current_ss = next(ss_iter, None)

            segment_words.append(word)
            seg_end = w_end

        if segment_words:
            steps.append(AnnotatedStep(
                text=" ".join(segment_words).strip(),
                start_sec=seg_start,
                end_sec=seg_end,
                screenshot=pending_ss,
            ))

        return steps

    def full_transcript(self) -> str:
        """Plain concatenated transcript — used as LLM input or manual fallback."""
        with self._lock:
            chunks = sorted(self._chunks, key=lambda c: c.start_offset)
        return " ".join(c.text for c in chunks).strip()