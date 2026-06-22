import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ai.transcriber import TranscribedChunk
from session.screenshot_capture import CapturedScreenshot


@dataclass
class AnnotatedStep:
    text: str
    start_sec: float
    end_sec: float
    screenshot: CapturedScreenshot | None = None


class TranscriptAssembler:
    def __init__(self, recording_start: datetime):
        self._recording_start = recording_start.replace(tzinfo=timezone.utc)
        self._chunks: list[TranscribedChunk] = []
        self._screenshots: list[CapturedScreenshot] = []
        self._lock = threading.Lock()

    def add_chunk(self, chunk: TranscribedChunk):
        with self._lock:
            self._chunks.append(chunk)

    def add_screenshot(self, screenshot: CapturedScreenshot):
        with self._lock:
            self._screenshots.append(screenshot)

    @staticmethod
    def _build_word_timeline(chunks: list[TranscribedChunk]) -> list[tuple[float, float, str]]:
        chunks = sorted(chunks, key=lambda c: c.start_offset)
        words: list[tuple[float, float, str]] = []

        for i, chunk in enumerate(chunks):
            if chunk.words:
                for start, end, word in chunk.words:
                    words.append((chunk.start_offset + start, chunk.start_offset + end, word))
                continue

            text = chunk.text.strip()
            if not text:
                continue

            span_start = chunk.start_offset
            span_end = chunk.start_offset + chunk.duration
            if span_end <= span_start:
                if i + 1 < len(chunks):
                    span_end = max(chunks[i + 1].start_offset, span_start)
                if span_end <= span_start:
                    span_end = span_start + 1.0

            toks = text.split()
            n = len(toks)
            step = (span_end - span_start) / n if n else 0.0
            for j, tok in enumerate(toks):
                ws = span_start + j * step
                we = span_start + (j + 1) * step
                words.append((ws, we, tok))

        return words

    def assemble(self) -> list[AnnotatedStep]:
        with self._lock:
            chunks = list(self._chunks)
            screenshots = list(self._screenshots)

        words = self._build_word_timeline(chunks)
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

        if pending_ss is None and current_ss is not None and steps:
            steps[-1].screenshot = current_ss[1]

        return steps

    def full_transcript(self) -> str:
        with self._lock:
            chunks = sorted(self._chunks, key=lambda c: c.start_offset)
        return " ".join(c.text for c in chunks).strip()

    def annotated_transcript(self) -> str:
        steps = self.assemble()
        if not steps:
            return self.full_transcript()
        parts: list[str] = []
        for step in steps:
            if step.screenshot is not None:
                parts.append(f"[SCREENSHOT: {step.screenshot.path.name}]")
            if step.text:
                parts.append(step.text)
        return "\n".join(parts)

    def screenshot_names(self) -> list[str]:
        with self._lock:
            return [s.path.name for s in self._screenshots]