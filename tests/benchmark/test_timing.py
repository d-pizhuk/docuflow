import sys
import time
import wave
import json
import tempfile
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from settings import Settings
from ai.languages import WHISPER_LANGUAGE_CODES
from ai.transcriber import Transcriber
from ai.step_structurer import StepStructurer
from ai.api_gateway import ApiGateway, ApiGatewayError

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"


@pytest.fixture(scope="module")
def transcriber():
    settings = Settings.load()
    whisper_lang = WHISPER_LANGUAGE_CODES.get(settings.documentation_language, "en")
    t = Transcriber(on_chunk_transcribed=lambda _: None, language=whisper_lang)
    t.wait_until_ready(timeout=120)
    return t


def test_nf_req1_transcription_speed(transcriber):
    samples = [d for d in SAMPLES_DIR.iterdir() if d.is_dir()]
    longest_duration = 0
    longest_wav = None

    for s in samples:
        wav = s / "recording.wav"
        if wav.exists():
            with wave.open(str(wav), "rb") as wf:
                dur = wf.getnframes() / float(wf.getframerate())
                if dur > longest_duration:
                    longest_duration = dur
                    longest_wav = wav

    assert longest_wav is not None, "No recording.wav found in benchmark samples"

    with wave.open(str(longest_wav), "rb") as wf:
        framerate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        chunk_frames = int(15.0 * framerate)
        chunks = []
        i = 0

        while True:
            frames = wf.readframes(chunk_frames)
            if not frames:
                break
            tmp = Path(tempfile.gettempdir()) / f"docuflow_test_chunk_{i}.wav"
            with wave.open(str(tmp), "wb") as wf_out:
                wf_out.setnchannels(n_channels)
                wf_out.setsampwidth(sampwidth)
                wf_out.setframerate(framerate)
                wf_out.writeframes(frames)
            chunks.append(tmp)
            i += 1

    try:
        print(f"\nSimulating {len(chunks)} chunks ({len(chunks) * 15}s of audio) in real-time...")
        for i, chunk in enumerate(chunks):
            transcriber.submit(chunk, i * 15.0)
            if i < len(chunks) - 1:
                time.sleep(15.0)

        print("Recording stopped. Measuring post-stop transcription drain time...")
        t_stop = time.monotonic()
        transcriber.finish()
        t_finish = time.monotonic()

        post_stop_time = t_finish - t_stop
        print(f"Post-stop transcription time: {post_stop_time:.1f}s")

        assert post_stop_time <= 120.0, f"Post-stop wait {post_stop_time:.1f}s exceeds 120s limit"
    finally:
        for chunk in chunks:
            chunk.unlink(missing_ok=True)


def test_nf_req2_generation_speed():
    samples = [d for d in SAMPLES_DIR.iterdir() if d.is_dir()]
    max_steps = 0
    target_sample = None

    for s in samples:
        gt_path = s / "steps_ground_truth.json"
        if gt_path.exists():
            try:
                with open(gt_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    step_count = len(data.get("steps", []))
                    if step_count > max_steps:
                        max_steps = step_count
                        target_sample = s
            except Exception:
                continue

    assert target_sample is not None, "No valid steps_ground_truth.json found in benchmark samples"

    transcript_path = target_sample / "transcript_whisper.txt"
    assert transcript_path.exists(), f"Missing transcript in {target_sample.name}"

    print(f"\nUsing {target_sample.name} (requires generating {max_steps} steps)...")
    transcript_text = transcript_path.read_text(encoding="utf-8")

    settings = Settings.load()
    gateway = ApiGateway(base_url=settings.api_base_url, api_key=settings.api_key)
    structurer = StepStructurer(gateway=gateway, model=settings.llm_model, language=settings.documentation_language)

    t0 = time.monotonic()

    try:
        doc = structurer.structure(transcript_text, valid_screenshots=[], session_dir=target_sample)
    except ApiGatewayError as e:
        pytest.skip(f"LLM API unreachable, skipping generation timing test: {e}")

    elapsed = time.monotonic() - t0
    print(f"Generated {len(doc.steps)} steps in {elapsed:.1f}s")
    assert elapsed <= 90.0, f"Took {elapsed:.1f}s, limit is 90s"


if __name__ == "__main__":
    print("--- Running NfReq1 & NfReq2 Timing Tests ---")

    t = Transcriber(on_chunk_transcribed=lambda _: None, language="en")
    t.wait_until_ready(timeout=120)

    test_nf_req1_transcription_speed(t)
    print("NfReq1 passed!\n")

    test_nf_req2_generation_speed()
    print("NfReq2 passed!")