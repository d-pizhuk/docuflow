import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.transcribe import TranscriptionError, transcribe_session
from session.session_manifest import SessionManifest


TEST_TEMP_ROOT = Path(__file__).parent / ".tmp"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


class TranscriptionTests(unittest.TestCase):
    def test_rejects_an_incomplete_session_before_loading_whisper(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = Path(temporary_dir) / "session"
            SessionManifest.create(
                session_dir,
                device_name="Test microphone",
                sample_rate=16_000,
                channels=1,
            )

            with self.assertRaisesRegex(TranscriptionError, "must be completed"):
                transcribe_session(session_dir)

    def test_returns_existing_completed_transcript_without_loading_whisper(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = Path(temporary_dir) / "session"
            audio_path = session_dir / "recording.wav"
            transcript_path = session_dir / "output" / "transcript.json"

            manifest = SessionManifest.create(
                session_dir,
                device_name="Test microphone",
                sample_rate=16_000,
                channels=1,
            )
            audio_path.write_bytes(b"audio")
            transcript_path.write_text("{}", encoding="utf-8")
            manifest.complete(audio_path, 2.0)
            manifest.start_processing_stage(
                "transcription",
                model="large-v3",
                device="cpu",
            )
            manifest.complete_processing_stage(
                "transcription",
                transcript_path,
                model="large-v3",
                device="cpu",
            )

            with patch.dict("sys.modules", {"whisper": None, "torch": None}):
                result = transcribe_session(session_dir)

            self.assertEqual(result, transcript_path)


if __name__ == "__main__":
    unittest.main()
