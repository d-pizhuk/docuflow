import json
import tempfile
import unittest
from pathlib import Path

from session.session_manifest import SessionManifest


TEST_TEMP_ROOT = Path(__file__).parent / ".tmp"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


class SessionManifestTests(unittest.TestCase):
    def test_create_builds_session_contract_and_directories(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = Path(temporary_dir) / "session"

            manifest = SessionManifest.create(
                session_dir,
                device_name="Test microphone",
                sample_rate=16_000,
                channels=1,
            )

            self.assertTrue((session_dir / "screenshots").is_dir())
            self.assertTrue((session_dir / "output").is_dir())
            self.assertTrue((session_dir / "session.json").is_file())
            self.assertEqual(manifest.data["version"], 1)
            self.assertEqual(manifest.data["status"], "recording")
            self.assertEqual(
                manifest.data["settings"]["output_language"],
                "German",
            )
            self.assertEqual(manifest.data["audio"]["path"], "recording.wav")
            self.assertEqual(
                manifest.data["audio"]["device_name"],
                "Test microphone",
            )

    def test_screenshots_and_completion_are_persisted(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = Path(temporary_dir) / "session"
            screenshot_path = session_dir / "screenshots" / "screenshot_001.png"
            audio_path = session_dir / "recording.wav"

            manifest = SessionManifest.create(
                session_dir,
                device_name="Test microphone",
                sample_rate=16_000,
                channels=1,
            )
            screenshot_path.write_bytes(b"image")
            audio_path.write_bytes(b"audio")

            manifest.add_screenshot(1, screenshot_path, 14.37291)
            manifest.complete(audio_path, 22.98761)
            manifest.start_processing_stage(
                "transcription",
                model="large-v3",
                device="cpu",
            )
            transcript_path = session_dir / "output" / "transcript.json"
            transcript_path.write_text("{}", encoding="utf-8")
            manifest.complete_processing_stage(
                "transcription",
                transcript_path,
                model="large-v3",
                device="cpu",
                segment_count=2,
            )

            data = json.loads(
                (session_dir / "session.json").read_text(encoding="utf-8")
            )
            self.assertEqual(data["status"], "completed")
            self.assertEqual(data["duration_seconds"], 22.988)
            self.assertEqual(
                data["screenshots"],
                [
                    {
                        "id": 1,
                        "path": "screenshots/screenshot_001.png",
                        "elapsed_seconds": 14.373,
                        "captured_at": data["screenshots"][0]["captured_at"],
                    }
                ],
            )

            loaded = SessionManifest.load(session_dir)
            self.assertEqual(loaded.next_screenshot_id, 2)
            self.assertEqual(
                loaded.resolve_artifact_path(data["audio"]["path"]),
                audio_path.resolve(),
            )
            self.assertEqual(
                data["processing"]["transcription"]["output_path"],
                "output/transcript.json",
            )

    def test_artifacts_outside_session_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            root = Path(temporary_dir)
            session_dir = root / "session"
            outside_path = root / "outside.png"

            manifest = SessionManifest.create(
                session_dir,
                device_name="Test microphone",
                sample_rate=16_000,
                channels=1,
            )

            with self.assertRaises(ValueError):
                manifest.add_screenshot(1, outside_path, 1.0)


if __name__ == "__main__":
    unittest.main()
