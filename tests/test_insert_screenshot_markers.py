import json
import tempfile
import unittest
from pathlib import Path

from processing.insert_screenshot_markers import (
    ScreenshotMarkerError,
    build_marker_transcript,
    insert_screenshot_markers,
)
from session.session_manifest import SessionManifest


TEST_TEMP_ROOT = Path(__file__).parent / ".tmp"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


class MarkerPlacementTests(unittest.TestCase):
    def test_places_markers_before_after_and_between_segments(self):
        segments = [
            {"index": 0, "id": 10, "start": 0.0, "end": 10.0, "text": "First."},
            {"index": 1, "id": 11, "start": 20.0, "end": 30.0, "text": "Second."},
        ]
        screenshots = [
            {"id": 1, "elapsed_seconds": 1.0, "path": "one.png"},
            {"id": 2, "elapsed_seconds": 9.0, "path": "two.png"},
            {"id": 3, "elapsed_seconds": 15.0, "path": "three.png"},
            {"id": 4, "elapsed_seconds": 21.0, "path": "four.png"},
        ]

        text, assignments = build_marker_transcript(segments, screenshots)

        self.assertEqual(
            text,
            "[Screenshot_1] First. [Screenshot_2]\n"
            "[Screenshot_3]\n"
            "[Screenshot_4] Second.\n",
        )
        self.assertEqual(
            [item["placement"] for item in assignments],
            [
                "before_segment",
                "after_segment",
                "standalone",
                "before_segment",
            ],
        )
        for screenshot_id in range(1, 5):
            self.assertEqual(text.count(f"[Screenshot_{screenshot_id}]"), 1)


class MarkerStageTests(unittest.TestCase):
    def test_writes_text_report_and_manifest_status(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = Path(temporary_dir) / "session"
            audio_path = session_dir / "recording.wav"
            screenshot_path = session_dir / "screenshots" / "screenshot_001.png"
            transcript_path = session_dir / "output" / "transcript.json"

            manifest = SessionManifest.create(
                session_dir,
                device_name="Test microphone",
                sample_rate=16_000,
                channels=1,
            )
            audio_path.write_bytes(b"audio")
            screenshot_path.write_bytes(b"image")
            manifest.add_screenshot(1, screenshot_path, 8.5)
            manifest.complete(audio_path, 12.0)

            transcript_path.write_text(
                json.dumps(
                    {
                        "text": "Click the button.",
                        "segments": [
                            {
                                "id": 0,
                                "start": 0.0,
                                "end": 10.0,
                                "text": "Click the button.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
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

            output_path = insert_screenshot_markers(session_dir)

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "Click the button. [Screenshot_1]\n",
            )
            report = json.loads(
                (
                    session_dir
                    / "output"
                    / "screenshot_assignments.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                report["assignments"][0]["placement"],
                "after_segment",
            )

            saved_manifest = SessionManifest.load(session_dir).data
            stage = saved_manifest["processing"]["screenshot_markers"]
            self.assertEqual(stage["status"], "completed")
            self.assertEqual(
                stage["output_path"],
                "output/transcript_with_screenshots.txt",
            )
            self.assertEqual(stage["assignment_count"], 1)

    def test_requires_completed_transcription(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = Path(temporary_dir) / "session"
            SessionManifest.create(
                session_dir,
                device_name="Test microphone",
                sample_rate=16_000,
                channels=1,
            )

            with self.assertRaisesRegex(
                ScreenshotMarkerError,
                "transcription stage must be completed",
            ):
                insert_screenshot_markers(session_dir)


if __name__ == "__main__":
    unittest.main()
