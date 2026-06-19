import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from processing.describe_screenshots import (
    ScreenshotDescriptionError,
    build_screenshot_contexts,
    describe_screenshots,
    extract_transcript_context,
)
from session.session_manifest import SessionManifest


TEST_TEMP_ROOT = Path(__file__).parent / ".tmp"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


class FakeResponses:
    def __init__(self):
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        parsed = kwargs["text_format"].model_validate(
            {
                "title": "Message field",
                "description": (
                    "The message field is visible. Enter the message here."
                ),
            }
        )
        return SimpleNamespace(
            id=f"resp_{len(self.calls)}",
            output_parsed=parsed,
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=25,
                total_tokens=125,
            ),
        )


class FakeClient:
    def __init__(self):
        self.responses = FakeResponses()


class ScreenshotContextTests(unittest.TestCase):
    def test_extracts_nearby_transcript_lines(self):
        transcript = "\n".join(
            [
                "Line zero.",
                "Line one.",
                "Open the chat. [Screenshot_1]",
                "Type the message.",
                "Press Enter.",
            ]
        )

        context = extract_transcript_context(
            transcript,
            "[Screenshot_1]",
            window=1,
        )

        self.assertEqual(
            context,
            "Line one. Open the chat. [Screenshot_1] Type the message.",
        )

    def test_builds_context_from_documentation_and_manifest(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = Path(temporary_dir) / "session"
            screenshot_path = (
                session_dir / "screenshots" / "screenshot_001.png"
            )
            manifest = SessionManifest.create(
                session_dir,
                device_name="Test microphone",
                sample_rate=16_000,
                channels=1,
            )
            screenshot_path.write_bytes(b"image")
            manifest.add_screenshot(1, screenshot_path, 4.5)

            contexts = build_screenshot_contexts(
                manifest,
                manifest.data["screenshots"],
                {
                    "title": "Send a message",
                    "steps": [
                        {
                            "major_step_number": 1,
                            "major_step_title": "Open chat",
                            "substeps": [
                                {
                                    "substep_id": "1a",
                                    "action": (
                                        "Open the chat. [Screenshot_1]"
                                    ),
                                }
                            ],
                        }
                    ],
                },
                "Open the chat. [Screenshot_1]\n",
            )

            self.assertEqual(contexts[0]["screenshot_id"], 1)
            self.assertEqual(contexts[0]["substep_id"], "1a")
            self.assertEqual(
                contexts[0]["absolute_path"],
                screenshot_path.resolve(),
            )

    def test_rejects_a_marker_missing_from_documentation(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = Path(temporary_dir) / "session"
            screenshot_path = (
                session_dir / "screenshots" / "screenshot_001.png"
            )
            manifest = SessionManifest.create(
                session_dir,
                device_name="Test microphone",
                sample_rate=16_000,
                channels=1,
            )
            screenshot_path.write_bytes(b"image")
            manifest.add_screenshot(1, screenshot_path, 4.5)

            with self.assertRaisesRegex(
                ScreenshotDescriptionError,
                "missing documentation markers",
            ):
                build_screenshot_contexts(
                    manifest,
                    manifest.data["screenshots"],
                    {"title": "Test", "steps": []},
                    "[Screenshot_1]\n",
                )


class ScreenshotDescriptionStageTests(unittest.TestCase):
    def test_generates_individual_and_index_descriptions(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = Path(temporary_dir) / "session"
            audio_path = session_dir / "recording.wav"
            screenshot_path = (
                session_dir / "screenshots" / "screenshot_001.png"
            )
            transcript_path = (
                session_dir
                / "output"
                / "transcript_with_screenshots.txt"
            )
            documentation_path = (
                session_dir / "output" / "process_documentation.json"
            )

            manifest = SessionManifest.create(
                session_dir,
                device_name="Test microphone",
                sample_rate=16_000,
                channels=1,
            )
            audio_path.write_bytes(b"audio")
            screenshot_path.write_bytes(b"image")
            manifest.add_screenshot(1, screenshot_path, 4.5)
            manifest.complete(audio_path, 8.0)

            transcript_path.write_text(
                "Open Discord. [Screenshot_1]\n",
                encoding="utf-8",
            )
            documentation_path.write_text(
                json.dumps(
                    {
                        "title": "Send a message",
                        "steps": [
                            {
                                "major_step_number": 1,
                                "major_step_title": "Open Discord",
                                "substeps": [
                                    {
                                        "substep_id": "1a",
                                        "action": (
                                            "Open Discord. [Screenshot_1]"
                                        ),
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest.start_processing_stage("screenshot_markers")
            manifest.complete_processing_stage(
                "screenshot_markers",
                transcript_path,
            )
            manifest.start_processing_stage(
                "process_documentation",
                output_language="English",
            )
            manifest.complete_processing_stage(
                "process_documentation",
                documentation_path,
                output_language="English",
            )

            fake_client = FakeClient()
            index_path = describe_screenshots(
                session_dir,
                client=fake_client,
            )

            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            description = index_data["descriptions"][0]
            self.assertEqual(index_data["output_language"], "English")
            self.assertEqual(description["screenshot_id"], 1)
            self.assertEqual(description["substep_id"], "1a")
            self.assertEqual(description["title"], "Message field")
            self.assertEqual(description["model"], "gpt-4o")
            self.assertEqual(description["output_language"], "English")
            self.assertTrue(
                (
                    session_dir
                    / "output"
                    / "descriptions"
                    / "screenshot_001.json"
                ).exists()
            )

            call = fake_client.responses.calls[0]
            self.assertEqual(call["model"], "gpt-4o")
            self.assertEqual(
                call["input"][0]["content"][1]["type"],
                "input_image",
            )
            self.assertTrue(
                call["input"][0]["content"][1]["image_url"].startswith(
                    "data:image/png;base64,"
                )
            )

            saved_manifest = SessionManifest.load(session_dir).data
            stage = saved_manifest["processing"]["screenshot_descriptions"]
            self.assertEqual(stage["status"], "completed")
            self.assertEqual(
                stage["output_path"],
                "output/screenshot_descriptions.json",
            )
            self.assertEqual(stage["screenshot_count"], 1)
            self.assertEqual(stage["usage"]["total_tokens"], 125)

    def test_requires_completed_documentation_stage(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = Path(temporary_dir) / "session"
            SessionManifest.create(
                session_dir,
                device_name="Test microphone",
                sample_rate=16_000,
                channels=1,
            )

            with self.assertRaisesRegex(
                ScreenshotDescriptionError,
                "process documentation stage must be completed",
            ):
                describe_screenshots(session_dir)


if __name__ == "__main__":
    unittest.main()
