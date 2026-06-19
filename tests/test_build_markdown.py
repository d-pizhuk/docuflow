import json
import tempfile
import unittest
from pathlib import Path

from processing.build_markdown import (
    MarkdownBuildError,
    build_markdown_document,
    render_markdown,
    slugify,
)
from session.session_manifest import SessionManifest


TEST_TEMP_ROOT = Path(__file__).parent / ".tmp"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


class MarkdownRenderingTests(unittest.TestCase):
    def test_renders_multiple_screenshots_in_marker_order(self):
        documentation = {
            "title": "Send a message",
            "steps": [
                {
                    "major_step_number": 1,
                    "major_step_title": "Compose message",
                    "substeps": [
                        {
                            "substep_id": "1a",
                            "action": (
                                "Type a message. [Screenshot_1] "
                                "Then review it. [Screenshot_2] Send it."
                            ),
                        }
                    ],
                }
            ],
        }
        descriptions = {
            1: {
                "title": "Message field",
                "description": "The message is visible.",
            },
            2: {
                "title": "Review message",
                "description": "Review the text before sending.",
            },
        }
        image_paths = {
            1: "screenshots/screenshot_001.png",
            2: "screenshots/screenshot_002.png",
        }

        markdown = render_markdown(
            documentation,
            descriptions,
            image_paths,
        )

        self.assertNotIn("[Screenshot_", markdown)
        self.assertLess(
            markdown.index("Type a message."),
            markdown.index("screenshots/screenshot_001.png"),
        )
        self.assertLess(
            markdown.index("Then review it."),
            markdown.index("screenshots/screenshot_002.png"),
        )
        self.assertLess(
            markdown.index("screenshots/screenshot_002.png"),
            markdown.index("Send it."),
        )
        self.assertIn("[1. Compose message](#step-1-compose-message)", markdown)

    def test_slugifies_unicode_titles(self):
        self.assertEqual(
            slugify("Nachricht öffnen & prüfen"),
            "nachricht-offnen-prufen",
        )


class MarkdownStageTests(unittest.TestCase):
    def test_builds_portable_package_and_updates_manifest(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = Path(temporary_dir) / "session"
            audio_path = session_dir / "recording.wav"
            screenshot_path = (
                session_dir / "screenshots" / "screenshot_001.png"
            )
            documentation_path = (
                session_dir / "output" / "process_documentation.json"
            )
            descriptions_path = (
                session_dir / "output" / "screenshot_descriptions.json"
            )

            manifest = SessionManifest.create(
                session_dir,
                device_name="Test microphone",
                sample_rate=16_000,
                channels=1,
            )
            audio_path.write_bytes(b"audio")
            screenshot_path.write_bytes(b"image")
            manifest.add_screenshot(1, screenshot_path, 4.0)
            manifest.complete(audio_path, 8.0)

            documentation_path.write_text(
                json.dumps(
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
                    }
                ),
                encoding="utf-8",
            )
            descriptions_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "descriptions": [
                            {
                                "screenshot_id": 1,
                                "screenshot_path": (
                                    "screenshots/screenshot_001.png"
                                ),
                                "title": "Open chat",
                                "description": "The selected chat is visible.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest.start_processing_stage("process_documentation")
            manifest.complete_processing_stage(
                "process_documentation",
                documentation_path,
            )
            manifest.start_processing_stage("screenshot_descriptions")
            manifest.complete_processing_stage(
                "screenshot_descriptions",
                descriptions_path,
            )

            output_path = build_markdown_document(session_dir)

            markdown = output_path.read_text(encoding="utf-8")
            self.assertIn("# Send a message", markdown)
            self.assertIn(
                "![Open chat](screenshots/screenshot_001.png)",
                markdown,
            )
            self.assertNotIn("[Screenshot_1]", markdown)
            self.assertEqual(
                (
                    session_dir
                    / "output"
                    / "final"
                    / "screenshots"
                    / "screenshot_001.png"
                ).read_bytes(),
                b"image",
            )

            saved_manifest = SessionManifest.load(session_dir).data
            stage = saved_manifest["processing"]["markdown"]
            self.assertEqual(stage["status"], "completed")
            self.assertEqual(
                stage["output_path"],
                "output/final/final_documentation.md",
            )
            self.assertEqual(stage["screenshot_count"], 1)

    def test_rejects_missing_description(self):
        documentation = {
            "title": "Test",
            "steps": [
                {
                    "major_step_number": 1,
                    "major_step_title": "Test",
                    "substeps": [
                        {
                            "substep_id": "1a",
                            "action": "Do it. [Screenshot_1]",
                        }
                    ],
                }
            ],
        }

        with self.assertRaises(KeyError):
            render_markdown(documentation, {}, {})


if __name__ == "__main__":
    unittest.main()
