import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from processing.render_html import (
    HtmlRenderError,
    build_html_page,
    render_html_document,
    validate_markdown_images,
)
from session.session_manifest import SessionManifest


TEST_TEMP_ROOT = Path(__file__).parent / ".tmp"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


class HtmlPageTests(unittest.TestCase):
    def test_builds_dynamic_portable_page(self):
        page = build_html_page(
            title="Open <Chat>",
            html_body="<h2>Step</h2>",
            generated_at=datetime(2026, 6, 19, tzinfo=timezone.utc),
            output_language="English",
        )

        self.assertIn("<html lang=\"en\">", page)
        self.assertIn("Open &lt;Chat&gt;", page)
        self.assertIn("Generated: 2026-06-19", page)
        self.assertIn("<h2>Step</h2>", page)
        self.assertIn("Back to top", page)
        self.assertNotIn("Erstellen eines neuen Discord-Servers", page)
        self.assertNotIn("https://", page)

    def test_validates_local_images(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            root = Path(temporary_dir)
            screenshot = root / "screenshots" / "screenshot_001.png"
            screenshot.parent.mkdir()
            screenshot.write_bytes(b"image")

            count = validate_markdown_images(
                "![Screenshot](screenshots/screenshot_001.png)",
                root,
            )
            self.assertEqual(count, 1)

            with self.assertRaisesRegex(
                HtmlRenderError,
                "escapes the final package",
            ):
                validate_markdown_images("![Bad](../secret.png)", root)


class HtmlStageTests(unittest.TestCase):
    def test_renders_html_and_updates_manifest(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = Path(temporary_dir) / "session"
            audio_path = session_dir / "recording.wav"
            documentation_path = (
                session_dir / "output" / "process_documentation.json"
            )
            markdown_path = (
                session_dir
                / "output"
                / "final"
                / "final_documentation.md"
            )
            screenshot_path = (
                session_dir
                / "output"
                / "final"
                / "screenshots"
                / "screenshot_001.png"
            )

            manifest = SessionManifest.create(
                session_dir,
                device_name="Test microphone",
                sample_rate=16_000,
                channels=1,
            )
            audio_path.write_bytes(b"audio")
            manifest.complete(audio_path, 5.0)

            documentation_path.write_text(
                json.dumps(
                    {
                        "title": "Send a message",
                        "steps": [],
                    }
                ),
                encoding="utf-8",
            )
            screenshot_path.parent.mkdir(parents=True)
            screenshot_path.write_bytes(b"image")
            markdown_path.write_text(
                "# Send a message\n\n"
                "## Contents\n\n"
                "![Screenshot](screenshots/screenshot_001.png)\n",
                encoding="utf-8",
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
            manifest.start_processing_stage("markdown")
            manifest.complete_processing_stage("markdown", markdown_path)

            output_path = render_html_document(session_dir)

            html_text = output_path.read_text(encoding="utf-8")
            self.assertIn("<title>Send a message · DocuFlow</title>", html_text)
            self.assertIn(
                'src="screenshots/screenshot_001.png"',
                html_text,
            )
            self.assertIn("DocuFlow process documentation", html_text)

            saved_manifest = SessionManifest.load(session_dir).data
            stage = saved_manifest["processing"]["html"]
            self.assertEqual(stage["status"], "completed")
            self.assertEqual(
                stage["output_path"],
                "output/final/process_documentation.html",
            )
            self.assertEqual(stage["screenshot_count"], 1)

    def test_requires_completed_markdown_stage(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = Path(temporary_dir) / "session"
            SessionManifest.create(
                session_dir,
                device_name="Test microphone",
                sample_rate=16_000,
                channels=1,
            )

            with self.assertRaisesRegex(
                HtmlRenderError,
                "Markdown stage must be completed",
            ):
                render_html_document(session_dir)


if __name__ == "__main__":
    unittest.main()
