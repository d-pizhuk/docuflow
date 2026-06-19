import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from processing.generate_process_documentation import (
    ProcessDocumentationError,
    _build_response_schema,
    generate_process_documentation,
    normalize_document_numbering,
    validate_document,
)
from session.session_manifest import SessionManifest


TEST_TEMP_ROOT = Path(__file__).parent / ".tmp"
TEST_TEMP_ROOT.mkdir(exist_ok=True)


class FakeResponses:
    def __init__(self, document):
        self.document = document
        self.call = None

    def parse(self, **kwargs):
        self.call = kwargs
        parsed = kwargs["text_format"].model_validate(self.document)
        return SimpleNamespace(
            id="resp_test",
            output_parsed=parsed,
            usage=SimpleNamespace(
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
            ),
        )


class FakeClient:
    def __init__(self, document):
        self.responses = FakeResponses(document)


class DocumentValidationTests(unittest.TestCase):
    def test_normalizes_numbering_and_preserves_markers(self):
        document = {
            "title": "Send a Discord message",
            "steps": [
                {
                    "major_step_number": 8,
                    "major_step_title": "Open the conversation",
                    "substeps": [
                        {
                            "substep_id": "wrong",
                            "action": "Open the conversation. [Screenshot_1]",
                        },
                        {
                            "substep_id": "wrong-again",
                            "action": "Type the message. [Screenshot_2]",
                        },
                    ],
                }
            ],
        }

        normalized = normalize_document_numbering(document)
        validate_document(
            normalized,
            ["[Screenshot_1]", "[Screenshot_2]"],
        )

        self.assertEqual(normalized["steps"][0]["major_step_number"], 1)
        self.assertEqual(
            [
                substep["substep_id"]
                for substep in normalized["steps"][0]["substeps"]
            ],
            ["1a", "1b"],
        )

    def test_rejects_missing_or_invented_screenshot_markers(self):
        document = {
            "title": "Send a message",
            "steps": [
                {
                    "major_step_number": 1,
                    "major_step_title": "Send",
                    "substeps": [
                        {
                            "substep_id": "1a",
                            "action": "Send it. [Screenshot_99]",
                        }
                    ],
                }
            ],
        }

        with self.assertRaisesRegex(
            ProcessDocumentationError,
            "Screenshot markers were not preserved exactly",
        ):
            validate_document(document, ["[Screenshot_1]"])


class DocumentationStageTests(unittest.TestCase):
    def test_generates_validated_json_and_updates_manifest(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = Path(temporary_dir) / "session"
            audio_path = session_dir / "recording.wav"
            transcript_path = (
                session_dir
                / "output"
                / "transcript_with_screenshots.txt"
            )

            manifest = SessionManifest.create(
                session_dir,
                device_name="Test microphone",
                sample_rate=16_000,
                channels=1,
            )
            audio_path.write_bytes(b"audio")
            manifest.complete(audio_path, 10.0)
            transcript_path.write_text(
                "Open Discord. [Screenshot_1]\n",
                encoding="utf-8",
            )
            manifest.start_processing_stage("screenshot_markers")
            manifest.complete_processing_stage(
                "screenshot_markers",
                transcript_path,
                screenshot_count=1,
            )

            fake_client = FakeClient(
                {
                    "title": "Send a Discord message",
                    "steps": [
                        {
                            "major_step_number": 4,
                            "major_step_title": "Open Discord",
                            "substeps": [
                                {
                                    "substep_id": "4z",
                                    "action": (
                                        "Open Discord. [Screenshot_1]"
                                    ),
                                }
                            ],
                        }
                    ],
                }
            )

            output_path = generate_process_documentation(
                session_dir,
                output_language="English",
                client=fake_client,
            )

            document = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(
                document["steps"][0]["major_step_number"],
                1,
            )
            self.assertEqual(
                document["steps"][0]["substeps"][0]["substep_id"],
                "1a",
            )
            self.assertEqual(
                fake_client.responses.call["model"],
                "gpt-4o",
            )

            saved_manifest = SessionManifest.load(session_dir).data
            stage = saved_manifest["processing"]["process_documentation"]
            self.assertEqual(stage["status"], "completed")
            self.assertEqual(
                stage["output_path"],
                "output/process_documentation.json",
            )
            self.assertEqual(stage["response_id"], "resp_test")
            self.assertEqual(stage["screenshot_marker_count"], 1)

    def test_requires_completed_marker_stage(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = Path(temporary_dir) / "session"
            SessionManifest.create(
                session_dir,
                device_name="Test microphone",
                sample_rate=16_000,
                channels=1,
            )

            with self.assertRaisesRegex(
                ProcessDocumentationError,
                "screenshot marker stage must be completed",
            ):
                generate_process_documentation(session_dir)


if __name__ == "__main__":
    unittest.main()
