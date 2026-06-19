import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from processing.pipeline import PipelineError, run_pipeline
from session.session_manifest import SessionManifest


TEST_TEMP_ROOT = Path(__file__).parent / ".tmp"
TEST_TEMP_ROOT.mkdir(exist_ok=True)

STAGE_OUTPUTS = {
    "transcription": Path("output") / "transcript.json",
    "screenshot_markers": (
        Path("output") / "transcript_with_screenshots.txt"
    ),
    "process_documentation": (
        Path("output") / "process_documentation.json"
    ),
    "screenshot_descriptions": (
        Path("output") / "screenshot_descriptions.json"
    ),
    "markdown": (
        Path("output") / "final" / "final_documentation.md"
    ),
    "html": (
        Path("output") / "final" / "process_documentation.html"
    ),
}


def _create_completed_session(
    root: Path,
    *,
    output_language: str = "German",
) -> Path:
    session_dir = root / "session"
    audio_path = session_dir / "recording.wav"
    manifest = SessionManifest.create(
        session_dir,
        device_name="Test microphone",
        sample_rate=16_000,
        channels=1,
        output_language=output_language,
    )
    audio_path.write_bytes(b"audio")
    manifest.complete(audio_path, 5.0)
    return session_dir


def _runner_for(session_dir: Path, stage_key: str, calls: list):
    def runner(*args, **kwargs):
        calls.append((stage_key, kwargs))
        output_path = session_dir / STAGE_OUTPUTS[stage_key]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(stage_key, encoding="utf-8")
        if stage_key == "screenshot_markers":
            (
                session_dir / "output" / "screenshot_assignments.json"
            ).write_text("{}", encoding="utf-8")
        manifest = SessionManifest.load(session_dir)
        manifest.start_processing_stage(stage_key, **_stage_config(stage_key))
        manifest.complete_processing_stage(
            stage_key,
            output_path,
            **_stage_config(stage_key),
        )
        return output_path

    return runner


def _stage_config(stage_key: str) -> dict:
    if stage_key == "transcription":
        return {
            "model": "large-v3",
            "device": "cuda",
            "requested_language": "auto",
            "language": "en",
        }
    if stage_key == "process_documentation":
        return {"model": "gpt-4o", "output_language": "English"}
    if stage_key == "screenshot_descriptions":
        return {
            "model": "gpt-4o",
            "output_language": "English",
            "context_window": 2,
        }
    return {}


def _mark_stage_completed(session_dir: Path, stage_key: str) -> None:
    output_path = session_dir / STAGE_OUTPUTS[stage_key]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(stage_key, encoding="utf-8")
    if stage_key == "screenshot_markers":
        (
            session_dir / "output" / "screenshot_assignments.json"
        ).write_text("{}", encoding="utf-8")
    manifest = SessionManifest.load(session_dir)
    manifest.start_processing_stage(stage_key, **_stage_config(stage_key))
    manifest.complete_processing_stage(
        stage_key,
        output_path,
        **_stage_config(stage_key),
    )


class PipelineTests(unittest.TestCase):
    def test_inherits_output_language_from_session_settings(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = _create_completed_session(
                Path(temporary_dir),
                output_language="French",
            )
            calls = []

            patches = [
                patch(
                    "processing.pipeline.transcribe_session",
                    side_effect=_runner_for(
                        session_dir,
                        "transcription",
                        calls,
                    ),
                ),
                patch(
                    "processing.pipeline.insert_screenshot_markers",
                    side_effect=_runner_for(
                        session_dir,
                        "screenshot_markers",
                        calls,
                    ),
                ),
                patch(
                    "processing.pipeline.generate_process_documentation",
                    side_effect=_runner_for(
                        session_dir,
                        "process_documentation",
                        calls,
                    ),
                ),
                patch(
                    "processing.pipeline.describe_screenshots",
                    side_effect=_runner_for(
                        session_dir,
                        "screenshot_descriptions",
                        calls,
                    ),
                ),
                patch(
                    "processing.pipeline.build_markdown_document",
                    side_effect=_runner_for(
                        session_dir,
                        "markdown",
                        calls,
                    ),
                ),
                patch(
                    "processing.pipeline.render_html_document",
                    side_effect=_runner_for(session_dir, "html", calls),
                ),
            ]
            with (
                patches[0],
                patches[1],
                patches[2],
                patches[3],
                patches[4],
                patches[5],
            ):
                run_pipeline(session_dir)

            process_call = next(
                kwargs
                for stage_key, kwargs in calls
                if stage_key == "process_documentation"
            )
            description_call = next(
                kwargs
                for stage_key, kwargs in calls
                if stage_key == "screenshot_descriptions"
            )
            self.assertEqual(process_call["output_language"], "French")
            self.assertEqual(description_call["output_language"], "French")

    def test_runs_all_stages_and_reports_progress(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = _create_completed_session(Path(temporary_dir))
            calls = []
            progress = []

            with (
                patch(
                    "processing.pipeline.transcribe_session",
                    side_effect=_runner_for(
                        session_dir,
                        "transcription",
                        calls,
                    ),
                ),
                patch(
                    "processing.pipeline.insert_screenshot_markers",
                    side_effect=_runner_for(
                        session_dir,
                        "screenshot_markers",
                        calls,
                    ),
                ),
                patch(
                    "processing.pipeline.generate_process_documentation",
                    side_effect=_runner_for(
                        session_dir,
                        "process_documentation",
                        calls,
                    ),
                ),
                patch(
                    "processing.pipeline.describe_screenshots",
                    side_effect=_runner_for(
                        session_dir,
                        "screenshot_descriptions",
                        calls,
                    ),
                ),
                patch(
                    "processing.pipeline.build_markdown_document",
                    side_effect=_runner_for(
                        session_dir,
                        "markdown",
                        calls,
                    ),
                ),
                patch(
                    "processing.pipeline.render_html_document",
                    side_effect=_runner_for(session_dir, "html", calls),
                ),
            ):
                result = run_pipeline(
                    session_dir,
                    output_language="English",
                    progress_callback=progress.append,
                )

            self.assertEqual(
                [stage_key for stage_key, _ in calls],
                list(STAGE_OUTPUTS),
            )
            self.assertEqual(
                result.executed_stages,
                tuple(STAGE_OUTPUTS),
            )
            self.assertEqual(result.reused_stages, ())
            self.assertEqual(
                [item.status for item in progress],
                ["starting", "completed"] * 6,
            )
            self.assertEqual(
                result.final_output_path,
                session_dir / STAGE_OUTPUTS["html"],
            )

            pipeline_status = SessionManifest.load(session_dir).data[
                "processing"
            ]["pipeline"]
            self.assertEqual(pipeline_status["status"], "completed")
            self.assertEqual(
                pipeline_status["executed_stages"],
                list(STAGE_OUTPUTS),
            )

    def test_reuses_all_compatible_completed_stages(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = _create_completed_session(Path(temporary_dir))
            for stage_key in STAGE_OUTPUTS:
                _mark_stage_completed(session_dir, stage_key)

            progress = []
            with (
                patch(
                    "processing.pipeline.transcribe_session",
                    return_value=session_dir / STAGE_OUTPUTS["transcription"],
                ),
                patch(
                    "processing.pipeline.insert_screenshot_markers",
                    return_value=(
                        session_dir / STAGE_OUTPUTS["screenshot_markers"]
                    ),
                ),
                patch(
                    "processing.pipeline.generate_process_documentation",
                    return_value=(
                        session_dir / STAGE_OUTPUTS["process_documentation"]
                    ),
                ),
                patch(
                    "processing.pipeline.describe_screenshots",
                    return_value=(
                        session_dir / STAGE_OUTPUTS["screenshot_descriptions"]
                    ),
                ),
                patch(
                    "processing.pipeline.build_markdown_document",
                    return_value=session_dir / STAGE_OUTPUTS["markdown"],
                ),
                patch(
                    "processing.pipeline.render_html_document",
                    return_value=session_dir / STAGE_OUTPUTS["html"],
                ),
            ):
                result = run_pipeline(
                    session_dir,
                    output_language="English",
                    progress_callback=progress.append,
                )

            self.assertEqual(result.executed_stages, ())
            self.assertEqual(result.reused_stages, tuple(STAGE_OUTPUTS))
            self.assertEqual(
                [item.status for item in progress if item.status != "starting"],
                ["reused"] * 6,
            )

    def test_language_change_restarts_documentation_and_downstream(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = _create_completed_session(Path(temporary_dir))
            for stage_key in STAGE_OUTPUTS:
                _mark_stage_completed(session_dir, stage_key)

            calls = []
            patches = [
                patch(
                    "processing.pipeline.transcribe_session",
                    side_effect=_runner_for(
                        session_dir,
                        "transcription",
                        calls,
                    ),
                ),
                patch(
                    "processing.pipeline.insert_screenshot_markers",
                    side_effect=_runner_for(
                        session_dir,
                        "screenshot_markers",
                        calls,
                    ),
                ),
                patch(
                    "processing.pipeline.generate_process_documentation",
                    side_effect=_runner_for(
                        session_dir,
                        "process_documentation",
                        calls,
                    ),
                ),
                patch(
                    "processing.pipeline.describe_screenshots",
                    side_effect=_runner_for(
                        session_dir,
                        "screenshot_descriptions",
                        calls,
                    ),
                ),
                patch(
                    "processing.pipeline.build_markdown_document",
                    side_effect=_runner_for(
                        session_dir,
                        "markdown",
                        calls,
                    ),
                ),
                patch(
                    "processing.pipeline.render_html_document",
                    side_effect=_runner_for(session_dir, "html", calls),
                ),
            ]
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
                result = run_pipeline(
                    session_dir,
                    output_language="German",
                )

            self.assertEqual(
                result.reused_stages,
                ("transcription", "screenshot_markers"),
            )
            self.assertEqual(
                result.executed_stages,
                (
                    "process_documentation",
                    "screenshot_descriptions",
                    "markdown",
                    "html",
                ),
            )
            process_call = next(
                kwargs
                for stage_key, kwargs in calls
                if stage_key == "process_documentation"
            )
            self.assertTrue(process_call["force"])

    def test_failure_is_recorded_and_can_be_retried(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = _create_completed_session(Path(temporary_dir))
            _mark_stage_completed(session_dir, "transcription")
            progress = []

            with (
                patch(
                    "processing.pipeline.transcribe_session",
                    return_value=session_dir / STAGE_OUTPUTS["transcription"],
                ),
                patch(
                    "processing.pipeline.insert_screenshot_markers",
                    side_effect=RuntimeError("alignment exploded"),
                ),
            ):
                with self.assertRaisesRegex(
                    PipelineError,
                    "alignment exploded",
                ):
                    run_pipeline(
                        session_dir,
                        output_language="English",
                        progress_callback=progress.append,
                    )

            pipeline_status = SessionManifest.load(session_dir).data[
                "processing"
            ]["pipeline"]
            self.assertEqual(pipeline_status["status"], "failed")
            self.assertEqual(
                pipeline_status["failed_stage"],
                "screenshot_markers",
            )
            self.assertEqual(progress[-1].status, "failed")


if __name__ == "__main__":
    unittest.main()
