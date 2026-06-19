import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from processing.pipeline import (
    PipelineProgress,
    PipelineResult,
)
from session.processing_dialog import PipelineWorker, ProcessingDialog
from session.session_manifest import SessionManifest


TEST_TEMP_ROOT = Path(__file__).parent / ".tmp"
TEST_TEMP_ROOT.mkdir(exist_ok=True)
_APP = None


def _application() -> QApplication:
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


class PipelineWorkerTests(unittest.TestCase):
    def test_worker_forwards_progress_and_completion(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = Path(temporary_dir) / "session"
            progress_events = []
            results = []
            failures = []
            result = PipelineResult(
                session_dir=session_dir,
                final_output_path=session_dir / "final.html",
                outputs={"markdown": session_dir / "final.md"},
                executed_stages=("transcription",),
                reused_stages=(),
            )

            worker = PipelineWorker(session_dir)
            worker.progress.connect(progress_events.append)
            worker.completed.connect(results.append)
            worker.failed.connect(failures.append)

            def fake_pipeline(_session_dir, *, progress_callback):
                progress_callback(
                    PipelineProgress(
                        stage_number=1,
                        total_stages=6,
                        stage_key="transcription",
                        label="Transcribing audio",
                        status="completed",
                        message="Transcribing audio - completed",
                    )
                )
                return result

            with patch(
                "session.processing_dialog.run_pipeline",
                side_effect=fake_pipeline,
            ):
                worker.run()

            self.assertEqual(len(progress_events), 1)
            self.assertEqual(results, [result])
            self.assertEqual(failures, [])


class ProcessingDialogTests(unittest.TestCase):
    def test_success_state_exposes_result_actions(self):
        app = _application()
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = Path(temporary_dir) / "session"
            audio_path = session_dir / "recording.wav"
            manifest = SessionManifest.create(
                session_dir,
                device_name="Test microphone",
                sample_rate=16_000,
                channels=1,
                output_language="English",
            )
            audio_path.write_bytes(b"audio")
            manifest.complete(audio_path, 2.0)

            dialog = ProcessingDialog(
                session_dir,
                auto_start=False,
            )
            progress = PipelineProgress(
                stage_number=6,
                total_stages=6,
                stage_key="html",
                label="Rendering HTML",
                status="completed",
                message="Rendering HTML - completed",
            )
            result = PipelineResult(
                session_dir=session_dir,
                final_output_path=session_dir / "final.html",
                outputs={"markdown": session_dir / "final.md"},
                executed_stages=("html",),
                reused_stages=(),
            )

            dialog._on_progress(progress)
            dialog._on_completed(result)
            dialog._on_worker_finished()
            app.processEvents()

            self.assertEqual(dialog._progress_bar.value(), 6)
            self.assertTrue(dialog._open_html_btn.isVisibleTo(dialog))
            self.assertTrue(dialog._open_markdown_btn.isVisibleTo(dialog))
            self.assertTrue(dialog._open_folder_btn.isVisibleTo(dialog))
            self.assertTrue(dialog._close_btn.isEnabled())
            dialog.close()

    def test_failure_state_allows_retry(self):
        _application()
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary_dir:
            session_dir = Path(temporary_dir) / "session"
            dialog = ProcessingDialog(
                session_dir,
                auto_start=False,
            )

            dialog._on_failed("API unavailable")
            dialog._on_worker_finished()

            self.assertTrue(dialog._retry_btn.isVisibleTo(dialog))
            self.assertTrue(dialog._error_label.isVisibleTo(dialog))
            self.assertTrue(dialog._close_btn.isEnabled())
            dialog.close()


if __name__ == "__main__":
    unittest.main()
