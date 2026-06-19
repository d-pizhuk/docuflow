from pathlib import Path

from PySide6.QtCore import QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices, QFont
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from processing.pipeline import (
    PipelineProgress,
    PipelineResult,
    run_pipeline,
)


class PipelineWorker(QThread):
    progress = Signal(object)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, session_dir: Path, parent=None):
        super().__init__(parent)
        self._session_dir = Path(session_dir)

    def run(self):
        try:
            result = run_pipeline(
                self._session_dir,
                progress_callback=self.progress.emit,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return

        self.completed.emit(result)


class ProcessingDialog(QDialog):
    def __init__(
        self,
        session_dir: Path,
        parent=None,
        *,
        auto_start: bool = True,
    ):
        super().__init__(parent)
        self._session_dir = Path(session_dir)
        self._worker: PipelineWorker | None = None
        self._result: PipelineResult | None = None
        self._is_running = False
        self._run_failed = False

        self.setWindowTitle("DocuFlow — Process Recording")
        self.setMinimumSize(620, 440)
        self.setModal(True)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)

        self._build_ui()
        if auto_start:
            QTimer.singleShot(0, self.start_processing)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(26, 24, 26, 24)
        root.setSpacing(14)

        title = QLabel("Process Recording")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        root.addWidget(title)

        language = self._session_language()
        subtitle = QLabel(
            "DocuFlow will transcribe the recording, align screenshots, and "
            f"generate the final documentation in {language}."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #5f6b7a;")
        root.addWidget(subtitle)

        self._stage_label = QLabel("Ready to process")
        self._stage_label.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        root.addWidget(self._stage_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 6)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%v / %m stages")
        self._progress_bar.setFixedHeight(20)
        root.addWidget(self._progress_bar)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Processing details will appear here.")
        self._log.setStyleSheet(
            "QPlainTextEdit {"
            "background: #f6f8fb;"
            "border: 1px solid #dce3ed;"
            "border-radius: 7px;"
            "padding: 8px;"
            "font-family: Consolas, monospace;"
            "}"
        )
        root.addWidget(self._log, 1)

        self._error_label = QLabel()
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet(
            "color: #b42318; background: #fff2f0; "
            "border: 1px solid #f3b7b0; border-radius: 6px; padding: 8px;"
        )
        self._error_label.hide()
        root.addWidget(self._error_label)

        actions = QHBoxLayout()

        self._open_html_btn = QPushButton("Open HTML")
        self._open_html_btn.clicked.connect(self._open_html)
        self._open_html_btn.hide()
        actions.addWidget(self._open_html_btn)

        self._open_markdown_btn = QPushButton("Open Markdown")
        self._open_markdown_btn.clicked.connect(self._open_markdown)
        self._open_markdown_btn.hide()
        actions.addWidget(self._open_markdown_btn)

        self._open_folder_btn = QPushButton("Open Session Folder")
        self._open_folder_btn.clicked.connect(self._open_session_folder)
        self._open_folder_btn.hide()
        actions.addWidget(self._open_folder_btn)

        actions.addStretch()

        self._retry_btn = QPushButton("Retry")
        self._retry_btn.clicked.connect(self.start_processing)
        self._retry_btn.hide()
        actions.addWidget(self._retry_btn)

        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.accept)
        actions.addWidget(self._close_btn)

        root.addLayout(actions)

    def start_processing(self):
        if self._is_running:
            return

        self._is_running = True
        self._run_failed = False
        self._result = None
        self._retry_btn.hide()
        self._error_label.hide()
        self._close_btn.setEnabled(False)
        self._set_result_buttons_visible(False)
        self._stage_label.setText("Starting processing…")
        self._log.appendPlainText("Starting DocuFlow pipeline")

        self._worker = PipelineWorker(self._session_dir, self)
        self._worker.progress.connect(self._on_progress)
        self._worker.completed.connect(self._on_completed)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_progress(self, progress: PipelineProgress):
        if progress.status in ("completed", "reused"):
            self._progress_bar.setValue(progress.stage_number)
        self._stage_label.setText(
            f"{progress.stage_number}/{progress.total_stages} — "
            f"{progress.label}"
        )
        self._log.appendPlainText(progress.message)

    def _on_completed(self, result: PipelineResult):
        self._result = result
        self._progress_bar.setValue(self._progress_bar.maximum())
        self._stage_label.setText("Documentation complete")
        self._log.appendPlainText(
            f"Final documentation: {result.final_output_path}"
        )

    def _on_failed(self, message: str):
        self._run_failed = True
        self._stage_label.setText("Processing failed")
        self._error_label.setText(message)
        self._error_label.show()
        self._log.appendPlainText(f"Error: {message}")

    def _on_worker_finished(self):
        worker = self._worker
        self._worker = None
        self._is_running = False
        if worker is not None:
            worker.deleteLater()
        self._close_btn.setEnabled(True)
        if self._run_failed:
            self._retry_btn.show()
            self._open_folder_btn.show()
        elif self._result is not None:
            self._set_result_buttons_visible(True)

    def _set_result_buttons_visible(self, visible: bool):
        self._open_html_btn.setVisible(visible)
        self._open_markdown_btn.setVisible(visible)
        self._open_folder_btn.setVisible(visible)

    def _open_html(self):
        if self._result is not None:
            self._open_local_path(self._result.final_output_path)

    def _open_markdown(self):
        if self._result is not None:
            markdown_path = self._result.outputs.get("markdown")
            if markdown_path is not None:
                self._open_local_path(markdown_path)

    def _open_session_folder(self):
        self._open_local_path(self._session_dir)

    @staticmethod
    def _open_local_path(path: Path):
        QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(Path(path).resolve()))
        )

    def _session_language(self) -> str:
        try:
            from session.session_manifest import SessionManifest

            manifest = SessionManifest.load(self._session_dir)
            return manifest.data.get("settings", {}).get(
                "output_language",
                "German",
            )
        except Exception:
            return "German"

    def closeEvent(self, event: QCloseEvent):
        if self._is_running:
            event.ignore()
            return
        super().closeEvent(event)
