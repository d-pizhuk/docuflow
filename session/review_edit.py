import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QFont, QImage, QPixmap, QTextDocument
from PySide6.QtWidgets import (
    QApplication, QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSplitter,
    QTabWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from ai.doc_merger import MergedDoc, MergedStep
from export import exporter

logger = logging.getLogger(__name__)

_PREVIEW_DEBOUNCE_MS = 300   # re-render this long after the last keystroke (< NfReq3's 1 s)
_PREVIEW_IMG_WIDTH = 460     # scale screenshots to this width in the live preview


class _PreviewBrowser(QTextBrowser):
    """Markdown preview that scales screenshots down on the fly via loadResource,
    so we never blow up the view with a full-resolution capture and need no temp
    files."""

    def __init__(self, image_dir: Path, parent=None):
        super().__init__(parent)
        self._image_dir = Path(image_dir)
        self.setOpenExternalLinks(False)

    def loadResource(self, type_, url: QUrl):
        if type_ == QTextDocument.ResourceType.ImageResource:
            name = url.fileName() or url.toString()
            path = self._image_dir / name
            if path.exists():
                img = QImage(str(path))
                if not img.isNull() and img.width() > _PREVIEW_IMG_WIDTH:
                    img = img.scaledToWidth(_PREVIEW_IMG_WIDTH, Qt.TransformationMode.SmoothTransformation)
                return img
        return super().loadResource(type_, url)


class _StepCard(QFrame):
    """One editable step: title, instruction, screenshot thumbnail and (if any)
    the VLM image caption fields. Carries an AI/Edited badge that flips to
    'Edited' the first time the user changes any field."""

    def __init__(self, index: int, step: MergedStep, image_dir: Path, on_changed):
        super().__init__()
        self._screenshot = step.screenshot
        self._on_changed = on_changed
        self._edited = False

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("QFrame { background: #fbfbfd; border: 1px solid #e3e3ea; border-radius: 8px; }")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 12)
        outer.setSpacing(6)

        header = QHBoxLayout()
        header.addWidget(_dim(f"Step {index}"))
        header.addStretch()
        self._badge = QLabel()
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge.setFixedWidth(64)
        header.addWidget(self._badge)
        outer.addLayout(header)

        self._title = QLineEdit(step.title)
        self._title.setStyleSheet("font-weight: bold; font-size: 14px; border: 1px solid #ddd; border-radius: 4px; padding: 4px;")
        self._title.textEdited.connect(self._mark_edited)
        outer.addWidget(self._title)

        self._instruction = QPlainTextEdit(step.instruction)
        self._instruction.setFixedHeight(64)
        self._instruction.textChanged.connect(self._mark_edited)
        outer.addWidget(self._instruction)

        self._image_title: QLineEdit | None = None
        self._image_desc: QPlainTextEdit | None = None

        if self._screenshot:
            row = QHBoxLayout()
            thumb = QLabel()
            thumb.setFixedSize(180, 110)
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb.setStyleSheet("border: 1px solid #ddd; border-radius: 4px; color: #999;")
            pix = QPixmap(str(Path(image_dir) / self._screenshot))
            if pix.isNull():
                thumb.setText("image\nnot found")
            else:
                thumb.setPixmap(pix.scaled(176, 106, Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation))
            row.addWidget(thumb)

            cap = QVBoxLayout()
            cap.addWidget(_dim(f"📷 {self._screenshot}"))
            self._image_title = QLineEdit(step.image_title or "")
            self._image_title.setPlaceholderText("Image title (AI)")
            self._image_title.textEdited.connect(self._mark_edited)
            cap.addWidget(self._image_title)
            self._image_desc = QPlainTextEdit(step.image_description or "")
            self._image_desc.setPlaceholderText("Image description (AI)")
            self._image_desc.setFixedHeight(48)
            self._image_desc.textChanged.connect(self._mark_edited)
            cap.addWidget(self._image_desc)
            row.addLayout(cap)
            outer.addLayout(row)

        self._update_badge()

    def _mark_edited(self, *_):
        if not self._edited:
            self._edited = True
            self._update_badge()
        self._on_changed()

    def _update_badge(self):
        if self._edited:
            self._badge.setText("Edited")
            self._badge.setStyleSheet("background:#f39c12; color:white; border-radius:8px; font-size:11px; padding:2px;")
        else:
            self._badge.setText("AI")
            self._badge.setStyleSheet("background:#27ae60; color:white; border-radius:8px; font-size:11px; padding:2px;")

    def to_step(self) -> MergedStep:
        return MergedStep(
            title=self._title.text().strip(),
            instruction=self._instruction.toPlainText().strip(),
            screenshot=self._screenshot,
            image_title=(self._image_title.text().strip() if self._image_title else None) or None,
            image_description=(self._image_desc.toPlainText().strip() if self._image_desc else None) or None,
        )


class ReviewEditWindow(QDialog):
    """Req9: editable, side-by-side review of the generated documentation, with a
    live Markdown preview, AI/Edited labels, the original transcript for
    comparison, and HTML/PDF export (Req10)."""

    def __init__(self, merged: MergedDoc | None, annotated: str, session_dir: Path,
                 error: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Review & Edit Documentation")
        self.resize(960, 680)

        self._session_dir = Path(session_dir)
        self._cards: list[_StepCard] = []

        root = QVBoxLayout(self)

        if error:
            banner = QLabel(f"⚠ AI generation failed: {error}\nYour transcript is shown on the right and saved in the session folder.")
            banner.setWordWrap(True)
            banner.setStyleSheet("background:#fdecea; color:#922b21; border-radius:6px; padding:8px;")
            root.addWidget(banner)

        # Document title
        title_row = QHBoxLayout()
        title_row.addWidget(_dim("Title"))
        self._title_edit = QLineEdit(merged.title if merged else "Documentation")
        self._title_edit.setStyleSheet("font-size: 16px; font-weight: bold; padding: 4px;")
        self._title_edit.textEdited.connect(self._schedule_preview)
        title_row.addWidget(self._title_edit)
        root.addLayout(title_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        # Left: editable step cards
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        if merged and merged.steps:
            for i, step in enumerate(merged.steps, 1):
                card = _StepCard(i, step, self._session_dir, self._schedule_preview)
                self._cards.append(card)
                left_layout.addWidget(card)
        else:
            left_layout.addWidget(QLabel("No steps to edit."))
        left_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left)
        splitter.addWidget(scroll)

        # Right: tabs (rendered preview + original transcript)
        tabs = QTabWidget()
        self._preview = _PreviewBrowser(self._session_dir)
        tabs.addTab(self._preview, "Preview")

        transcript_view = QTextBrowser()
        transcript_view.setFont(QFont("Consolas", 10))
        transcript_view.setPlainText(annotated or "(no transcript)")
        tabs.addTab(transcript_view, "Original transcript")
        splitter.addWidget(tabs)
        splitter.setSizes([520, 440])

        # Bottom bar
        bar = QHBoxLayout()
        self._status = QLabel("")
        self._status.setStyleSheet("color:#555; font-size: 11px;")
        bar.addWidget(self._status)
        bar.addStretch()

        self._html_btn = QPushButton("Export HTML")
        self._html_btn.clicked.connect(lambda: self._export("html"))
        bar.addWidget(self._html_btn)

        self._pdf_btn = QPushButton("Export PDF")
        self._pdf_btn.clicked.connect(lambda: self._export("pdf"))
        bar.addWidget(self._pdf_btn)

        done = QPushButton("Done")
        done.clicked.connect(self.accept)
        bar.addWidget(done)
        root.addLayout(bar)

        if not self._cards:
            self._html_btn.setEnabled(False)
            self._pdf_btn.setEnabled(False)

        # Debounced live preview (NfReq3: update within 1 s of an edit)
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(_PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._render_preview)

        self._render_preview()

    # ------------------------------------------------------------------ #
    # Building the current (possibly edited) document
    # ------------------------------------------------------------------ #

    def current_doc(self) -> MergedDoc:
        return MergedDoc(
            title=self._title_edit.text().strip() or "Documentation",
            steps=[c.to_step() for c in self._cards],
        )

    # ------------------------------------------------------------------ #
    # Live preview
    # ------------------------------------------------------------------ #

    def _schedule_preview(self, *_):
        self._preview_timer.start()  # restart; renders once typing pauses

    def _render_preview(self):
        self._preview.setMarkdown(self._build_markdown(self.current_doc()))

    @staticmethod
    def _build_markdown(doc: MergedDoc) -> str:
        lines = [f"# {doc.title}", ""]
        for i, s in enumerate(doc.steps, 1):
            lines.append(f"## {i}. {s.title}")
            if s.instruction:
                lines.append("")
                lines.append(s.instruction)
            if s.screenshot:
                lines.append("")
                lines.append(f"![{s.image_title or s.screenshot}]({s.screenshot})")
                caption = []
                if s.image_title:
                    caption.append(f"**{s.image_title}**")
                if s.image_description:
                    caption.append(s.image_description)
                if caption:
                    lines.append("")
                    lines.append(" — ".join(caption))
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Export (Req10)
    # ------------------------------------------------------------------ #

    def _export(self, fmt: str):
        suffix = ".html" if fmt == "html" else ".pdf"
        default = str(self._session_dir / f"documentation{suffix}")
        filt = "HTML (*.html)" if fmt == "html" else "PDF (*.pdf)"
        out_path, _ = QFileDialog.getSaveFileName(self, f"Export {fmt.upper()}", default, filt)
        if not out_path:
            return

        try:
            if fmt == "html":
                exporter.to_html(self.current_doc(), self._session_dir, Path(out_path))
            else:
                exporter.to_pdf(self.current_doc(), self._session_dir, Path(out_path))
            self._status.setText(f"Exported to {out_path}")
        except exporter.ExportError as exc:
            logger.warning("Export failed: %s", exc)
            QMessageBox.warning(self, "Export failed", str(exc))
            self._status.setText("Export failed — your edits are unchanged.")
        except Exception as exc:  # noqa: BLE001 — never let export crash the editor
            logger.exception("Unexpected export error")
            QMessageBox.warning(self, "Export failed", f"Unexpected error: {exc}")
            self._status.setText("Export failed — your edits are unchanged.")


def _dim(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("color:#888; font-size: 11px;")
    return lbl