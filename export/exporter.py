import base64
import html
import logging
from pathlib import Path

from ai.doc_merger import MergedDoc

logger = logging.getLogger(__name__)

# Downscale embedded/printed images so output files stay a sane size.
_HTML_MAX_IMG_WIDTH = 1400
_PDF_MAX_IMG_WIDTH = 640


class ExportError(RuntimeError):
    """Raised when an export fails. The caller keeps the editor open and shows
    the message (Req10: graceful error, preview stays intact)."""


_CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       max-width: 820px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; line-height: 1.6; }
h1 { font-size: 28px; border-bottom: 2px solid #eee; padding-bottom: 10px; }
h2 { font-size: 20px; margin-top: 32px; }
.step { margin-bottom: 28px; }
.step p { margin: 8px 0; }
figure { margin: 12px 0; }
img { max-width: 100%; border: 1px solid #ddd; border-radius: 6px; }
figcaption { font-size: 13px; color: #555; margin-top: 6px; }
""".strip()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def to_html(doc: MergedDoc, image_dir: Path, out_path: Path) -> Path:
    """Write a single self-contained HTML file (images embedded as data URIs)."""
    try:
        markup = _build_html(doc, Path(image_dir), embed=True)
        Path(out_path).write_text(markup, encoding="utf-8")
        return Path(out_path)
    except OSError as exc:
        raise ExportError(f"Could not write HTML file: {exc}") from exc


def to_pdf(doc: MergedDoc, image_dir: Path, out_path: Path) -> Path:
    """Render the documentation to a PDF using Qt (no external dependencies)."""
    try:
        from PySide6.QtCore import QMarginsF, QSizeF, QUrl, Qt
        from PySide6.QtGui import (
            QImage, QPageLayout, QPageSize, QPdfWriter, QTextDocument,
        )
    except Exception as exc:  # pragma: no cover - PySide6 always present in-app
        raise ExportError(f"PDF export requires PySide6: {exc}") from exc

    image_dir = Path(image_dir)
    writer = QPdfWriter(str(out_path))
    writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
    writer.setPageMargins(QMarginsF(15, 15, 15, 15), QPageLayout.Unit.Millimeter)
    writer.setResolution(96)

    td = QTextDocument()
    td.setDefaultStyleSheet(_CSS)

    # Register each screenshot as a resource so <img src="filename"> resolves.
    # Scale large images down so they fit the page.
    for step in doc.steps:
        if not step.screenshot:
            continue
        path = image_dir / step.screenshot
        if not path.exists():
            logger.warning("PDF export: screenshot not found: %s", path)
            continue
        img = QImage(str(path))
        if img.isNull():
            continue
        if img.width() > _PDF_MAX_IMG_WIDTH:
            img = img.scaledToWidth(_PDF_MAX_IMG_WIDTH, Qt.TransformationMode.SmoothTransformation)
        td.addResource(QTextDocument.ResourceType.ImageResource, QUrl(step.screenshot), img)

    td.setHtml(_build_html(doc, image_dir, embed=False))

    layout = writer.pageLayout()
    page_px = layout.paintRectPixels(writer.resolution())
    td.setPageSize(QSizeF(page_px.size()))

    try:
        td.print_(writer)
    except Exception as exc:
        raise ExportError(f"PDF rendering failed: {exc}") from exc
    return Path(out_path)


# --------------------------------------------------------------------------- #
# HTML construction (shared by HTML export and the PDF renderer)
# --------------------------------------------------------------------------- #

def _build_html(doc: MergedDoc, image_dir: Path, *, embed: bool) -> str:
    parts: list[str] = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{_esc(doc.title)}</title>",
        f"<style>{_CSS}</style>",
        "</head><body>",
        f"<h1>{_esc(doc.title)}</h1>",
    ]

    for i, step in enumerate(doc.steps, 1):
        parts.append("<section class='step'>")
        parts.append(f"<h2>{i}. {_esc(step.title)}</h2>")
        if step.instruction:
            parts.append(f"<p>{_esc(step.instruction)}</p>")

        if step.screenshot:
            if embed:
                src = _image_data_uri(image_dir / step.screenshot)
            else:
                # PDF path: reference the resource registered by URL (filename).
                src = step.screenshot if (image_dir / step.screenshot).exists() else None

            if src:
                alt = _esc(step.image_title or step.screenshot)
                parts.append("<figure>")
                parts.append(f"<img src='{src}' alt='{alt}'>")
                caption = []
                if step.image_title:
                    caption.append(f"<strong>{_esc(step.image_title)}</strong>")
                if step.image_description:
                    caption.append(_esc(step.image_description))
                if caption:
                    parts.append(f"<figcaption>{' — '.join(caption)}</figcaption>")
                parts.append("</figure>")
            else:
                parts.append(f"<p><em>[screenshot missing: {_esc(step.screenshot)}]</em></p>")

        parts.append("</section>")

    parts.append("</body></html>")
    return "\n".join(parts)


def _image_data_uri(path: Path) -> str | None:
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError:
        logger.warning("export: screenshot not found: %s", path)
        return None

    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

    # Downscale for HTML so the embedded file does not balloon. Optional: if
    # Pillow is unavailable we embed the original bytes.
    try:
        import io
        from PIL import Image

        with Image.open(io.BytesIO(raw)) as im:
            if max(im.size) > _HTML_MAX_IMG_WIDTH:
                im = im.convert("RGB")
                ratio = _HTML_MAX_IMG_WIDTH / max(im.size)
                im = im.resize((max(1, int(im.width * ratio)), max(1, int(im.height * ratio))))
                buf = io.BytesIO()
                im.save(buf, format="PNG", optimize=True)
                raw = buf.getvalue()
                media = "image/png"
    except Exception:
        logger.debug("export: Pillow unavailable; embedding original image bytes")

    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{media};base64,{b64}"


def _esc(s: str | None) -> str:
    return html.escape(s or "")