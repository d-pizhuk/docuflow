import base64
import html
import logging
import re
from pathlib import Path

from xhtml2pdf import pisa
import markdownify
from bs4 import BeautifulSoup

from ai.doc_merger import MergedDoc

logger = logging.getLogger(__name__)

_HTML_MAX_IMG_WIDTH = 1400


class ExportError(RuntimeError):
    """Raised when an export fails."""


_CSS = """
body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
       max-width: 820px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; line-height: 1.6; }
h1 { font-size: 28px; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-bottom: 20px; }
h2 { font-size: 20px; margin-top: 32px; margin-bottom: 10px; }
.step { margin-bottom: 28px; }
p { margin: 0 0 12px 0; line-height: 1.5; }
.img-wrap { text-align: center; margin: 0; padding: 0; }
.img-wrap img { max-width: 100%; border: 1px solid #ddd; display: block; margin: 0 auto; }
.img-caption { text-align: center; font-size: 13px; color: #555;
               line-height: 1.3; margin: 4px 0 16px 0; padding: 0; }
""".strip()


def to_html(doc: MergedDoc, image_dir: Path, out_path: Path) -> Path:
    try:
        markup = _build_html(doc, Path(image_dir), embed=True)
        Path(out_path).write_text(markup, encoding="utf-8")
        return Path(out_path)
    except OSError as exc:
        raise ExportError(f"Could not write HTML file: {exc}") from exc


def to_pdf(doc: MergedDoc, image_dir: Path, out_path: Path) -> Path:
    markup = _build_html(doc, Path(image_dir), embed=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as pdf_file:
        result = pisa.CreatePDF(
            src=markup,
            dest=pdf_file,
            encoding="utf-8",
        )

    if result.err:
        raise ExportError(f"PDF rendering failed with {result.err} error(s).")

    return out_path


def to_markdown(doc: MergedDoc, image_dir: Path, out_path: Path) -> Path:
    try:
        markup = _build_html(doc, Path(image_dir), embed=False)

        soup = BeautifulSoup(markup, "html.parser")
        body = soup.find("body")
        html_body = str(body) if body else markup

        content = markdownify.markdownify(html_body, heading_style="ATX", bullets="-")

        content = re.sub(r'\n{3,}', '\n\n', content).strip()

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content + "\n", encoding="utf-8")
        return out_path
    except OSError as exc:
        raise ExportError(f"Could not write Markdown file: {exc}") from exc


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
            src = (
                _image_data_uri(image_dir / step.screenshot)
                if embed
                else (step.screenshot if (image_dir / step.screenshot).exists() else None)
            )

            if src:
                alt = _esc(step.image_title or step.screenshot)
                parts.append(
                    f"<div class='img-wrap'>"
                    f"<img src='{src}' alt='{alt}'>"
                    f"</div>"
                )

                if step.image_title or step.image_description:
                    caption_parts = []
                    if step.image_title:
                        caption_parts.append(f"<strong>{_esc(step.image_title)}</strong>")
                    if step.image_description:
                        caption_parts.append(_esc(step.image_description))
                    parts.append(
                        f"<p class='img-caption'>{' — '.join(caption_parts)}</p>"
                    )
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