import argparse
import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from session.session_manifest import SessionManifest


STAGE_NAME = "html"
MARKDOWN_STAGE = "markdown"
DOCUMENTATION_STAGE = "process_documentation"
DEFAULT_OUTPUT_PATH = Path("output") / "final" / "process_documentation.html"
MARKDOWN_IMAGE_PATTERN = re.compile(
    r"!\[[^\]]*\]\(\s*([^) \t]+)(?:\s+['\"][^'\"]*['\"])?\s*\)"
)


class HtmlRenderError(RuntimeError):
    pass


def render_html_document(
    session_dir: Path,
    *,
    force: bool = False,
) -> Path:
    session_dir = Path(session_dir)
    manifest = SessionManifest.load(session_dir)
    manifest_data = manifest.data
    processing = manifest_data.get("processing", {})

    markdown_stage = processing.get(MARKDOWN_STAGE, {})
    if markdown_stage.get("status") != "completed":
        raise HtmlRenderError(
            "The Markdown stage must be completed before rendering HTML."
        )

    documentation_stage = processing.get(DOCUMENTATION_STAGE, {})
    if documentation_stage.get("status") != "completed":
        raise HtmlRenderError(
            "The process documentation stage must be completed before "
            "rendering HTML."
        )

    markdown_path = manifest.resolve_artifact_path(
        markdown_stage.get(
            "output_path",
            "output/final/final_documentation.md",
        )
    )
    documentation_path = manifest.resolve_artifact_path(
        documentation_stage.get(
            "output_path",
            "output/process_documentation.json",
        )
    )
    output_path = session_dir / DEFAULT_OUTPUT_PATH

    previous_stage = processing.get(STAGE_NAME, {})
    if (
        not force
        and previous_stage.get("status") == "completed"
        and output_path.exists()
    ):
        return output_path

    manifest.start_processing_stage(
        STAGE_NAME,
        markdown_path=markdown_stage.get(
            "output_path",
            "output/final/final_documentation.md",
        ),
    )

    try:
        markdown_text = markdown_path.read_text(encoding="utf-8")
        documentation = _read_json(documentation_path)
        title = str(documentation.get("title", "")).strip()
        if not title:
            raise HtmlRenderError(
                "The process documentation has no valid title."
            )

        image_count = validate_markdown_images(
            markdown_text,
            markdown_path.parent,
        )
        html_body = _convert_markdown(markdown_text)
        generated_at = datetime.now().astimezone()
        output_language = documentation_stage.get(
            "output_language",
            "English",
        )
        html_text = build_html_page(
            title=title,
            html_body=html_body,
            generated_at=generated_at,
            output_language=output_language,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_text(output_path, html_text)

        manifest.complete_processing_stage(
            STAGE_NAME,
            output_path,
            title=title,
            output_language=output_language,
            generated_at=generated_at.isoformat(timespec="seconds"),
            screenshot_count=image_count,
        )
        return output_path
    except Exception as exc:
        manifest.fail_processing_stage(STAGE_NAME, str(exc))
        if isinstance(exc, HtmlRenderError):
            raise
        raise HtmlRenderError(f"HTML rendering failed: {exc}") from exc


def validate_markdown_images(
    markdown_text: str,
    markdown_directory: Path,
) -> int:
    markdown_root = markdown_directory.resolve()
    image_paths = MARKDOWN_IMAGE_PATTERN.findall(markdown_text)

    for image_reference in image_paths:
        normalized_reference = image_reference.replace("%20", " ")
        reference_path = Path(normalized_reference)
        if reference_path.is_absolute():
            raise HtmlRenderError(
                f"Markdown image path must be relative: {image_reference}"
            )

        resolved_path = (markdown_root / reference_path).resolve()
        try:
            resolved_path.relative_to(markdown_root)
        except ValueError as exc:
            raise HtmlRenderError(
                f"Markdown image escapes the final package: {image_reference}"
            ) from exc

        if not resolved_path.is_file():
            raise HtmlRenderError(
                f"Markdown image does not exist: {image_reference}"
            )

    return len(image_paths)


def build_html_page(
    *,
    title: str,
    html_body: str,
    generated_at: datetime,
    output_language: str,
) -> str:
    escaped_title = html.escape(title, quote=True)
    escaped_language = html.escape(output_language, quote=True)
    language_code = _language_code(output_language)
    generated_date = generated_at.strftime("%Y-%m-%d")

    return f"""<!DOCTYPE html>
<html lang="{language_code}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title} · DocuFlow</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #657189;
      --surface: #ffffff;
      --canvas: #eef2f7;
      --line: #dce3ed;
      --accent: #4158d0;
      --accent-soft: #eef0ff;
      --shadow: 0 24px 70px rgba(23, 32, 51, 0.12);
    }}

    * {{
      box-sizing: border-box;
    }}

    html {{
      scroll-behavior: smooth;
      background: var(--canvas);
    }}

    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #f7f4ff 0, transparent 34rem),
        var(--canvas);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.65;
    }}

    .page {{
      width: min(1080px, calc(100% - 32px));
      margin: 32px auto;
      overflow: hidden;
      background: var(--surface);
      border: 1px solid rgba(220, 227, 237, 0.9);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}

    .document-header {{
      padding: 34px clamp(24px, 6vw, 64px);
      color: #ffffff;
      background:
        linear-gradient(135deg, rgba(65, 88, 208, 0.98), rgba(83, 52, 131, 0.96));
    }}

    .eyebrow {{
      margin: 0 0 10px;
      font-size: 0.76rem;
      font-weight: 700;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      opacity: 0.76;
    }}

    .document-header h1 {{
      max-width: 780px;
      margin: 0;
      font-size: clamp(2rem, 5vw, 3.35rem);
      line-height: 1.08;
      letter-spacing: -0.04em;
    }}

    .metadata {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 18px;
      margin: 18px 0 0;
      font-size: 0.9rem;
      opacity: 0.82;
    }}

    main {{
      padding: 20px clamp(24px, 6vw, 64px) 60px;
    }}

    main > h1:first-child {{
      display: none;
    }}

    main > h2:first-of-type {{
      margin-top: 16px;
      color: var(--muted);
      font-size: 0.78rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }}

    main > h2:first-of-type + ul {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px;
      margin: 10px 0 34px;
      padding: 0;
      list-style: none;
    }}

    main > h2:first-of-type + ul a {{
      display: block;
      height: 100%;
      padding: 12px 14px;
      color: var(--accent);
      background: var(--accent-soft);
      border: 1px solid #dfe3ff;
      border-radius: 12px;
      font-weight: 650;
      text-decoration: none;
      transition: transform 140ms ease, background 140ms ease;
    }}

    main > h2:first-of-type + ul a:hover {{
      transform: translateY(-1px);
      background: #e5e8ff;
    }}

    h2 {{
      margin: 42px 0 14px;
      padding-left: 14px;
      border-left: 5px solid var(--accent);
      font-size: clamp(1.35rem, 3vw, 1.8rem);
      line-height: 1.25;
    }}

    h3 {{
      width: max-content;
      min-width: 42px;
      margin: 26px 0 10px;
      padding: 4px 10px;
      color: var(--accent);
      background: var(--accent-soft);
      border-radius: 999px;
      font-size: 0.92rem;
    }}

    h4 {{
      margin: 24px 0 10px;
      font-size: 1.05rem;
    }}

    p {{
      margin: 10px 0 16px;
    }}

    img {{
      display: block;
      width: auto;
      max-width: 100%;
      height: auto;
      margin: 14px auto;
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 14px 36px rgba(23, 32, 51, 0.12);
    }}

    hr {{
      margin: 42px 0;
      border: 0;
      border-top: 1px solid var(--line);
    }}

    a {{
      color: var(--accent);
    }}

    .back-to-top {{
      position: fixed;
      right: 22px;
      bottom: 22px;
      padding: 10px 15px;
      color: #ffffff;
      background: #202a44;
      border: 0;
      border-radius: 999px;
      box-shadow: 0 10px 24px rgba(23, 32, 51, 0.22);
      cursor: pointer;
      font: inherit;
      font-size: 0.84rem;
    }}

    .back-to-top:hover {{
      background: var(--accent);
    }}

    @media (max-width: 640px) {{
      .page {{
        width: 100%;
        margin: 0;
        border: 0;
        border-radius: 0;
      }}

      .document-header {{
        padding-top: 28px;
      }}

      .back-to-top {{
        right: 14px;
        bottom: 14px;
      }}
    }}

    @media print {{
      html, body {{
        background: #ffffff;
      }}

      .page {{
        width: 100%;
        margin: 0;
        border: 0;
        box-shadow: none;
      }}

      .document-header {{
        color: #111827;
        background: #ffffff;
        border-bottom: 2px solid #111827;
      }}

      .back-to-top {{
        display: none;
      }}
    }}
  </style>
</head>
<body id="top">
  <div class="page">
    <header class="document-header">
      <p class="eyebrow">DocuFlow process documentation</p>
      <h1>{escaped_title}</h1>
      <p class="metadata">
        <span>Generated: {generated_date}</span>
        <span>Language: {escaped_language}</span>
      </p>
    </header>
    <main>
{html_body}
    </main>
  </div>
  <button class="back-to-top" type="button"
    onclick="window.scrollTo({{top: 0, behavior: 'smooth'}})">
    Back to top
  </button>
</body>
</html>
"""


def _convert_markdown(markdown_text: str) -> str:
    try:
        from markdown import markdown
    except ModuleNotFoundError as exc:
        raise HtmlRenderError(
            "The Markdown package is missing. Activate the AI environment and "
            "install markdown."
        ) from exc

    return markdown(
        markdown_text,
        extensions=["extra"],
        output_format="html5",
    )


def _language_code(language: str) -> str:
    normalized = language.strip().lower()
    known_languages = {
        "english": "en",
        "german": "de",
        "deutsch": "de",
        "spanish": "es",
        "french": "fr",
        "italian": "it",
        "portuguese": "pt",
        "dutch": "nl",
        "polish": "pl",
        "czech": "cs",
        "japanese": "ja",
        "korean": "ko",
        "chinese": "zh",
        "chinese (simplified)": "zh-Hans",
    }
    if normalized in known_languages:
        return known_languages[normalized]
    if re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})?", normalized):
        return normalized
    return "en"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise HtmlRenderError(f"Expected a JSON object in {path}.")
    return data


def _write_text(path: Path, text: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8", newline="\n")
    temporary_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a portable styled HTML viewer from final DocuFlow "
            "Markdown."
        )
    )
    parser.add_argument(
        "session_dir",
        type=Path,
        help="Path to a recording_session_* directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate and overwrite the existing HTML viewer.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        output_path = render_html_document(
            args.session_dir,
            force=args.force,
        )
    except (
        FileNotFoundError,
        KeyError,
        json.JSONDecodeError,
        HtmlRenderError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"HTML documentation saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
