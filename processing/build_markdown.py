import argparse
import json
import re
import shutil
import sys
import unicodedata
from pathlib import Path

from session.session_manifest import SessionManifest


STAGE_NAME = "markdown"
DOCUMENTATION_STAGE = "process_documentation"
DESCRIPTION_STAGE = "screenshot_descriptions"
DEFAULT_FINAL_DIR = Path("output") / "final"
DEFAULT_OUTPUT_PATH = DEFAULT_FINAL_DIR / "final_documentation.md"
DEFAULT_SCREENSHOTS_DIR = DEFAULT_FINAL_DIR / "screenshots"
SCREENSHOT_MARKER_PATTERN = re.compile(r"\[Screenshot_(\d+)\]")


class MarkdownBuildError(RuntimeError):
    pass


def build_markdown_document(
    session_dir: Path,
    *,
    force: bool = False,
) -> Path:
    session_dir = Path(session_dir)
    manifest = SessionManifest.load(session_dir)
    manifest_data = manifest.data
    processing = manifest_data.get("processing", {})

    documentation_stage = processing.get(DOCUMENTATION_STAGE, {})
    if documentation_stage.get("status") != "completed":
        raise MarkdownBuildError(
            "The process documentation stage must be completed before "
            "building Markdown."
        )

    description_stage = processing.get(DESCRIPTION_STAGE, {})
    if description_stage.get("status") != "completed":
        raise MarkdownBuildError(
            "The screenshot description stage must be completed before "
            "building Markdown."
        )

    documentation_path = manifest.resolve_artifact_path(
        documentation_stage.get(
            "output_path",
            "output/process_documentation.json",
        )
    )
    descriptions_path = manifest.resolve_artifact_path(
        description_stage.get(
            "output_path",
            "output/screenshot_descriptions.json",
        )
    )
    output_path = session_dir / DEFAULT_OUTPUT_PATH
    screenshots_dir = session_dir / DEFAULT_SCREENSHOTS_DIR

    previous_stage = processing.get(STAGE_NAME, {})
    if (
        not force
        and previous_stage.get("status") == "completed"
        and output_path.exists()
    ):
        return output_path

    manifest.start_processing_stage(
        STAGE_NAME,
        documentation_path=documentation_stage.get(
            "output_path",
            "output/process_documentation.json",
        ),
        descriptions_path=description_stage.get(
            "output_path",
            "output/screenshot_descriptions.json",
        ),
    )

    try:
        documentation = _read_json(documentation_path)
        descriptions_index = _read_json(descriptions_path)
        descriptions = _index_descriptions(
            descriptions_index.get("descriptions")
        )
        referenced_ids = _referenced_screenshot_ids(documentation)
        _validate_description_coverage(referenced_ids, descriptions)

        image_paths = _copy_referenced_screenshots(
            manifest,
            referenced_ids,
            descriptions,
            screenshots_dir,
        )
        markdown = render_markdown(
            documentation,
            descriptions,
            image_paths,
        )
        if SCREENSHOT_MARKER_PATTERN.search(markdown):
            raise MarkdownBuildError(
                "The rendered Markdown still contains raw screenshot markers."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_text(output_path, markdown)

        manifest.complete_processing_stage(
            STAGE_NAME,
            output_path,
            final_dir=DEFAULT_FINAL_DIR.as_posix(),
            screenshots_dir=DEFAULT_SCREENSHOTS_DIR.as_posix(),
            screenshot_count=len(referenced_ids),
            major_step_count=len(documentation["steps"]),
            substep_count=sum(
                len(step["substeps"])
                for step in documentation["steps"]
            ),
        )
        return output_path
    except Exception as exc:
        manifest.fail_processing_stage(STAGE_NAME, str(exc))
        if isinstance(exc, MarkdownBuildError):
            raise
        raise MarkdownBuildError(f"Markdown generation failed: {exc}") from exc


def render_markdown(
    documentation: dict,
    descriptions: dict[int, dict],
    image_paths: dict[int, str],
) -> str:
    title = documentation.get("title")
    steps = documentation.get("steps")
    if not isinstance(title, str) or not title.strip():
        raise MarkdownBuildError("The process documentation has no title.")
    if not isinstance(steps, list) or not steps:
        raise MarkdownBuildError(
            "The process documentation contains no major steps."
        )

    lines = [f"# {title.strip()}", "", "## Contents", ""]
    anchors = {}
    for step in steps:
        step_number = step["major_step_number"]
        step_title = str(step["major_step_title"]).strip()
        anchor = f"step-{step_number}-{slugify(step_title)}"
        anchors[step_number] = anchor
        lines.append(
            f"- [{step_number}. {step_title}](#{anchor})"
        )

    lines.extend(["", "---", ""])

    for step_index, step in enumerate(steps):
        step_number = step["major_step_number"]
        step_title = str(step["major_step_title"]).strip()
        lines.extend(
            [
                f'<a id="{anchors[step_number]}"></a>',
                f"## {step_number}. {step_title}",
                "",
            ]
        )

        substeps = step.get("substeps")
        if not isinstance(substeps, list) or not substeps:
            raise MarkdownBuildError(
                f"Major step {step_number} contains no substeps."
            )

        for substep in substeps:
            substep_id = str(substep["substep_id"]).strip()
            action = str(substep["action"]).strip()
            if not substep_id or not action:
                raise MarkdownBuildError(
                    f"Major step {step_number} has an invalid substep."
                )

            lines.extend([f"### {substep_id}", ""])
            lines.extend(
                _render_action(
                    action,
                    descriptions,
                    image_paths,
                )
            )

        if step_index < len(steps) - 1:
            lines.extend(["---", ""])

    return "\n".join(lines).rstrip() + "\n"


def slugify(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return slug or "step"


def _render_action(
    action: str,
    descriptions: dict[int, dict],
    image_paths: dict[int, str],
) -> list[str]:
    lines = []
    cursor = 0

    for match in SCREENSHOT_MARKER_PATTERN.finditer(action):
        text_before = _clean_action_fragment(action[cursor:match.start()])
        if text_before:
            lines.extend([text_before, ""])

        screenshot_id = int(match.group(1))
        description = descriptions[screenshot_id]
        title = str(description["title"]).strip()
        description_text = str(description["description"]).strip()
        alt_text = _escape_alt_text(title)

        lines.extend(
            [
                f"#### {title}",
                "",
                f"![{alt_text}]({image_paths[screenshot_id]})",
                "",
                description_text,
                "",
            ]
        )
        cursor = match.end()

    remaining_text = _clean_action_fragment(action[cursor:])
    if remaining_text:
        lines.extend([remaining_text, ""])

    return lines


def _referenced_screenshot_ids(documentation: dict) -> list[int]:
    ids = []
    for step in documentation.get("steps", []):
        for substep in step.get("substeps", []):
            action = str(substep.get("action", ""))
            ids.extend(
                int(marker_id)
                for marker_id in SCREENSHOT_MARKER_PATTERN.findall(action)
            )

    if len(ids) != len(set(ids)):
        raise MarkdownBuildError(
            "A screenshot marker is referenced more than once in the process "
            "documentation."
        )
    return ids


def _index_descriptions(raw_descriptions) -> dict[int, dict]:
    if not isinstance(raw_descriptions, list):
        raise MarkdownBuildError(
            "The screenshot description index is invalid."
        )

    descriptions = {}
    for index, description in enumerate(raw_descriptions):
        if not isinstance(description, dict):
            raise MarkdownBuildError(
                f"Screenshot description entry {index} is invalid."
            )
        try:
            screenshot_id = int(description["screenshot_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MarkdownBuildError(
                f"Screenshot description entry {index} has no valid ID."
            ) from exc
        if screenshot_id in descriptions:
            raise MarkdownBuildError(
                f"Screenshot description {screenshot_id} is duplicated."
            )
        for field in ("screenshot_path", "title", "description"):
            if not str(description.get(field, "")).strip():
                raise MarkdownBuildError(
                    f"Screenshot description {screenshot_id} has no {field}."
                )
        descriptions[screenshot_id] = description
    return descriptions


def _validate_description_coverage(
    referenced_ids: list[int],
    descriptions: dict[int, dict],
) -> None:
    referenced = set(referenced_ids)
    available = set(descriptions)
    if referenced != available:
        missing = sorted(referenced - available)
        unexpected = sorted(available - referenced)
        details = []
        if missing:
            details.append(
                "missing descriptions for IDs " + ", ".join(map(str, missing))
            )
        if unexpected:
            details.append(
                "unreferenced descriptions for IDs "
                + ", ".join(map(str, unexpected))
            )
        raise MarkdownBuildError(
            "Screenshot description coverage mismatch: " + "; ".join(details)
        )


def _copy_referenced_screenshots(
    manifest: SessionManifest,
    referenced_ids: list[int],
    descriptions: dict[int, dict],
    screenshots_dir: Path,
) -> dict[int, str]:
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    image_paths = {}
    expected_filenames = set()

    for screenshot_id in referenced_ids:
        source_path = manifest.resolve_artifact_path(
            descriptions[screenshot_id]["screenshot_path"]
        )
        extension = source_path.suffix.lower() or ".png"
        filename = f"screenshot_{screenshot_id:03d}{extension}"
        destination_path = screenshots_dir / filename
        shutil.copy2(source_path, destination_path)
        expected_filenames.add(filename)
        image_paths[screenshot_id] = f"screenshots/{filename}"

    for existing_path in screenshots_dir.iterdir():
        if (
            existing_path.is_file()
            and existing_path.name.startswith("screenshot_")
            and existing_path.name not in expected_filenames
        ):
            existing_path.unlink()

    return image_paths


def _clean_action_fragment(text: str) -> str:
    return " ".join(text.split()).strip()


def _escape_alt_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise MarkdownBuildError(f"Expected a JSON object in {path}.")
    return data


def _write_text(path: Path, text: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8", newline="\n")
    temporary_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a portable Markdown document from a processed DocuFlow "
            "session."
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
        help="Rebuild and overwrite the existing Markdown package.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        output_path = build_markdown_document(
            args.session_dir,
            force=args.force,
        )
    except (
        FileNotFoundError,
        KeyError,
        json.JSONDecodeError,
        MarkdownBuildError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Markdown documentation saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
