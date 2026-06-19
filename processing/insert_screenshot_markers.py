import argparse
import json
import re
import sys
from pathlib import Path

from session.session_manifest import SessionManifest


STAGE_NAME = "screenshot_markers"
TRANSCRIPTION_STAGE = "transcription"
DEFAULT_OUTPUT_PATH = Path("output") / "transcript_with_screenshots.txt"
DEFAULT_REPORT_PATH = Path("output") / "screenshot_assignments.json"


class ScreenshotMarkerError(RuntimeError):
    pass


def insert_screenshot_markers(
    session_dir: Path,
    *,
    force: bool = False,
) -> Path:
    session_dir = Path(session_dir)
    manifest = SessionManifest.load(session_dir)
    manifest_data = manifest.data

    transcription = manifest_data.get("processing", {}).get(
        TRANSCRIPTION_STAGE,
        {},
    )
    if transcription.get("status") != "completed":
        raise ScreenshotMarkerError(
            "The transcription stage must be completed before inserting "
            "screenshot markers."
        )

    transcript_path = manifest.resolve_artifact_path(
        transcription.get("output_path", "output/transcript.json")
    )
    output_path = session_dir / DEFAULT_OUTPUT_PATH
    report_path = session_dir / DEFAULT_REPORT_PATH

    previous_stage = manifest_data.get("processing", {}).get(STAGE_NAME, {})
    if (
        not force
        and previous_stage.get("status") == "completed"
        and output_path.exists()
        and report_path.exists()
    ):
        return output_path

    manifest.start_processing_stage(
        STAGE_NAME,
        transcript_path="output/transcript.json",
        screenshot_count=len(manifest_data.get("screenshots", [])),
    )

    try:
        transcript = _read_json(transcript_path)
        segments = _normalize_segments(transcript.get("segments"))
        screenshots = _normalize_screenshots(
            manifest,
            manifest_data.get("screenshots", []),
        )

        enriched_text, assignments = build_marker_transcript(
            segments,
            screenshots,
        )
        _validate_marker_coverage(enriched_text, screenshots)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_text(output_path, enriched_text)
        _write_json(
            report_path,
            {
                "version": 1,
                "transcript_path": "output/transcript.json",
                "output_path": DEFAULT_OUTPUT_PATH.as_posix(),
                "assignments": assignments,
            },
        )

        placement_counts = {
            placement: sum(
                assignment["placement"] == placement
                for assignment in assignments
            )
            for placement in (
                "before_segment",
                "after_segment",
                "standalone",
            )
        }
        manifest.complete_processing_stage(
            STAGE_NAME,
            output_path,
            assignment_report_path=DEFAULT_REPORT_PATH.as_posix(),
            screenshot_count=len(screenshots),
            assignment_count=len(assignments),
            placement_counts=placement_counts,
        )
        return output_path
    except Exception as exc:
        manifest.fail_processing_stage(STAGE_NAME, str(exc))
        if isinstance(exc, ScreenshotMarkerError):
            raise
        raise ScreenshotMarkerError(
            f"Screenshot marker insertion failed: {exc}"
        ) from exc


def build_marker_transcript(
    segments: list[dict],
    screenshots: list[dict],
) -> tuple[str, list[dict]]:
    marker_buckets = {
        segment["index"]: {"before": [], "after": []}
        for segment in segments
    }
    standalone = []
    assignments = []

    for screenshot in screenshots:
        segment = _find_containing_segment(
            segments,
            screenshot["elapsed_seconds"],
        )

        if segment is None:
            assignment = {
                **_screenshot_assignment_data(screenshot),
                "placement": "standalone",
                "segment_index": None,
                "segment_id": None,
                "segment_start": None,
                "segment_end": None,
            }
            standalone.append(assignment)
        else:
            distance_from_start = (
                screenshot["elapsed_seconds"] - segment["start"]
            )
            distance_from_end = segment["end"] - screenshot["elapsed_seconds"]
            position = (
                "before"
                if distance_from_start < distance_from_end
                else "after"
            )
            placement = f"{position}_segment"
            assignment = {
                **_screenshot_assignment_data(screenshot),
                "placement": placement,
                "segment_index": segment["index"],
                "segment_id": segment["id"],
                "segment_start": segment["start"],
                "segment_end": segment["end"],
            }
            marker_buckets[segment["index"]][position].append(assignment)

        assignments.append(assignment)

    lines = []
    remaining_standalone = sorted(
        standalone,
        key=lambda item: (item["elapsed_seconds"], item["screenshot_id"]),
    )

    for segment in segments:
        while (
            remaining_standalone
            and remaining_standalone[0]["elapsed_seconds"] < segment["start"]
        ):
            lines.append(remaining_standalone.pop(0)["marker"])

        before = sorted(
            marker_buckets[segment["index"]]["before"],
            key=lambda item: (item["elapsed_seconds"], item["screenshot_id"]),
        )
        after = sorted(
            marker_buckets[segment["index"]]["after"],
            key=lambda item: (item["elapsed_seconds"], item["screenshot_id"]),
        )

        parts = [item["marker"] for item in before]
        if segment["text"]:
            parts.append(segment["text"])
        parts.extend(item["marker"] for item in after)
        if parts:
            lines.append(" ".join(parts))

    lines.extend(item["marker"] for item in remaining_standalone)
    text = "\n".join(lines).strip()
    if text:
        text += "\n"
    return text, assignments


def _find_containing_segment(
    segments: list[dict],
    elapsed_seconds: float,
) -> dict | None:
    candidates = [
        segment
        for segment in segments
        if segment["start"] <= elapsed_seconds <= segment["end"]
    ]
    if not candidates:
        return None

    return min(
        candidates,
        key=lambda segment: (
            abs(
                elapsed_seconds
                - (segment["start"] + segment["end"]) / 2.0
            ),
            -segment["start"],
            segment["index"],
        ),
    )


def _normalize_segments(raw_segments) -> list[dict]:
    if not isinstance(raw_segments, list):
        raise ScreenshotMarkerError(
            "The Whisper transcript does not contain a valid segments list."
        )

    segments = []
    for original_index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            raise ScreenshotMarkerError(
                f"Transcript segment {original_index} is not an object."
            )
        try:
            start = float(raw_segment["start"])
            end = float(raw_segment["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ScreenshotMarkerError(
                f"Transcript segment {original_index} has invalid timestamps."
            ) from exc
        if start < 0 or end < start:
            raise ScreenshotMarkerError(
                f"Transcript segment {original_index} has an invalid time range."
            )

        segments.append(
            {
                "index": original_index,
                "id": raw_segment.get("id", original_index),
                "start": start,
                "end": end,
                "text": str(raw_segment.get("text", "")).strip(),
            }
        )

    return sorted(
        segments,
        key=lambda segment: (
            segment["start"],
            segment["end"],
            segment["index"],
        ),
    )


def _normalize_screenshots(
    manifest: SessionManifest,
    raw_screenshots,
) -> list[dict]:
    if not isinstance(raw_screenshots, list):
        raise ScreenshotMarkerError(
            "The session manifest does not contain a valid screenshots list."
        )

    screenshots = []
    seen_ids = set()
    for index, raw_screenshot in enumerate(raw_screenshots):
        if not isinstance(raw_screenshot, dict):
            raise ScreenshotMarkerError(
                f"Screenshot entry {index} is not an object."
            )
        try:
            screenshot_id = int(raw_screenshot["id"])
            elapsed_seconds = float(raw_screenshot["elapsed_seconds"])
            screenshot_path = raw_screenshot["path"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ScreenshotMarkerError(
                f"Screenshot entry {index} is invalid."
            ) from exc

        if screenshot_id < 1 or screenshot_id in seen_ids:
            raise ScreenshotMarkerError(
                f"Screenshot ID {screenshot_id} is invalid or duplicated."
            )
        if elapsed_seconds < 0:
            raise ScreenshotMarkerError(
                f"Screenshot {screenshot_id} has a negative timestamp."
            )

        manifest.resolve_artifact_path(screenshot_path)
        seen_ids.add(screenshot_id)
        screenshots.append(
            {
                "id": screenshot_id,
                "elapsed_seconds": elapsed_seconds,
                "path": screenshot_path,
            }
        )

    return sorted(
        screenshots,
        key=lambda item: (item["elapsed_seconds"], item["id"]),
    )


def _screenshot_assignment_data(screenshot: dict) -> dict:
    return {
        "screenshot_id": screenshot["id"],
        "marker": f"[Screenshot_{screenshot['id']}]",
        "elapsed_seconds": screenshot["elapsed_seconds"],
        "screenshot_path": screenshot["path"],
    }


def _validate_marker_coverage(text: str, screenshots: list[dict]) -> None:
    found_ids = [
        int(match)
        for match in re.findall(r"\[Screenshot_(\d+)\]", text)
    ]
    expected_ids = [screenshot["id"] for screenshot in screenshots]

    if sorted(found_ids) != sorted(expected_ids):
        raise ScreenshotMarkerError(
            "Generated marker coverage does not match the session screenshots."
        )
    if len(found_ids) != len(set(found_ids)):
        raise ScreenshotMarkerError(
            "A screenshot marker was inserted more than once."
        )


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ScreenshotMarkerError(f"Expected a JSON object in {path}.")
    return data


def _write_text(path: Path, text: str) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(text, encoding="utf-8", newline="\n")
    temporary_path.replace(path)


def _write_json(path: Path, data: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Insert timestamped screenshot markers into a Whisper transcript."
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
        help="Overwrite existing marker output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        output_path = insert_screenshot_markers(
            args.session_dir,
            force=args.force,
        )
    except (
        FileNotFoundError,
        KeyError,
        json.JSONDecodeError,
        ScreenshotMarkerError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Transcript with screenshot markers saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
