import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from processing.build_markdown import build_markdown_document
from processing.describe_screenshots import describe_screenshots
from processing.generate_process_documentation import (
    generate_process_documentation,
)
from processing.insert_screenshot_markers import insert_screenshot_markers
from processing.render_html import render_html_document
from processing.transcribe import transcribe_session
from session.language_options import DEFAULT_DOCUMENTATION_LANGUAGE
from session.session_manifest import SessionManifest


PIPELINE_STAGE_NAME = "pipeline"
TOTAL_STAGES = 6
STAGE_KEYS = (
    "transcription",
    "screenshot_markers",
    "process_documentation",
    "screenshot_descriptions",
    "markdown",
    "html",
)


class PipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class PipelineProgress:
    stage_number: int
    total_stages: int
    stage_key: str
    label: str
    status: str
    message: str
    output_path: Path | None = None


@dataclass(frozen=True)
class PipelineResult:
    session_dir: Path
    final_output_path: Path
    outputs: dict[str, Path]
    executed_stages: tuple[str, ...]
    reused_stages: tuple[str, ...]


@dataclass(frozen=True)
class _StageDefinition:
    key: str
    label: str
    output_path: Path
    extra_required_paths: tuple[Path, ...]
    run: Callable[[bool], Path]
    configuration_matches: Callable[[dict], bool]


ProgressCallback = Callable[[PipelineProgress], None]


def run_pipeline(
    session_dir: Path,
    *,
    whisper_model: str = "large-v3",
    device: str = "auto",
    spoken_language: str | None = None,
    openai_model: str = "gpt-4o",
    output_language: str | None = None,
    context_window: int = 2,
    force: bool = False,
    restart_from: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PipelineResult:
    session_dir = Path(session_dir)
    manifest = SessionManifest.load(session_dir)
    if manifest.status != "completed":
        raise PipelineError(
            "The recording session must be completed before processing."
        )
    if restart_from is not None and restart_from not in STAGE_KEYS:
        raise PipelineError(
            f"Unknown restart stage '{restart_from}'."
        )
    if context_window < 0:
        raise PipelineError("The transcript context window cannot be negative.")

    selected_output_language = (
        output_language
        or manifest.data.get("settings", {}).get("output_language")
        or DEFAULT_DOCUMENTATION_LANGUAGE
    )
    restart_index = (
        STAGE_KEYS.index(restart_from)
        if restart_from is not None
        else None
    )
    stages = _build_stages(
        session_dir,
        whisper_model=whisper_model,
        device=device,
        spoken_language=spoken_language,
        openai_model=openai_model,
        output_language=selected_output_language,
        context_window=context_window,
    )

    manifest.start_processing_stage(
        PIPELINE_STAGE_NAME,
        total_stages=TOTAL_STAGES,
        whisper_model=whisper_model,
        device=device,
        spoken_language=spoken_language or "auto",
        openai_model=openai_model,
        output_language=selected_output_language,
        context_window=context_window,
        force=force,
        restart_from=restart_from,
    )

    outputs = {}
    executed_stages = []
    reused_stages = []
    upstream_changed = False
    current_stage = None

    try:
        for index, stage in enumerate(stages):
            current_stage = stage
            current_manifest = SessionManifest.load(session_dir).data
            stage_data = current_manifest.get("processing", {}).get(
                stage.key,
                {},
            )
            reusable = _stage_is_reusable(
                session_dir,
                stage,
                stage_data,
            )
            configuration_matches = stage.configuration_matches(stage_data)
            restart_requested = (
                restart_index is not None and index >= restart_index
            )
            effective_force = (
                force
                or restart_requested
                or upstream_changed
                or (reusable and not configuration_matches)
            )
            will_reuse = reusable and not effective_force

            _emit(
                progress_callback,
                PipelineProgress(
                    stage_number=index + 1,
                    total_stages=TOTAL_STAGES,
                    stage_key=stage.key,
                    label=stage.label,
                    status="starting",
                    message=(
                        f"{stage.label} - checking existing output"
                        if will_reuse
                        else f"{stage.label}..."
                    ),
                ),
            )

            output_path = stage.run(effective_force)
            if not output_path.is_file():
                raise PipelineError(
                    f"{stage.label} did not create its expected output: "
                    f"{output_path}"
                )
            outputs[stage.key] = output_path

            if will_reuse:
                reused_stages.append(stage.key)
                status = "reused"
                message = f"{stage.label} - reused existing output"
            else:
                executed_stages.append(stage.key)
                upstream_changed = True
                status = "completed"
                message = f"{stage.label} - completed"

            _emit(
                progress_callback,
                PipelineProgress(
                    stage_number=index + 1,
                    total_stages=TOTAL_STAGES,
                    stage_key=stage.key,
                    label=stage.label,
                    status=status,
                    message=message,
                    output_path=output_path,
                ),
            )

        final_output = outputs["html"]
        completed_manifest = SessionManifest.load(session_dir)
        completed_manifest.complete_processing_stage(
            PIPELINE_STAGE_NAME,
            final_output,
            total_stages=TOTAL_STAGES,
            executed_stages=executed_stages,
            reused_stages=reused_stages,
            whisper_model=whisper_model,
            device=device,
            spoken_language=spoken_language or "auto",
            openai_model=openai_model,
            output_language=selected_output_language,
            context_window=context_window,
        )
        return PipelineResult(
            session_dir=session_dir,
            final_output_path=final_output,
            outputs=outputs,
            executed_stages=tuple(executed_stages),
            reused_stages=tuple(reused_stages),
        )
    except Exception as exc:
        failed_key = current_stage.key if current_stage else "initialization"
        failed_label = (
            current_stage.label if current_stage else "Pipeline initialization"
        )
        failed_manifest = SessionManifest.load(session_dir)
        failed_manifest.fail_processing_stage(
            PIPELINE_STAGE_NAME,
            str(exc),
            failed_stage=failed_key,
            executed_stages=executed_stages,
            reused_stages=reused_stages,
        )
        _emit(
            progress_callback,
            PipelineProgress(
                stage_number=(
                    STAGE_KEYS.index(failed_key) + 1
                    if failed_key in STAGE_KEYS
                    else 0
                ),
                total_stages=TOTAL_STAGES,
                stage_key=failed_key,
                label=failed_label,
                status="failed",
                message=f"{failed_label} - failed: {exc}",
            ),
        )
        if isinstance(exc, PipelineError):
            raise
        raise PipelineError(
            f"Pipeline failed during {failed_label.lower()}: {exc}"
        ) from exc


def _build_stages(
    session_dir: Path,
    *,
    whisper_model: str,
    device: str,
    spoken_language: str | None,
    openai_model: str,
    output_language: str,
    context_window: int,
) -> tuple[_StageDefinition, ...]:
    requested_spoken_language = spoken_language or "auto"

    return (
        _StageDefinition(
            key="transcription",
            label="Transcribing audio",
            output_path=Path("output") / "transcript.json",
            extra_required_paths=(),
            run=lambda force: transcribe_session(
                session_dir,
                model_name=whisper_model,
                device=device,
                language=spoken_language,
                force=force,
            ),
            configuration_matches=lambda data: (
                data.get("model") == whisper_model
                and (device == "auto" or data.get("device") == device)
                and (
                    (
                        requested_spoken_language == "auto"
                        and data.get("requested_language", "auto") == "auto"
                    )
                    or (
                        requested_spoken_language != "auto"
                        and data.get(
                            "requested_language",
                            data.get("language"),
                        )
                        == requested_spoken_language
                    )
                )
            ),
        ),
        _StageDefinition(
            key="screenshot_markers",
            label="Aligning screenshots",
            output_path=(
                Path("output") / "transcript_with_screenshots.txt"
            ),
            extra_required_paths=(
                Path("output") / "screenshot_assignments.json",
            ),
            run=lambda force: insert_screenshot_markers(
                session_dir,
                force=force,
            ),
            configuration_matches=lambda _data: True,
        ),
        _StageDefinition(
            key="process_documentation",
            label="Generating process steps",
            output_path=Path("output") / "process_documentation.json",
            extra_required_paths=(),
            run=lambda force: generate_process_documentation(
                session_dir,
                model_name=openai_model,
                output_language=output_language,
                force=force,
            ),
            configuration_matches=lambda data: (
                data.get("model") == openai_model
                and data.get("output_language") == output_language
            ),
        ),
        _StageDefinition(
            key="screenshot_descriptions",
            label="Describing screenshots",
            output_path=Path("output") / "screenshot_descriptions.json",
            extra_required_paths=(),
            run=lambda force: describe_screenshots(
                session_dir,
                model_name=openai_model,
                output_language=output_language,
                context_window=context_window,
                force=force,
            ),
            configuration_matches=lambda data: (
                data.get("model") == openai_model
                and data.get("output_language") == output_language
                and data.get("context_window") == context_window
            ),
        ),
        _StageDefinition(
            key="markdown",
            label="Building Markdown",
            output_path=(
                Path("output") / "final" / "final_documentation.md"
            ),
            extra_required_paths=(),
            run=lambda force: build_markdown_document(
                session_dir,
                force=force,
            ),
            configuration_matches=lambda _data: True,
        ),
        _StageDefinition(
            key="html",
            label="Rendering HTML",
            output_path=(
                Path("output") / "final" / "process_documentation.html"
            ),
            extra_required_paths=(),
            run=lambda force: render_html_document(
                session_dir,
                force=force,
            ),
            configuration_matches=lambda _data: True,
        ),
    )


def _stage_is_reusable(
    session_dir: Path,
    stage: _StageDefinition,
    stage_data: dict,
) -> bool:
    if stage_data.get("status") != "completed":
        return False

    required_paths = (stage.output_path, *stage.extra_required_paths)
    return all((session_dir / path).is_file() for path in required_paths)


def _emit(
    callback: ProgressCallback | None,
    progress: PipelineProgress,
) -> None:
    if callback is not None:
        callback(progress)


def _console_progress(progress: PipelineProgress) -> None:
    prefix = (
        f"[{progress.stage_number}/{progress.total_stages}]"
        if progress.stage_number
        else "[pipeline]"
    )
    print(f"{prefix} {progress.message}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete DocuFlow processing pipeline for one recording "
            "session."
        )
    )
    parser.add_argument(
        "session_dir",
        type=Path,
        help="Path to a recording_session_* directory.",
    )
    parser.add_argument(
        "--whisper-model",
        default="large-v3",
        help="Whisper model name (default: large-v3).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Whisper processing device (default: auto).",
    )
    parser.add_argument(
        "--spoken-language",
        help="Optional spoken-language code, such as en or de.",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o",
        help="OpenAI text and vision model (default: gpt-4o).",
    )
    parser.add_argument(
        "--output-language",
        help=(
            "Override the language saved with the recording session. "
            "Sessions without a saved language default to German."
        ),
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=2,
        help="Transcript lines around each screenshot marker (default: 2).",
    )
    parser.add_argument(
        "--restart-from",
        choices=STAGE_KEYS,
        help=(
            "Rerun the selected stage and every downstream stage while "
            "reusing earlier completed stages."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun all stages even when completed outputs exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        result = run_pipeline(
            args.session_dir,
            whisper_model=args.whisper_model,
            device=args.device,
            spoken_language=args.spoken_language,
            openai_model=args.openai_model,
            output_language=args.output_language,
            context_window=args.context_window,
            force=args.force,
            restart_from=args.restart_from,
            progress_callback=_console_progress,
        )
    except (
        FileNotFoundError,
        KeyError,
        json.JSONDecodeError,
        PipelineError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Pipeline complete: {result.final_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
