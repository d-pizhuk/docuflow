import argparse
import json
import sys
from pathlib import Path

from session.session_manifest import SessionManifest


STAGE_NAME = "transcription"
DEFAULT_MODEL = "large-v3"
DEFAULT_OUTPUT_PATH = Path("output") / "transcript.json"


class TranscriptionError(RuntimeError):
    pass


def transcribe_session(
    session_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
    language: str | None = None,
    force: bool = False,
) -> Path:
    session_dir = Path(session_dir)
    manifest = SessionManifest.load(session_dir)
    manifest_data = manifest.data

    if manifest_data.get("status") != "completed":
        raise TranscriptionError(
            "The recording session must be completed before transcription."
        )

    audio_path = manifest.resolve_artifact_path(manifest_data["audio"]["path"])
    output_path = session_dir / DEFAULT_OUTPUT_PATH

    previous_stage = manifest_data.get("processing", {}).get(STAGE_NAME, {})
    if (
        not force
        and previous_stage.get("status") == "completed"
        and output_path.exists()
    ):
        return output_path

    try:
        import torch
        import whisper
    except ModuleNotFoundError as exc:
        raise TranscriptionError(
            "Whisper dependencies are missing. Activate the AI environment and "
            "install openai-whisper and torch."
        ) from exc

    selected_device = _select_device(device, torch)
    manifest.start_processing_stage(
        STAGE_NAME,
        model=model_name,
        device=selected_device,
        requested_language=language or "auto",
        language=language or "auto",
    )

    try:
        print(f"Loading Whisper model '{model_name}' on {selected_device}...")
        model = whisper.load_model(model_name, device=selected_device)

        print(f"Transcribing {audio_path}...")
        result = model.transcribe(
            str(audio_path),
            language=language,
            fp16=selected_device == "cuda",
            verbose=False,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(output_path, result)

        manifest.complete_processing_stage(
            STAGE_NAME,
            output_path,
            model=model_name,
            device=selected_device,
            requested_language=language or "auto",
            language=result.get("language", language or "unknown"),
            segment_count=len(result.get("segments", [])),
        )
        return output_path
    except Exception as exc:
        manifest.fail_processing_stage(
            STAGE_NAME,
            str(exc),
            model=model_name,
            device=selected_device,
        )
        raise TranscriptionError(f"Transcription failed: {exc}") from exc


def _select_device(requested_device: str, torch_module) -> str:
    if requested_device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch_module.cuda.is_available():
        raise TranscriptionError(
            "CUDA was requested, but PyTorch cannot access a CUDA GPU."
        )
    return requested_device


def _write_json(path: Path, data: dict) -> None:
    temporary_path = path.with_suffix(".json.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe one completed DocuFlow recording session."
    )
    parser.add_argument(
        "session_dir",
        type=Path,
        help="Path to a recording_session_* directory.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Whisper model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Processing device (default: auto).",
    )
    parser.add_argument(
        "--language",
        help="Optional language code such as en or de. Default: auto-detect.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing completed transcript.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        output_path = transcribe_session(
            args.session_dir,
            model_name=args.model,
            device=args.device,
            language=args.language,
            force=args.force,
        )
    except (FileNotFoundError, KeyError, json.JSONDecodeError, TranscriptionError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Transcript saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
