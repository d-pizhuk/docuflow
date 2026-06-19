import json
import os
import threading
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from session.language_options import DEFAULT_DOCUMENTATION_LANGUAGE


SESSION_VERSION = 1
MANIFEST_FILENAME = "session.json"


def _local_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


class SessionManifest:
    """Owns the persisted metadata shared by recording and processing stages."""

    def __init__(self, session_dir: Path, data: dict):
        self.session_dir = Path(session_dir)
        self.path = self.session_dir / MANIFEST_FILENAME
        self._data = data
        self._lock = threading.Lock()

    @classmethod
    def create(
        cls,
        session_dir: Path,
        *,
        device_name: str,
        sample_rate: int,
        channels: int,
        output_language: str = DEFAULT_DOCUMENTATION_LANGUAGE,
    ) -> "SessionManifest":
        session_dir = Path(session_dir)
        (session_dir / "screenshots").mkdir(parents=True, exist_ok=True)
        (session_dir / "output").mkdir(parents=True, exist_ok=True)

        manifest = cls(
            session_dir,
            {
                "version": SESSION_VERSION,
                "status": "recording",
                "started_at": _local_timestamp(),
                "ended_at": None,
                "duration_seconds": None,
                "settings": {
                    "output_language": output_language,
                },
                "audio": {
                    "path": "recording.wav",
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "device_name": device_name,
                },
                "screenshots": [],
            },
        )
        manifest.save()
        return manifest

    @classmethod
    def load(cls, session_dir: Path) -> "SessionManifest":
        session_dir = Path(session_dir)
        with (session_dir / MANIFEST_FILENAME).open("r", encoding="utf-8") as file:
            data = json.load(file)
        return cls(session_dir, data)

    @property
    def data(self) -> dict:
        with self._lock:
            return deepcopy(self._data)

    @property
    def status(self) -> str:
        with self._lock:
            return self._data["status"]

    @property
    def next_screenshot_id(self) -> int:
        with self._lock:
            ids = [item["id"] for item in self._data["screenshots"]]
            return max(ids, default=0) + 1

    def add_screenshot(
        self,
        screenshot_id: int,
        screenshot_path: Path,
        elapsed_seconds: float,
    ) -> None:
        if screenshot_id < 1:
            raise ValueError("Screenshot IDs must be positive integers.")
        if elapsed_seconds < 0:
            raise ValueError("Screenshot elapsed time cannot be negative.")

        with self._lock:
            if any(item["id"] == screenshot_id for item in self._data["screenshots"]):
                raise ValueError(f"Screenshot ID {screenshot_id} already exists.")

            self._data["screenshots"].append(
                {
                    "id": screenshot_id,
                    "path": self._relative_path(screenshot_path),
                    "elapsed_seconds": round(float(elapsed_seconds), 3),
                    "captured_at": _local_timestamp(),
                }
            )
            self._save_unlocked()

    def complete(self, audio_path: Path, duration_seconds: float) -> None:
        with self._lock:
            self._data["status"] = "completed"
            self._data["ended_at"] = _local_timestamp()
            self._data["duration_seconds"] = round(max(0.0, float(duration_seconds)), 3)
            self._data["audio"]["path"] = self._relative_path(audio_path)
            self._save_unlocked()

    def start_processing_stage(self, stage: str, **details) -> None:
        with self._lock:
            processing = self._data.setdefault("processing", {})
            processing[stage] = {
                "status": "running",
                "started_at": _local_timestamp(),
                **details,
            }
            self._save_unlocked()

    def complete_processing_stage(
        self,
        stage: str,
        output_path: Path,
        **details,
    ) -> None:
        with self._lock:
            processing = self._data.setdefault("processing", {})
            stage_data = processing.setdefault(stage, {})
            stage_data.update(
                {
                    "status": "completed",
                    "completed_at": _local_timestamp(),
                    "output_path": self._relative_path(output_path),
                    **details,
                }
            )
            stage_data.pop("error", None)
            self._save_unlocked()

    def fail_processing_stage(self, stage: str, message: str, **details) -> None:
        with self._lock:
            processing = self._data.setdefault("processing", {})
            stage_data = processing.setdefault(stage, {})
            stage_data.update(
                {
                    "status": "failed",
                    "failed_at": _local_timestamp(),
                    "error": message,
                    **details,
                }
            )
            self._save_unlocked()

    def resolve_artifact_path(self, artifact_path: str | Path) -> Path:
        """Resolve corrected and early relative-path manifest formats safely."""
        path = Path(artifact_path)
        session_root = self.session_dir.resolve()

        candidates = [path.resolve()] if path.is_absolute() else [
            (self.session_dir / path).resolve(),
            path.resolve(),
        ]

        for candidate in candidates:
            try:
                candidate.relative_to(session_root)
            except ValueError:
                continue
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Session artifact does not exist inside the session directory: {artifact_path}"
        )

    def fail(self, message: str) -> None:
        with self._lock:
            self._data["status"] = "failed"
            self._data["ended_at"] = _local_timestamp()
            self._data["error"] = message
            self._save_unlocked()

    def save(self) -> None:
        with self._lock:
            self._save_unlocked()

    def _relative_path(self, path: Path) -> str:
        path = Path(path)
        session_root = self.session_dir.resolve()

        if path.is_absolute():
            absolute_path = path.resolve()
        else:
            cwd_candidate = path.resolve()
            try:
                cwd_candidate.relative_to(session_root)
                absolute_path = cwd_candidate
            except ValueError:
                absolute_path = (self.session_dir / path).resolve()

        try:
            relative = absolute_path.relative_to(session_root)
        except ValueError as exc:
            raise ValueError("Session artifacts must stay inside the session directory.") from exc

        return relative.as_posix()

    def _save_unlocked(self) -> None:
        temporary_path = self.path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(self._data, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary_path, self.path)
