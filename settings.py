# settings.py
import json
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class Settings:
    consent_given: bool = False
    output_dir: str = str(Path.home() / "DocuFlow" / "sessions")
    documentation_language: str = "English"

    # Cloud AI Config
    llm_model: str = "casperhansen/llama-3.3-70b-instruct-awq"
    vlm_model: str = "RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16"
    api_base_url: str = "https://vllm-api.scch.at/v1/"
    api_key: str = "EMPTY"

    @classmethod
    def load(cls) -> "Settings":
        path = Path.home() / "DocuFlow" / "settings.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Filter to only known keys to prevent crashes on schema updates
                valid_data = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
                return cls(**valid_data)
            except Exception:
                pass
        return cls()

    def save(self):
        path = Path.home() / "DocuFlow" / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=4)
        except Exception:
            pass