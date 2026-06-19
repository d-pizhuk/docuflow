import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ai.api_gateway import ApiGateway, ApiGatewayError

logger = logging.getLogger(__name__)

# Pick any chat model your vLLM server has loaded (`curl <base_url>/models`).
# Llama-3.3-70B has very reliable tool calling; the 27-35B "Reasoning OFF"
# variants are faster and help meet NfReq2 (10 steps within 90 s).
DEFAULT_MODEL = "casperhansen/llama-3.3-70b-instruct-awq"
# Alternatives observed on the server:
#   "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
#   "Qwen/Qwen3.6-27B-FP8"
#   "Qwen/Qwen3.6-35B-A3B-FP8 - Reasoning OFF"   # fast, less verbose
#   "RedHatAI/gemma-4-31B-it-FP8-Dynamic"

TEMPERATURE = 0.3              # design Q2: low but non-zero
MAX_JSON_REPAIR_ATTEMPTS = 2   # design Q6: retry twice, then save raw + error

_TOOL_NAME = "emit_documentation"


# --------------------------------------------------------------------------- #
# Output data model (consumed later by Req7 VLM + Req8 JSON Doc Merger)
# --------------------------------------------------------------------------- #

@dataclass
class DocStep:
    title: str
    instruction: str
    screenshot: str | None = None   # screenshot filename placeholder, or None


@dataclass
class StructuredDoc:
    title: str
    steps: list[DocStep] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "title": self.title,
            "steps": [
                {"title": s.title, "instruction": s.instruction, "screenshot": s.screenshot}
                for s in self.steps
            ],
        }


# --------------------------------------------------------------------------- #
# Tool schema + prompt
# --------------------------------------------------------------------------- #

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": (
                "Return the finished step-by-step documentation built from the "
                "user's spoken transcript. Call this exactly once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short title for the whole guide.",
                    },
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {
                                    "type": "string",
                                    "description": "Imperative step heading, e.g. 'Open the Settings menu'.",
                                },
                                "instruction": {
                                    "type": "string",
                                    "description": "1-3 sentence instruction in clean written English.",
                                },
                                "screenshot": {
                                    "type": ["string", "null"],
                                    "description": (
                                        "Filename of the screenshot for this step, copied exactly "
                                        "from a [SCREENSHOT: ...] marker, or null if none."
                                    ),
                                },
                            },
                            "required": ["title", "instruction"],
                        },
                    },
                },
                "required": ["title", "steps"],
            },
        },
    }
]

_SYSTEM_PROMPT = (
    "You are a technical writer. You convert a raw, spoken-aloud transcript of "
    "someone demonstrating a computer task into clean, structured step-by-step "
    "documentation.\n\n"
    "Rules:\n"
    "- Group the rambling transcript into a small number of clear, ordered steps.\n"
    "- Rewrite spoken filler into concise written instructions. Do not invent "
    "steps that were not described.\n"
    "- The transcript contains [SCREENSHOT: filename.png] markers. Attach each "
    "marker to the step it belongs to by putting that exact filename in that "
    "step's \"screenshot\" field. Use each filename at most once. Steps with no "
    "screenshot use null.\n"
    "- Return the result ONLY by calling the emit_documentation function."
)


# --------------------------------------------------------------------------- #
# Structurer
# --------------------------------------------------------------------------- #

class StepStructurer:
    """Req6: turns an annotated transcript into structured JSON documentation
    steps via an OpenAI-compatible LLM endpoint (vLLM)."""

    def __init__(self, gateway: ApiGateway | None = None, model: str = DEFAULT_MODEL):
        self._gateway = gateway or ApiGateway()
        self._model = model

    def structure(
        self,
        annotated_transcript: str,
        *,
        valid_screenshots: list[str] | None = None,
        session_dir: Path | None = None,
    ) -> StructuredDoc:
        """Returns a StructuredDoc. Raises ApiGatewayError if the server is
        unreachable or the output can't be parsed after repair attempts."""
        if not annotated_transcript.strip():
            return StructuredDoc(title="Empty session", steps=[])

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n\n{annotated_transcript}"},
        ]

        last_raw = ""
        for attempt in range(MAX_JSON_REPAIR_ATTEMPTS + 1):
            msg = self._gateway.chat(
                model=self._model,
                messages=messages,
                tools=OPENAI_TOOLS,
                tool_choice="auto",
                temperature=TEMPERATURE,
            )
            raw = self._extract_arguments(msg)
            last_raw = raw
            try:
                doc = self._parse(raw, valid_screenshots)
                print(f"[structurer] ok: '{doc.title}' with {len(doc.steps)} step(s)", flush=True)
                return doc
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                logger.warning("Structuring parse failed (attempt %d): %s", attempt + 1, exc)
                print(f"[structurer] parse failed (attempt {attempt + 1}): {exc}", flush=True)
                if attempt >= MAX_JSON_REPAIR_ATTEMPTS:
                    break
                # design Q6: corrective re-prompt. Echo the bad output as plain
                # text (not as tool_calls) to keep the message list valid on
                # any server.
                messages.append({"role": "assistant", "content": f"(invalid output: {raw[:600]})"})
                messages.append({
                    "role": "user",
                    "content": (
                        f"That output was invalid: {exc}. Call emit_documentation "
                        f"again with arguments that exactly match the schema."
                    ),
                })

        # All repairs failed -> persist raw output for manual recovery (design Q6).
        if session_dir is not None:
            self._dump_raw(session_dir, last_raw)
        raise ApiGatewayError("LLM returned unparseable documentation after retries")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_arguments(msg) -> str:
        # Preferred: the model called our tool -> arguments is a JSON string.
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                if call.function and call.function.name == _TOOL_NAME:
                    return call.function.arguments or ""
            return tool_calls[0].function.arguments or ""

        # Fallback: tool_choice="auto" did not force a call; the model answered
        # with JSON in content. Strip ```json fences if present.
        content = (msg.content or "").strip()
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()
        return content

    @staticmethod
    def _parse(raw: str, valid_screenshots: list[str] | None) -> StructuredDoc:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("top-level JSON is not an object")

        title = str(data.get("title") or "Untitled guide").strip()
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("'steps' must be a non-empty array")

        allowed = set(valid_screenshots or [])
        used: set[str] = set()
        steps: list[DocStep] = []

        for i, s in enumerate(raw_steps):
            if not isinstance(s, dict):
                raise ValueError(f"step {i} is not an object")
            st = str(s.get("title") or "").strip()
            instr = str(s.get("instruction") or "").strip()
            if not st or not instr:
                raise ValueError(f"step {i} missing title/instruction")

            shot = s.get("screenshot")
            if shot is not None:
                shot = str(shot).strip() or None
                # Drop hallucinated or duplicate filenames; keep only real ones.
                if shot and allowed and shot not in allowed:
                    shot = None
                elif shot in used:
                    shot = None
                if shot:
                    used.add(shot)

            steps.append(DocStep(title=st, instruction=instr, screenshot=shot))

        return StructuredDoc(title=title, steps=steps)

    @staticmethod
    def _dump_raw(session_dir: Path, raw: str):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = session_dir / f"llm_raw_output_{ts}.txt"
        try:
            out.write_text(raw, encoding="utf-8")
            print(f"[structurer] saved raw LLM output -> {out}", flush=True)
        except Exception:
            logger.exception("Failed to write raw LLM output")