import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ai.api_gateway import ApiGateway, ApiGatewayError
from ai.step_structurer import StructuredDoc

logger = logging.getLogger(__name__)

# A vision-capable model on the vLLM server (`curl <base_url>/models`).
# Llama-4-Scout is natively multimodal. gemma-4-31B-it is the listed fallback.
DEFAULT_VLM_MODEL = "RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16"
# Alternative with vision: "RedHatAI/gemma-4-31B-it-FP8-Dynamic"

TEMPERATURE = 0.2              # captions should be factual, not creative
MAX_JSON_REPAIR_ATTEMPTS = 1   # one corrective re-prompt, then give up on that image
MAX_IMAGE_DIM = 1536           # downscale large screenshots before sending
DESCRIBE_CONCURRENCY = 3       # parallel VLM calls (helps the NfReq2 90 s budget)


@dataclass
class ScreenshotDescription:
    filename: str
    title: str
    description: str


_SYSTEM_PROMPT = (
    "You are a technical writer creating step-by-step software documentation. "
    "You are shown ONE screenshot that illustrates a single step of a workflow. "
    "Look only at what is actually visible in the image — never invent UI text, "
    "buttons, or values that you cannot see.\n\n"
    "Respond with ONLY a JSON object (no markdown, no code fences, no extra text) "
    "with exactly these fields:\n"
    '  "title": a 3-7 word label for what the screenshot shows.\n'
    '  "description": 1-2 sentences describing the relevant UI element or action '
    "visible in the screenshot, tied to the step."
)


class ScreenshotDescriber:
    """Req7: uses a vision-language model to write a title + short description for
    each screenshot, given the image plus the step it belongs to as context."""

    def __init__(self, gateway: ApiGateway | None = None, model: str = DEFAULT_VLM_MODEL):
        self._gateway = gateway or ApiGateway()
        self._model = model

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def describe_all(self, doc: StructuredDoc, screenshot_dir: Path) -> dict[str, ScreenshotDescription]:
        """Describe every screenshot referenced by the document, in parallel.

        Returns {filename: ScreenshotDescription}. Robust: a single image failing
        (VLM error, missing file, bad JSON) is logged and skipped, never aborting
        the others — the merger simply omits the missing description.
        """
        targets = []
        seen: set[str] = set()
        for step in doc.steps:
            if step.screenshot and step.screenshot not in seen:
                seen.add(step.screenshot)
                targets.append(step)

        if not targets:
            return {}

        results: dict[str, ScreenshotDescription] = {}

        def _work(step):
            path = Path(screenshot_dir) / step.screenshot
            return step.screenshot, self.describe(
                path,
                step_title=step.title,
                step_instruction=step.instruction,
                doc_title=doc.title,
            )

        workers = min(DESCRIBE_CONCURRENCY, len(targets))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vlm") as ex:
            futures = [ex.submit(_work, s) for s in targets]
            for fut in as_completed(futures):
                try:
                    filename, desc = fut.result()
                    results[filename] = desc
                    print(f"[vlm] described {filename}: '{desc.title}'", flush=True)
                except Exception as exc:  # noqa: BLE001 — degrade gracefully per image
                    logger.warning("VLM description failed for a screenshot: %s", exc)
                    print(f"[vlm] !! failed to describe a screenshot: {exc}", flush=True)

        return results

    def describe(self, image_path: Path, *, step_title: str, step_instruction: str,
                 doc_title: str) -> ScreenshotDescription:
        """Describe a single screenshot. Raises on unrecoverable failure."""
        media_type, b64 = self._encode_image(Path(image_path))
        data_url = f"data:{media_type};base64,{b64}"

        user_text = (
            f"Guide: {doc_title}\n"
            f"Step: {step_title}\n"
            f"Step instruction: {step_instruction}\n\n"
            "Describe what this screenshot shows for this step."
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ]

        last_raw = ""
        for attempt in range(MAX_JSON_REPAIR_ATTEMPTS + 1):
            msg = self._gateway.chat(
                model=self._model,
                messages=messages,
                temperature=TEMPERATURE,
            )
            raw = self._extract_content(msg)
            last_raw = raw
            try:
                return self._parse(raw, Path(image_path).name)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                if attempt >= MAX_JSON_REPAIR_ATTEMPTS:
                    raise ApiGatewayError(
                        f"VLM returned unparseable description: {exc}"
                    ) from exc
                messages.append({"role": "assistant", "content": f"(invalid output: {last_raw[:300]})"})
                messages.append({
                    "role": "user",
                    "content": (
                        f"That was not valid JSON: {exc}. Reply with ONLY a JSON object "
                        'with "title" and "description" fields.'
                    ),
                })

        raise ApiGatewayError("VLM description unreachable")  # pragma: no cover

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse(raw: str, filename: str) -> ScreenshotDescription:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("top-level JSON is not an object")
        title = str(data.get("title") or "").strip()
        desc = str(data.get("description") or "").strip()
        if not title or not desc:
            raise ValueError("missing title/description")
        return ScreenshotDescription(filename=filename, title=title, description=desc)

    @staticmethod
    def _extract_content(msg) -> str:
        content = (msg.content or "").strip()
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.lower().startswith("json"):
                content = content[4:].strip()
        return content

    @staticmethod
    def _encode_image(path: Path) -> tuple[str, str]:
        """Return (media_type, base64). Downscales large images via Pillow when
        available; falls back to the original bytes if Pillow is missing."""
        raw = path.read_bytes()
        media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        try:
            import io
            from PIL import Image

            with Image.open(io.BytesIO(raw)) as im:
                im = im.convert("RGB")
                if max(im.size) > MAX_IMAGE_DIM:
                    ratio = MAX_IMAGE_DIM / max(im.size)
                    im = im.resize((max(1, int(im.width * ratio)), max(1, int(im.height * ratio))))
                buf = io.BytesIO()
                im.save(buf, format="PNG", optimize=True)
                raw = buf.getvalue()
                media = "image/png"
        except Exception:
            logger.debug("Pillow unavailable or resize failed; sending original image bytes")

        return media, base64.b64encode(raw).decode("ascii")