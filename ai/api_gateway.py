# ai/api_gateway.py
import logging
import time
from dataclasses import dataclass

from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
    RateLimitError,
    InternalServerError,
)

logger = logging.getLogger(__name__)

# Per-request wall-clock budget. NfReq2 allows 90 s end to end for 10 steps
# (LLM + later VLM); the structuring call takes the larger share.
REQUEST_TIMEOUT_S = 60.0

# Transport-level retries for transient failures (network blip, 5xx, 429),
# per design Q4: "retries up to 3 times with exponential backoff".
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.5


class ApiGatewayError(RuntimeError):
    """Raised when retries are exhausted or a non-retryable error occurs."""


@dataclass
class ApiGateway:
    """Thin wrapper around the OpenAI client that owns retry/backoff/timeout."""

    base_url: str
    api_key: str
    timeout_s: float = REQUEST_TIMEOUT_S
    max_retries: int = MAX_RETRIES

    def __post_init__(self):
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout_s,
            max_retries=0,
        )

    def chat(self, *, model, messages, tools=None, tool_choice="auto", temperature=0.3):
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs = dict(model=model, messages=messages, temperature=temperature)
                if tools is not None:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice
                completion = self._client.chat.completions.create(**kwargs)
                return completion.choices[0].message

            except (APIConnectionError, APITimeoutError, RateLimitError,
                    InternalServerError) as exc:
                last_exc = exc
                wait = BACKOFF_BASE_S * (2 ** (attempt - 1))
                logger.warning("LLM call failed (attempt %d/%d): %s -> retry in %.1fs",
                               attempt, self.max_retries, exc, wait)
                print(f"[gateway] attempt {attempt}/{self.max_retries} failed: "
                      f"{exc} -> retry in {wait:.1f}s", flush=True)
                if attempt < self.max_retries:
                    time.sleep(wait)

            except APIStatusError as exc:
                if 500 <= exc.status_code < 600:
                    last_exc = exc
                    wait = BACKOFF_BASE_S * (2 ** (attempt - 1))
                    if attempt < self.max_retries:
                        time.sleep(wait)
                    continue
                raise ApiGatewayError(
                    f"LLM request rejected ({exc.status_code}): {exc}"
                ) from exc

        raise ApiGatewayError(
            f"LLM unreachable after {self.max_retries} attempts"
        ) from last_exc