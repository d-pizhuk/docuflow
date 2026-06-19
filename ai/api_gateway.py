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

# Self-hosted OpenAI-compatible vLLM endpoint. TLS is terminated by the server
# (https://), which satisfies NfReq9; no key is required on this deployment.
DEFAULT_BASE_URL = "https://vllm-api.scch.at/v1/"
DEFAULT_API_KEY = "EMPTY"

# Per-request wall-clock budget. NfReq2 allows 90 s end to end for 10 steps
# (LLM + later VLM); the structuring call takes the larger share.
REQUEST_TIMEOUT_S = 60.0

# Transport-level retries for transient failures (network blip, 5xx, 429),
# per design Q4: "retries up to 3 times with exponential backoff".
MAX_RETRIES = 3
BACKOFF_BASE_S = 1.5


class ApiGatewayError(RuntimeError):
    """Raised when retries are exhausted or a non-retryable error occurs.

    The caller is expected to fall back to the manual path: keep the raw
    transcript and screenshots on disk so the user can finish by hand.
    """


@dataclass
class ApiGateway:
    """Thin wrapper around the OpenAI client that owns retry/backoff/timeout.

    Reusable by both the LLM Step Structurer (Req6) and the future VLM (Req7),
    since both call the same vLLM server.
    """

    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    timeout_s: float = REQUEST_TIMEOUT_S
    max_retries: int = MAX_RETRIES

    def __post_init__(self):
        # max_retries=0 on the SDK: we drive retry/backoff ourselves so we can
        # log each attempt and checkpoint on final failure, rather than letting
        # the SDK retry silently.
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout_s,
            max_retries=0,
        )

    def chat(self, *, model, messages, tools=None, tool_choice="auto", temperature=0.3):
        """Single chat completion with retries. Returns the message object
        (`completion.choices[0].message`). Raises ApiGatewayError on failure."""
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
                # 5xx is transient and worth retrying; other 4xx (bad model
                # name, malformed request) will not improve on retry.
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