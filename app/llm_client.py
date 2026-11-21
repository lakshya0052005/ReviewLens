"""LLM Client module for ReviewLens.

Isolates all external LLM API communication. Handles:
- HTTP requests to Google Gemini REST API (JSON mode).
- Exponential backoff with full jitter and Retry-After header support.
- Fast-failure on non-retryable client errors (400, 401).
- Circuit breaker pattern to stop calling the API when provider is failing.
- Structured logging without exposing secrets or user review text.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from typing import Any, Callable, Coroutine
import httpx

from app.config import settings

# Setup structured logger
logger = logging.getLogger("reviewlens.llm")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [llm_client] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class CircuitBreakerOpenError(Exception):
    """Raised when consecutive failures have tripped the circuit breaker."""
    pass


class NonRetryableLLMError(Exception):
    """Raised on client or authentication errors (400, 401, 403) that must fail fast."""
    pass


class RetryableLLMError(Exception):
    """Raised on transient network or server errors (429, 5xx, timeouts)."""
    pass


class CircuitBreaker:
    """Circuit breaker around LLM calls to prevent swamping failing APIs.

    Why: If the Gemini API is down or quota is completely exhausted, continuing to fire
    requests wastes time and quota. The circuit breaker trips after N consecutive failures,
    cooldowns for a period, and lets remaining background tasks fail fast with 'provider_unavailable'.
    """

    def __init__(self, threshold: int = 3, cooldown_seconds: float = 30.0):
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.failure_count = 0
        self.state = "CLOSED"  # "CLOSED", "OPEN", "HALF_OPEN"
        self.opened_at = 0.0

    def can_execute(self, current_time: float | None = None) -> bool:
        now = time.time() if current_time is None else current_time
        if self.state == "CLOSED":
            return True

        if self.state == "OPEN":
            if now - self.opened_at >= self.cooldown_seconds:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker transitioned from OPEN to HALF_OPEN (probing API)")
                return True
            raise CircuitBreakerOpenError("Circuit breaker is OPEN: provider_unavailable")

        # HALF_OPEN allows a single probe through
        return True

    def record_success(self) -> None:
        if self.state != "CLOSED":
            logger.info("Circuit breaker closed after successful probe")
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self, current_time: float | None = None) -> None:
        self.failure_count += 1
        now = time.time() if current_time is None else current_time
        if self.failure_count >= self.threshold:
            self.state = "OPEN"
            self.opened_at = now
            logger.warning(
                "Circuit breaker TRIPPED to OPEN after %d consecutive failures (cooldown: %.1fs)",
                self.failure_count,
                self.cooldown_seconds,
            )


def strip_markdown_fences(text: str) -> str:
    """Remove surrounding markdown code blocks (```json ... ```) if present."""
    trimmed = text.strip()
    if trimmed.startswith("```"):
        # Remove opening fence (``` or ```json)
        trimmed = re.sub(r"^```[a-zA-Z]*\n?", "", trimmed)
        # Remove closing fence
        trimmed = re.sub(r"\n?```$", "", trimmed)
    return trimmed.strip()


class LLMClient:
    """Client for calling LLM providers via HTTP with resilience mechanisms."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        max_retries: int | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        sleep_fn: Callable[[float], Coroutine[Any, Any, None]] | None = None,
    ):
        self.api_key = api_key if api_key is not None else settings.GEMINI_API_KEY
        self.model_name = model_name if model_name is not None else settings.MODEL_NAME
        self.max_retries = max_retries if max_retries is not None else settings.MAX_RETRIES
        self.circuit_breaker = (
            circuit_breaker
            if circuit_breaker is not None
            else CircuitBreaker(
                threshold=settings.CIRCUIT_BREAKER_THRESHOLD,
                cooldown_seconds=settings.CIRCUIT_BREAKER_COOLDOWN_SECONDS,
            )
        )
        # Injectable sleep for zero-wait unit testing
        self.sleep_fn = sleep_fn if sleep_fn is not None else asyncio.sleep

    def _get_endpoint_url(self) -> str:
        return f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"

    async def generate_json(
        self,
        system_instruction: str,
        user_prompt: str,
        http_client: httpx.AsyncClient | None = None,
    ) -> str:
        """Execute a prompt against Gemini with retry/backoff and circuit breaker.

        Returns raw JSON string response from the model.
        """
        # 1. Check circuit breaker status first
        self.circuit_breaker.can_execute()

        url = self._get_endpoint_url()
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        }

        # Determine whether to use caller-provided client or create a short-lived one
        should_close = False
        client = http_client
        if client is None:
            client = httpx.AsyncClient(timeout=30.0)
            should_close = True

        try:
            for attempt in range(1, self.max_retries + 1):
                start_time = time.time()
                try:
                    response = await client.post(url, headers=headers, json=payload)
                    latency = time.time() - start_time

                    # Check for non-retryable authentication or bad request errors (400, 401, 403)
                    if response.status_code in (400, 401, 403):
                        logger.error(
                            "LLM Non-retryable error | model=%s attempt=%d latency=%.2fs status=%d",
                            self.model_name,
                            attempt,
                            latency,
                            response.status_code,
                        )
                        raise NonRetryableLLMError(
                            f"LLM API client error {response.status_code}: {response.text}"
                        )

                    # Check for rate-limiting (429) or server errors (5xx)
                    if response.status_code == 429 or response.status_code >= 500:
                        retry_after_header = response.headers.get("Retry-After")
                        logger.warning(
                            "LLM Transient error | model=%s attempt=%d latency=%.2fs status=%d retry_after=%s",
                            self.model_name,
                            attempt,
                            latency,
                            response.status_code,
                            retry_after_header,
                        )
                        raise RetryableLLMError(f"HTTP {response.status_code}")

                    response.raise_for_status()
                    data = response.json()

                    # Extract text content from Gemini's candidate output
                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise RetryableLLMError("No candidates returned by Gemini")

                    raw_text = candidates[0]["content"]["parts"][0]["text"]
                    cleaned_output = strip_markdown_fences(raw_text)

                    # Successful call: reset circuit breaker and log structured metric
                    self.circuit_breaker.record_success()
                    logger.info(
                        "LLM call succeeded | model=%s attempt=%d latency=%.2fs outcome=success",
                        self.model_name,
                        attempt,
                        latency,
                    )
                    return cleaned_output

                except (RetryableLLMError, httpx.TimeoutException, httpx.NetworkError) as err:
                    latency = time.time() - start_time
                    logger.warning(
                        "LLM call failed | model=%s attempt=%d/%d latency=%.2fs error=%s",
                        self.model_name,
                        attempt,
                        self.max_retries,
                        latency,
                        type(err).__name__,
                    )

                    if attempt == self.max_retries:
                        # Record failure on circuit breaker when all retries are exhausted
                        self.circuit_breaker.record_failure()
                        raise

                    # Calculate exponential backoff with full jitter
                    # Sleep is random between 0 and min(max_backoff, initial_backoff * 2^(attempt-1))
                    backoff_cap = min(
                        settings.MAX_BACKOFF_SECONDS,
                        settings.INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)),
                    )
                    sleep_duration = random.uniform(0.1, backoff_cap)

                    # Honour Retry-After header if present
                    if isinstance(err, RetryableLLMError) and "retry_after_header" in locals() and retry_after_header:
                        try:
                            sleep_duration = max(sleep_duration, float(retry_after_header))
                        except ValueError:
                            pass

                    await self.sleep_fn(sleep_duration)

            raise RetryableLLMError(f"Exhausted {self.max_retries} attempts")

        finally:
            if should_close:
                await client.aclose()
