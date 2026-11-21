"""Unit tests for LLMClient, retry policy, backoff, and circuit breaker."""

import pytest
import httpx
from unittest.mock import AsyncMock

from app.llm_client import (
    LLMClient,
    CircuitBreaker,
    CircuitBreakerOpenError,
    NonRetryableLLMError,
    RetryableLLMError,
    strip_markdown_fences,
)


def test_strip_markdown_fences():
    """Verify markdown fences are cleanly stripped from model output."""
    raw_with_fences = '```json\n{"sentiment": "positive"}\n```'
    assert strip_markdown_fences(raw_with_fences) == '{"sentiment": "positive"}'

    raw_plain = '{"sentiment": "positive"}'
    assert strip_markdown_fences(raw_plain) == '{"sentiment": "positive"}'

    raw_no_lang = '```\n{"sentiment": "positive"}\n```'
    assert strip_markdown_fences(raw_no_lang) == '{"sentiment": "positive"}'


@pytest.mark.asyncio
async def test_non_retryable_error_fails_fast():
    """Verify 401 Unauthorized fails immediately without retrying."""
    sleep_calls = []

    async def fake_sleep(duration: float):
        sleep_calls.append(duration)

    # Mock transport returning 401 Unauthorized
    transport = httpx.MockTransport(
        lambda request: httpx.Response(401, json={"error": "Invalid API key"})
    )

    client = LLMClient(
        api_key="bad_key",
        max_retries=3,
        sleep_fn=fake_sleep,
    )

    async with httpx.AsyncClient(transport=transport) as http_client:
        with pytest.raises(NonRetryableLLMError) as exc_info:
            await client.generate_json("system", "prompt", http_client=http_client)

    assert "401" in str(exc_info.value)
    # Failed fast: zero retry sleep calls occurred
    assert len(sleep_calls) == 0


@pytest.mark.asyncio
async def test_retry_on_429_with_backoff_and_eventual_success():
    """Verify client retries on 429 rate-limit and succeeds on attempt 3."""
    attempts = 0
    sleep_durations = []

    async def fake_sleep(duration: float):
        sleep_durations.append(duration)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(
                429,
                headers={"Retry-After": "2"},
                json={"error": "Rate limit exceeded"},
            )
        # Succeeded on 3rd attempt
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": '{"result": "ok"}'}]}}
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = LLMClient(max_retries=3, sleep_fn=fake_sleep)

    async with httpx.AsyncClient(transport=transport) as http_client:
        result = await client.generate_json("system", "prompt", http_client=http_client)

    assert result == '{"result": "ok"}'
    assert attempts == 3
    # Two retries happened before success
    assert len(sleep_durations) == 2
    # Retry-After of 2 seconds was honored
    assert all(d >= 2.0 for d in sleep_durations)


@pytest.mark.asyncio
async def test_circuit_breaker_trips_and_blocks():
    """Verify circuit breaker opens after consecutive failures and blocks further requests."""
    cb = CircuitBreaker(threshold=3, cooldown_seconds=60.0)
    fake_sleep = AsyncMock()

    # Always return 500 server error
    transport = httpx.MockTransport(
        lambda request: httpx.Response(500, json={"error": "Internal Error"})
    )
    client = LLMClient(max_retries=1, circuit_breaker=cb, sleep_fn=fake_sleep)

    async with httpx.AsyncClient(transport=transport) as http_client:
        # Request 1 fails
        with pytest.raises(RetryableLLMError):
            await client.generate_json("system", "prompt", http_client=http_client)
        assert cb.state == "CLOSED"

        # Request 2 fails
        with pytest.raises(RetryableLLMError):
            await client.generate_json("system", "prompt", http_client=http_client)
        assert cb.state == "CLOSED"

        # Request 3 fails -> trips circuit breaker to OPEN
        with pytest.raises(RetryableLLMError):
            await client.generate_json("system", "prompt", http_client=http_client)
        assert cb.state == "OPEN"

        # Request 4 should be immediately rejected by the circuit breaker without hitting the API
        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await client.generate_json("system", "prompt", http_client=http_client)
        assert "provider_unavailable" in str(exc_info.value)


def test_circuit_breaker_cooldown_and_recovery():
    """Verify circuit breaker transitions to HALF_OPEN after cooldown and resets on success."""
    cb = CircuitBreaker(threshold=2, cooldown_seconds=10.0)

    # 2 failures trip to OPEN
    cb.record_failure(current_time=100.0)
    cb.record_failure(current_time=101.0)
    assert cb.state == "OPEN"

    # Before cooldown expires (at 105s), it remains OPEN
    with pytest.raises(CircuitBreakerOpenError):
        cb.can_execute(current_time=105.0)

    # After cooldown expires (at 112s), it allows execution in HALF_OPEN
    assert cb.can_execute(current_time=112.0) is True
    assert cb.state == "HALF_OPEN"

    # Success in HALF_OPEN closes the breaker
    cb.record_success()
    assert cb.state == "CLOSED"
    assert cb.failure_count == 0
