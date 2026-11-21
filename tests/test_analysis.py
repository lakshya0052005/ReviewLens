"""Unit tests for analysis module: schema validation, batching, caching, fallback, and R012."""

import pytest
from unittest.mock import AsyncMock
from pydantic import ValidationError

from app.analysis import (
    ReviewAnalysis,
    BatchAnalysisResponse,
    parse_and_validate_response,
    process_batch,
    is_rating_mismatch,
)
from app.cache import init_db
from app.cleaning import CleanedReview


@pytest.fixture
def test_db(tmp_path):
    db_file = str(tmp_path / "test_analysis.db")
    init_db(db_file)
    return db_file


# ---------------------------------------------------------------------------
# Schema Validation Tests
# ---------------------------------------------------------------------------

def test_pydantic_schema_accepts_valid_payload():
    """Verify standard valid output passes Pydantic validation."""
    data = {
        "review_id": "R001",
        "sentiment": "positive",
        "topics": ["quality", "price"],
        "summary": "Colors did not fade after washing.",
        "needs_reply": False,
    }
    obj = ReviewAnalysis.model_validate(data)
    assert obj.review_id == "R001"
    assert obj.sentiment == "positive"
    assert obj.topics == ["quality", "price"]
    assert obj.needs_reply is False


def test_pydantic_schema_rejects_invalid_sentiment():
    """Verify arbitrary sentiment strings are rejected by Literal type."""
    data = {
        "review_id": "R001",
        "sentiment": "awesome",  # Invalid!
        "topics": ["quality"],
        "summary": "Great item.",
        "needs_reply": False,
    }
    with pytest.raises(ValidationError):
        ReviewAnalysis.model_validate(data)


def test_pydantic_schema_rejects_empty_topics():
    """Verify empty topics list is rejected."""
    data = {
        "review_id": "R001",
        "sentiment": "positive",
        "topics": [],  # Invalid: min_length=1
        "summary": "Great item.",
        "needs_reply": False,
    }
    with pytest.raises(ValidationError):
        ReviewAnalysis.model_validate(data)


def test_pydantic_schema_rejects_unknown_topic():
    """Verify unknown topic names outside the allowed set are rejected."""
    data = {
        "review_id": "R001",
        "sentiment": "positive",
        "topics": ["durability"],  # Invalid: not in allowed topics list
        "summary": "Great item.",
        "needs_reply": False,
    }
    with pytest.raises(ValidationError):
        ReviewAnalysis.model_validate(data)


# ---------------------------------------------------------------------------
# Prompt Injection Defence Test (R012)
# ---------------------------------------------------------------------------

def test_r012_injection_defense():
    """Verify R012 with prompt injection is correctly classified as negative with needs_reply=True."""
    # Simulated model response adhering strictly to our system instructions
    raw_response = """{
      "reviews": [
        {
          "review_id": "R012",
          "sentiment": "negative",
          "topics": ["quality"],
          "summary": "Left earbud stopped working after two days.",
          "needs_reply": true
        }
      ]
    }"""
    results = parse_and_validate_response(raw_response, {"R012"})
    assert "R012" in results
    r012 = results["R012"]
    assert r012.sentiment == "negative"
    assert r012.needs_reply is True
    assert "quality" in r012.topics


# ---------------------------------------------------------------------------
# Rating Mismatch Tests
# ---------------------------------------------------------------------------

def test_rating_mismatch_detection():
    """Verify 5-star negative and 1-star positive are flagged as mismatches."""
    # R008: 5 stars but negative text
    assert is_rating_mismatch(5, "negative") is True
    # R003: 1 star sarcasm with positive sentiment
    assert is_rating_mismatch(1, "positive") is True

    # Standard aligned ratings
    assert is_rating_mismatch(5, "positive") is False
    assert is_rating_mismatch(1, "negative") is False

    # Unknown rating (None) should not flag mismatch
    assert is_rating_mismatch(None, "negative") is False


# ---------------------------------------------------------------------------
# Batching, Caching, and Fallback Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_batch_caching(test_db):
    """Verify uncached reviews call the LLM and the second run hits the cache with 0 calls."""
    review1 = CleanedReview(
        review_id="R001",
        product="Bedsheet",
        rating=5,
        rating_status="valid",
        date="2026-10-02",
        raw_text="Super soft fabric",
        cleaned_text="Super soft fabric",
        normalized_text="super soft fabric",
        clean_status="ok",
        injection_suspected=False,
    )

    mock_llm = AsyncMock()
    mock_llm.generate_json.return_value = """{
      "reviews": [
        {
          "review_id": "R001",
          "sentiment": "positive",
          "topics": ["quality"],
          "summary": "Super soft fabric.",
          "needs_reply": false
        }
      ]
    }"""

    # First run: uncached -> 1 LLM call
    results1, calls1, hits1 = await process_batch([review1], mock_llm, db_path=test_db)
    assert calls1 == 1
    assert hits1 == 0
    assert results1["R001"].sentiment == "positive"
    assert mock_llm.generate_json.call_count == 1

    # Second run: cached -> 0 LLM calls, 1 cache hit
    results2, calls2, hits2 = await process_batch([review1], mock_llm, db_path=test_db)
    assert calls2 == 0
    assert hits2 == 1
    assert results2["R001"].sentiment == "positive"
    # Call count remained 1 (no new LLM calls made)
    assert mock_llm.generate_json.call_count == 1


@pytest.mark.asyncio
async def test_retry_once_on_malformed_json(test_db):
    """Verify batch retries once if first response is invalid JSON."""
    review = CleanedReview(
        review_id="R002",
        product="Bedsheet",
        rating=2,
        rating_status="valid",
        date="2026-10-03",
        raw_text="Wrong size",
        cleaned_text="Wrong size",
        normalized_text="wrong size",
        clean_status="ok",
        injection_suspected=False,
    )

    mock_llm = AsyncMock()
    # Call 1 returns malformed text, Call 2 returns valid JSON
    mock_llm.generate_json.side_effect = [
        "not valid json at all",
        """{
          "reviews": [
            {
              "review_id": "R002",
              "sentiment": "negative",
              "topics": ["size_fit"],
              "summary": "Wrong size received.",
              "needs_reply": true
            }
          ]
        }""",
    ]

    results, calls, hits = await process_batch([review], mock_llm, db_path=test_db)
    assert calls == 1
    assert results["R002"].sentiment == "negative"
    assert mock_llm.generate_json.call_count == 2


@pytest.mark.asyncio
async def test_batch_splitting_fallback(test_db):
    """Verify that if a batch fails twice, it splits into individual reviews (Upgrade A.6)."""
    revA = CleanedReview(
        review_id="RA",
        product="Earbuds",
        rating=1,
        rating_status="valid",
        date="2026-10-01",
        raw_text="Broken earbud",
        cleaned_text="Broken earbud",
        normalized_text="broken earbud",
        clean_status="ok",
        injection_suspected=False,
    )
    revB = CleanedReview(
        review_id="RB",
        product="Earbuds",
        rating=5,
        rating_status="valid",
        date="2026-10-01",
        raw_text="Great sound",
        cleaned_text="Great sound",
        normalized_text="great sound",
        clean_status="ok",
        injection_suspected=False,
    )

    mock_llm = AsyncMock()
    # 1st and 2nd batch calls fail with bad JSON
    # 3rd call (single revA) succeeds
    # 4th call (single revB) succeeds
    mock_llm.generate_json.side_effect = [
        "bad batch 1",
        "bad batch 2",
        """{"reviews": [{"review_id": "RA", "sentiment": "negative", "topics": ["quality"], "summary": "Broken.", "needs_reply": true}]}""",
        """{"reviews": [{"review_id": "RB", "sentiment": "positive", "topics": ["quality"], "summary": "Great.", "needs_reply": false}]}""",
    ]

    results, calls, hits = await process_batch([revA, revB], mock_llm, db_path=test_db)
    # Batch failed twice -> triggered fallback -> 2 single calls
    assert calls == 3  # 1 initial batch attempt + 2 individual single attempts
    assert results["RA"].sentiment == "negative"
    assert results["RB"].sentiment == "positive"


@pytest.mark.asyncio
async def test_unrecoverable_failure_marks_analysis_failed(test_db):
    """Verify that persistently failing reviews receive 'analysis_failed' without crashing."""
    rev = CleanedReview(
        review_id="R_FAIL",
        product="Unknown",
        rating=3,
        rating_status="valid",
        date="2026-10-01",
        raw_text="Weird text",
        cleaned_text="Weird text",
        normalized_text="weird text",
        clean_status="ok",
        injection_suspected=False,
    )

    mock_llm = AsyncMock()
    # Always return broken JSON
    mock_llm.generate_json.return_value = "never valid json"

    results, calls, hits = await process_batch([rev], mock_llm, db_path=test_db)
    assert results["R_FAIL"] == "analysis_failed"
