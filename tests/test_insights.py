"""Unit tests for insights aggregation and summary statistics."""

import pytest
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

from app.insights import (
    compute_summary_stats,
    compute_per_product_insights,
    deterministic_top_insights,
    generate_top_insights,
    get_job_insights,
)
from app.cache import init_db, create_job, save_review_result
from app.main import app


@pytest.fixture
def test_db(tmp_path):
    db_file = str(tmp_path / "test_insights.db")
    init_db(db_file)
    return db_file


def test_compute_summary_stats_empty():
    """Verify empty reviews list returns clean zeroed stats."""
    stats = compute_summary_stats([])
    assert stats["total_reviews"] == 0
    assert stats["positive_pct"] == 0.0
    assert stats["rating_agreement_pct"] == 100.0


def test_compute_summary_stats_and_agreement():
    """Verify percentage calculations and mismatch detection in summary stats."""
    reviews = [
        {"sentiment": "positive", "rating": 5, "mismatch_flag": False, "needs_reply": False, "clean_status": "ok"},
        {"sentiment": "positive", "rating": 4, "mismatch_flag": False, "needs_reply": False, "clean_status": "ok"},
        # R008 mismatch: 5 star rating, negative sentiment
        {"sentiment": "negative", "rating": 5, "mismatch_flag": True, "needs_reply": True, "clean_status": "ok"},
        # Neutral review
        {"sentiment": "neutral", "rating": 3, "mismatch_flag": False, "needs_reply": False, "clean_status": "ok"},
        # Skipped review
        {"sentiment": None, "rating": None, "mismatch_flag": False, "needs_reply": False, "clean_status": "skipped_empty"},
        # Duplicate review
        {"sentiment": "positive", "rating": 5, "mismatch_flag": False, "needs_reply": False, "clean_status": "duplicate_of:R001"},
    ]

    stats = compute_summary_stats(reviews)
    assert stats["total_reviews"] == 6
    assert stats["analyzed_count"] == 5  # 3 pos + 1 neu + 1 neg
    assert stats["positive_count"] == 3
    assert stats["neutral_count"] == 1
    assert stats["negative_count"] == 1
    assert stats["positive_pct"] == 60.0
    assert stats["needs_reply_count"] == 1
    assert stats["mismatch_count"] == 1
    # 5 rated reviews with 1 mismatch = 80.0% agreement
    assert stats["rating_agreement_pct"] == 80.0
    assert stats["skipped_empty_count"] == 1
    assert stats["duplicates_count"] == 1


def test_compute_per_product_insights():
    """Verify product-level grouping and top topics."""
    reviews = [
        {"product": "Earbuds", "sentiment": "negative", "topics": ["quality", "support"], "needs_reply": True},
        {"product": "Earbuds", "sentiment": "positive", "topics": ["quality"], "needs_reply": False},
        {"product": "Mug", "sentiment": "positive", "topics": ["packaging"], "needs_reply": False},
    ]

    products = compute_per_product_insights(reviews)
    assert len(products) == 2

    earbuds = next(p for p in products if p["product"] == "Earbuds")
    assert earbuds["total_reviews"] == 2
    assert earbuds["positive_count"] == 1
    assert earbuds["negative_count"] == 1
    assert earbuds["positive_pct"] == 50.0
    assert earbuds["top_complaint_topic"] == "quality"

    mug = next(p for p in products if p["product"] == "Mug")
    assert mug["total_reviews"] == 1
    assert mug["positive_pct"] == 100.0
    assert mug["top_praise_topic"] == "packaging"


def test_deterministic_top_insights():
    """Verify deterministic fallback produces top 3 praises and complaints."""
    reviews = [
        {"sentiment": "positive", "summary": "Super soft fabric."},
        {"sentiment": "positive", "summary": "Beautiful mugs."},
        {"sentiment": "negative", "summary": "One mug arrived broken."},
    ]
    insights = deterministic_top_insights(reviews)
    assert len(insights["top_praises"]) == 3
    assert len(insights["top_complaints"]) == 3
    assert "Super soft fabric." in insights["top_praises"]
    assert "One mug arrived broken." in insights["top_complaints"]


@pytest.mark.asyncio
async def test_generate_top_insights_with_llm_and_cache(test_db):
    """Verify LLM phrasing call and that second call is cached."""
    reviews = [
        {"product": "Mug", "sentiment": "positive", "summary": "Beautiful mugs."},
        {"product": "Mug", "sentiment": "negative", "summary": "Broken handles."},
    ]
    job_id = "job-insights-test"

    mock_llm = AsyncMock()
    mock_llm.generate_json.return_value = """{
      "top_praises": ["Aesthetic design", "Good value", "Durable build"],
      "top_complaints": ["Fragile handles", "Packaging broken", "Slow support"]
    }"""

    # Call 1: makes LLM call and populates cache
    res1 = await generate_top_insights(reviews, llm_client=mock_llm, job_id=job_id, db_path=test_db)
    assert res1["top_praises"][0] == "Aesthetic design"
    assert res1["top_complaints"][0] == "Fragile handles"
    assert mock_llm.generate_json.call_count == 1

    # Call 2: hits cache (mock LLM not invoked)
    res2 = await generate_top_insights(reviews, llm_client=mock_llm, job_id=job_id, db_path=test_db)
    assert res2["top_praises"][0] == "Aesthetic design"
    assert mock_llm.generate_json.call_count == 1


@pytest.mark.asyncio
async def test_get_job_insights_endpoint(test_db, monkeypatch):
    """Verify GET /api/jobs/{job_id}/insights API endpoint."""
    monkeypatch.setattr("app.config.settings.DB_PATH", test_db)
    job_id = "job-endpoint-test"
    create_job(job_id, 1, db_path=test_db)
    save_review_result(
        job_id=job_id,
        review_id="R001",
        product="Bedsheet",
        rating=5,
        rating_status="valid",
        date="2026-10-02",
        raw_text="Soft",
        cleaned_text="Soft",
        clean_status="ok",
        injection_suspected=False,
        sentiment="positive",
        topics=["quality"],
        summary="Soft bedsheet",
        needs_reply=False,
        db_path=test_db,
    )

    client = TestClient(app)
    response = client.get(f"/api/jobs/{job_id}/insights")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == job_id
    assert data["summary"]["total_reviews"] == 1
    assert data["summary"]["positive_pct"] == 100.0
    assert len(data["per_product"]) == 1
    assert data["per_product"][0]["product"] == "Bedsheet"
    assert len(data["top_praises"]) == 3
    assert len(data["top_complaints"]) == 3
