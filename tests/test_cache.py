"""Unit tests for SQLite cache and persistence layer."""

import pytest
from app.cache import (
    compute_cache_key,
    init_db,
    get_cached_analysis,
    set_cached_analysis,
    create_job,
    update_job_progress,
    get_job,
    save_review_result,
    get_job_reviews,
    update_suggested_reply,
)


@pytest.fixture
def test_db(tmp_path):
    """Provides a fresh isolated temporary SQLite database for each test."""
    db_file = str(tmp_path / "test_reviewlens.db")
    init_db(db_file)
    return db_file


def test_cache_key_deterministic_and_invalidation():
    """Verify cache key is deterministic, and changes when prompt_version or model changes."""
    text = "great sound quality"
    key1 = compute_cache_key(text, "v1", "gemini-1.5-flash")
    key2 = compute_cache_key(text, "v1", "gemini-1.5-flash")
    assert key1 == key2

    # Changing prompt version invalidates the cache key
    key_v2 = compute_cache_key(text, "v2", "gemini-1.5-flash")
    assert key1 != key_v2

    # Changing model name invalidates the cache key
    key_pro = compute_cache_key(text, "v1", "gemini-1.5-pro")
    assert key1 != key_pro


def test_cache_miss_and_hit(test_db):
    """Verify setting and fetching cached review analyses."""
    cache_key = compute_cache_key("super soft fabric", "v1", "gemini-1.5-flash")

    # Cache miss
    assert get_cached_analysis(cache_key, test_db) is None

    # Set cache
    set_cached_analysis(
        cache_key=cache_key,
        normalized_text="super soft fabric",
        prompt_version="v1",
        model_name="gemini-1.5-flash",
        sentiment="positive",
        topics=["quality", "price"],
        summary="Customer loves the fabric quality.",
        needs_reply=False,
        db_path=test_db,
    )

    # Cache hit
    cached = get_cached_analysis(cache_key, test_db)
    assert cached is not None
    assert cached["sentiment"] == "positive"
    assert cached["topics"] == ["quality", "price"]
    assert cached["summary"] == "Customer loves the fabric quality."
    assert cached["needs_reply"] is False


def test_job_lifecycle_and_tracking(test_db):
    """Verify job creation, status progression, and metric tracking."""
    job_id = "test-job-001"
    create_job(job_id=job_id, total_reviews=30, db_path=test_db)

    job = get_job(job_id, test_db)
    assert job["status"] == "pending"
    assert job["total_reviews"] == 30
    assert job["processed_reviews"] == 0

    # Update progress
    update_job_progress(
        job_id=job_id,
        processed_reviews=15,
        llm_calls=3,
        cache_hits=12,
        status="processing",
        db_path=test_db,
    )
    job_updated = get_job(job_id, test_db)
    assert job_updated["processed_reviews"] == 15
    assert job_updated["llm_calls"] == 3
    assert job_updated["cache_hits"] == 12

    # Complete job
    update_job_progress(
        job_id=job_id,
        processed_reviews=30,
        llm_calls=5,
        cache_hits=25,
        status="completed",
        db_path=test_db,
    )
    job_completed = get_job(job_id, test_db)
    assert job_completed["status"] == "completed"
    assert job_completed["duration_seconds"] is not None


def test_save_and_retrieve_reviews(test_db):
    """Verify storing cleaned review rows and loading them for dashboard view."""
    job_id = "test-job-002"
    create_job(job_id, 2, test_db)

    # Save row 1
    save_review_result(
        job_id=job_id,
        review_id="R001",
        product="Cotton Bedsheet Set",
        rating=5,
        rating_status="valid",
        date="2026-10-02",
        raw_text="Super soft fabric",
        cleaned_text="Super soft fabric",
        clean_status="ok",
        injection_suspected=False,
        sentiment="positive",
        topics=["quality"],
        summary="High quality soft fabric.",
        needs_reply=False,
        db_path=test_db,
    )

    # Save row 2 (needs reply)
    save_review_result(
        job_id=job_id,
        review_id="R007",
        product="Coffee Mug",
        rating=1,
        rating_status="valid",
        date="2026-10-04",
        raw_text="One mug arrived broken.",
        cleaned_text="One mug arrived broken.",
        clean_status="ok",
        injection_suspected=False,
        sentiment="negative",
        topics=["packaging", "support"],
        summary="Mug was broken upon arrival, refund delayed.",
        needs_reply=True,
        db_path=test_db,
    )

    reviews = get_job_reviews(job_id, test_db)
    assert len(reviews) == 2
    assert reviews[0]["review_id"] == "R001"
    assert reviews[0]["sentiment"] == "positive"
    assert reviews[0]["needs_reply"] is False

    assert reviews[1]["review_id"] == "R007"
    assert reviews[1]["sentiment"] == "negative"
    assert reviews[1]["needs_reply"] is True
    assert "packaging" in reviews[1]["topics"]

    # Test update suggested reply on-click
    update_suggested_reply(job_id, "R007", "We apologize for the damaged item! Refund processed.", test_db)
    updated_reviews = get_job_reviews(job_id, test_db)
    assert updated_reviews[1]["suggested_reply"] == "We apologize for the damaged item! Refund processed."
