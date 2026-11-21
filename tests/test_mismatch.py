"""Unit tests specifically verifying the Bonus Rating Mismatch feature."""

from app.analysis import is_rating_mismatch
from app.insights import compute_summary_stats


def test_rating_mismatch_core_rules():
    """Verify rating mismatch rules for extreme stars vs contradictory sentiments."""
    # 5 stars but negative sentiment (e.g. R008: "Terrible. Both handles came off...")
    assert is_rating_mismatch(rating=5, sentiment="negative") is True

    # 4 stars but negative sentiment
    assert is_rating_mismatch(rating=4, sentiment="negative") is True

    # 1 star with positive sentiment (e.g. R003 if sarcastic text is read positively)
    assert is_rating_mismatch(rating=1, sentiment="positive") is True

    # 2 stars with positive sentiment
    assert is_rating_mismatch(rating=2, sentiment="positive") is True

    # Consistent cases (no mismatch)
    assert is_rating_mismatch(rating=5, sentiment="positive") is False
    assert is_rating_mismatch(rating=1, sentiment="negative") is False
    assert is_rating_mismatch(rating=3, sentiment="neutral") is False

    # Neutral sentiment with 4 or 2 stars is not flagged as a hard mismatch
    assert is_rating_mismatch(rating=4, sentiment="neutral") is False

    # Missing rating (None) cannot be compared, so never flag mismatch
    assert is_rating_mismatch(rating=None, sentiment="negative") is False


def test_store_rating_agreement_percentage():
    """Verify calculation of store-wide rating agreement percentage."""
    reviews = [
        {"rating": 5, "sentiment": "positive", "mismatch_flag": False},
        {"rating": 5, "sentiment": "positive", "mismatch_flag": False},
        {"rating": 4, "sentiment": "positive", "mismatch_flag": False},
        {"rating": 1, "sentiment": "negative", "mismatch_flag": False},
        # 1 mismatch out of 5 valid ratings -> (4/5) * 100 = 80.0%
        {"rating": 5, "sentiment": "negative", "mismatch_flag": True},
        # Missing rating should be excluded from denominator
        {"rating": None, "sentiment": "negative", "mismatch_flag": False},
    ]

    stats = compute_summary_stats(reviews)
    assert stats["mismatch_count"] == 1
    assert stats["rating_agreement_pct"] == 80.0
