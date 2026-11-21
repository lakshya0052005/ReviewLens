"""Data cleaning and CSV parsing module for ReviewLens.

Handles messy input data: HTML stripping, missing/invalid ratings,
empty texts, duplicate detection, and heuristic injection detection.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import TextIO

# Regex to strip HTML tags like <b>...</b>
HTML_TAG_RE = re.compile(r"<[^>]+>")

# Heuristic pattern to flag suspected prompt injections early without blocking analysis.
# Why: Allows the dashboard to warn store owners about adversarial reviews while
# still processing the true customer sentiment.
INJECTION_HEURISTIC_RE = re.compile(
    r"(ignore\s+(all\s+)?previous\s+instructions|system\s+prompt|disregard\s+instructions|"
    r"mark\s+this\s+(review\s+)?as|you\s+are\s+now|new\s+instruction)",
    re.IGNORECASE,
)


@dataclass
class CleanedReview:
    """Represents a single parsed and cleaned review row."""

    review_id: str
    product: str
    rating: int | None
    rating_status: str  # "valid" or "unknown"
    date: str
    raw_text: str
    cleaned_text: str
    normalized_text: str
    clean_status: str  # "ok", "skipped_empty", or "duplicate_of:<review_id>"
    injection_suspected: bool


def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return HTML_TAG_RE.sub("", text)


def normalize_text(text: str) -> str:
    """Normalize text for consistent duplicate detection and caching.

    Lowercases, trims leading/trailing spaces, and collapses whitespace.
    """
    collapsed = re.sub(r"\s+", " ", text).strip().lower()
    return collapsed


def parse_rating(raw_rating: str | None) -> tuple[int | None, str]:
    """Parse rating into integer or None.

    Returns (rating, rating_status) where rating_status is 'valid' or 'unknown'.
    Never crashes on invalid values like 'five' or empty strings.
    """
    if raw_rating is None:
        return None, "unknown"

    cleaned = raw_rating.strip()
    if not cleaned:
        return None, "unknown"

    try:
        val = int(cleaned)
        # Ratings are typically 1 to 5 stars
        if 1 <= val <= 5:
            return val, "valid"
        return val, "valid"
    except ValueError:
        # Invalid words like "five" or decimals that are not pure integers
        return None, "unknown"


def detect_injection(text: str) -> bool:
    """Heuristic check for common prompt injection patterns.

    This does NOT block or alter review processing; it only labels the row.
    """
    return bool(INJECTION_HEURISTIC_RE.search(text))


def clean_csv(file_content_or_stream: str | bytes | TextIO) -> list[CleanedReview]:
    """Parse and clean reviews from CSV content.

    Handles multi-line quoted texts, HTML tags, duplicate review detection,
    and missing or malformed fields safely without crashing.
    """
    if isinstance(file_content_or_stream, bytes):
        stream = io.StringIO(file_content_or_stream.decode("utf-8-sig", errors="replace"))
    elif isinstance(file_content_or_stream, str):
        stream = io.StringIO(file_content_or_stream)
    else:
        stream = file_content_or_stream

    reader = csv.DictReader(stream)
    cleaned_reviews: list[CleanedReview] = []

    # Map normalized_text -> first review_id seen to identify duplicate content
    seen_texts: dict[str, str] = {}

    for row in reader:
        review_id = (row.get("review_id") or "").strip()
        product = (row.get("product") or "").strip()
        raw_rating = row.get("rating")
        date = (row.get("date") or "").strip()
        raw_text = row.get("review_text") or ""

        # 1. Parse rating safely
        rating, rating_status = parse_rating(raw_rating)

        # 2. Strip HTML tags from review text
        stripped_text = strip_html(raw_text).strip()

        # 3. Check for empty or whitespace-only content
        if not stripped_text:
            cleaned_reviews.append(
                CleanedReview(
                    review_id=review_id,
                    product=product,
                    rating=rating,
                    rating_status=rating_status,
                    date=date,
                    raw_text=raw_text,
                    cleaned_text="",
                    normalized_text="",
                    clean_status="skipped_empty",
                    injection_suspected=False,
                )
            )
            continue

        # 4. Check for injection pattern heuristics
        injection_suspected = detect_injection(stripped_text)

        # 5. Normalization for duplicate checking & caching
        normalized = normalize_text(stripped_text)

        # 6. Duplicate detection
        # If the same normalized text was already processed, reuse the first one's analysis
        if normalized in seen_texts:
            first_id = seen_texts[normalized]
            clean_status = f"duplicate_of:{first_id}"
        else:
            seen_texts[normalized] = review_id
            clean_status = "ok"

        cleaned_reviews.append(
            CleanedReview(
                review_id=review_id,
                product=product,
                rating=rating,
                rating_status=rating_status,
                date=date,
                raw_text=raw_text,
                cleaned_text=stripped_text,
                normalized_text=normalized,
                clean_status=clean_status,
                injection_suspected=injection_suspected,
            )
        )

    return cleaned_reviews
