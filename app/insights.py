"""Insights aggregation module for ReviewLens.

Calculates:
- High-level summary metrics (sentiment percentages, needs-reply counts, rating agreement %).
- Deterministic topic frequency counts.
- Per-product breakdown with product-level sentiment splits and top issues.
- Top 3 complaints and top 3 praises phrased in plain English (using 1 cached LLM call with deterministic fallback).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

from app.cache import (
    compute_cache_key,
    get_cached_analysis,
    set_cached_analysis,
    get_connection,
    get_job,
    get_job_reviews,
    DEFAULT_DB_PATH,
)
from app.config import settings
from app.llm_client import LLMClient

logger = logging.getLogger("reviewlens.insights")

TOP_INSIGHTS_SYSTEM_PROMPT = """You are an e-commerce data analyst.
Given the categorized customer feedback summaries, generate:
1. Top 3 distinct customer praises across all products.
2. Top 3 distinct customer complaints across all products.

Return ONLY a valid JSON object in this exact schema:
{
  "top_praises": ["Praise 1", "Praise 2", "Praise 3"],
  "top_complaints": ["Complaint 1", "Complaint 2", "Complaint 3"]
}
Keep each bullet point to a single concise, punchy sentence in English.
"""


def compute_summary_stats(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute overall store summary statistics."""
    total = len(reviews)
    if total == 0:
        return {
            "total_reviews": 0,
            "positive_count": 0,
            "neutral_count": 0,
            "negative_count": 0,
            "positive_pct": 0.0,
            "neutral_pct": 0.0,
            "negative_pct": 0.0,
            "needs_reply_count": 0,
            "mismatch_count": 0,
            "rating_agreement_pct": 100.0,
            "skipped_empty_count": 0,
            "duplicates_count": 0,
            "analysis_failed_count": 0,
        }

    positive_count = sum(1 for r in reviews if r.get("sentiment") == "positive")
    neutral_count = sum(1 for r in reviews if r.get("sentiment") == "neutral")
    negative_count = sum(1 for r in reviews if r.get("sentiment") == "negative")

    analyzed_count = positive_count + neutral_count + negative_count
    pos_pct = round((positive_count / analyzed_count) * 100, 1) if analyzed_count > 0 else 0.0
    neu_pct = round((neutral_count / analyzed_count) * 100, 1) if analyzed_count > 0 else 0.0
    neg_pct = round((negative_count / analyzed_count) * 100, 1) if analyzed_count > 0 else 0.0

    needs_reply_count = sum(1 for r in reviews if r.get("needs_reply") is True)
    mismatch_count = sum(1 for r in reviews if r.get("mismatch_flag") is True)

    # Agreement percentage: reviews with valid ratings where star rating agreed with sentiment
    rated_reviews = [r for r in reviews if r.get("rating") is not None and r.get("sentiment") is not None]
    if rated_reviews:
        agreement_pct = round(((len(rated_reviews) - mismatch_count) / len(rated_reviews)) * 100, 1)
    else:
        agreement_pct = 100.0

    skipped_empty_count = sum(1 for r in reviews if r.get("clean_status") == "skipped_empty")
    duplicates_count = sum(1 for r in reviews if str(r.get("clean_status", "")).startswith("duplicate_of:"))
    analysis_failed_count = sum(1 for r in reviews if r.get("clean_status") == "analysis_failed")

    return {
        "total_reviews": total,
        "analyzed_count": analyzed_count,
        "positive_count": positive_count,
        "neutral_count": neutral_count,
        "negative_count": negative_count,
        "positive_pct": pos_pct,
        "neutral_pct": neu_pct,
        "negative_pct": neg_pct,
        "needs_reply_count": needs_reply_count,
        "mismatch_count": mismatch_count,
        "rating_agreement_pct": agreement_pct,
        "skipped_empty_count": skipped_empty_count,
        "duplicates_count": duplicates_count,
        "analysis_failed_count": analysis_failed_count,
    }


def compute_per_product_insights(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute per-product sentiment breakdown and top complaint/praise topics."""
    products: dict[str, list[dict[str, Any]]] = {}
    for r in reviews:
        prod = r.get("product") or "Unknown Product"
        products.setdefault(prod, []).append(r)

    results: list[dict[str, Any]] = []
    for prod_name, p_reviews in sorted(products.items()):
        total = len(p_reviews)
        pos = sum(1 for r in p_reviews if r.get("sentiment") == "positive")
        neu = sum(1 for r in p_reviews if r.get("sentiment") == "neutral")
        neg = sum(1 for r in p_reviews if r.get("sentiment") == "negative")
        analyzed = pos + neu + neg
        pos_pct = round((pos / analyzed) * 100, 1) if analyzed > 0 else 0.0
        needs_reply = sum(1 for r in p_reviews if r.get("needs_reply") is True)

        # Count topics
        neg_topics = Counter(
            t
            for r in p_reviews
            if r.get("sentiment") == "negative"
            for t in (r.get("topics") or [])
        )
        pos_topics = Counter(
            t
            for r in p_reviews
            if r.get("sentiment") == "positive"
            for t in (r.get("topics") or [])
        )

        top_issue = neg_topics.most_common(1)[0][0] if neg_topics else "None"
        top_strength = pos_topics.most_common(1)[0][0] if pos_topics else "None"

        results.append(
            {
                "product": prod_name,
                "total_reviews": total,
                "positive_count": pos,
                "neutral_count": neu,
                "negative_count": neg,
                "positive_pct": pos_pct,
                "needs_reply_count": needs_reply,
                "top_complaint_topic": top_issue,
                "top_praise_topic": top_strength,
            }
        )

    return results


def deterministic_top_insights(reviews: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Deterministic fallback to extract top 3 complaints and praises directly from topic frequencies."""
    pos_summaries = [
        r.get("summary")
        for r in reviews
        if r.get("sentiment") == "positive" and r.get("summary") and not str(r.get("summary")).startswith("Skipped")
    ]
    neg_summaries = [
        r.get("summary")
        for r in reviews
        if r.get("sentiment") == "negative" and r.get("summary") and not str(r.get("summary")).startswith("Skipped")
    ]

    top_praises = pos_summaries[:3] if pos_summaries else ["Customers appreciate product quality and fast delivery."]
    top_complaints = neg_summaries[:3] if neg_summaries else ["Occasional packaging damage and support delays reported."]

    # Pad to 3 if fewer
    while len(top_praises) < 3:
        top_praises.append("Positive customer experiences reported.")
    while len(top_complaints) < 3:
        top_complaints.append("Minor issues reported by customers.")

    return {
        "top_praises": top_praises[:3],
        "top_complaints": top_complaints[:3],
    }


async def generate_top_insights(
    reviews: list[dict[str, Any]],
    llm_client: LLMClient | None = None,
    job_id: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> dict[str, list[str]]:
    """Generate top 3 complaints and top 3 praises using 1 cached LLM call with fallback."""
    # Deterministic baseline
    fallback = deterministic_top_insights(reviews)

    if llm_client is None:
        return fallback

    # Check cache for this job's aggregate insights
    cache_key = f"insights:{job_id}" if job_id else None
    if cache_key:
        cached = get_cached_analysis(cache_key, db_path=db_path)
        if cached and "top_praises" in cached.get("topics", []):
            try:
                data = json.loads(cached["summary"])
                return data
            except Exception:
                pass

    # Extract distinct summaries
    positive_samples = [
        f"- {r.get('product')}: {r.get('summary')}"
        for r in reviews
        if r.get("sentiment") == "positive" and r.get("summary")
    ][:10]

    negative_samples = [
        f"- {r.get('product')}: {r.get('summary')}"
        for r in reviews
        if r.get("sentiment") == "negative" and r.get("summary")
    ][:10]

    if not positive_samples and not negative_samples:
        return fallback

    user_prompt = f"""Customer Praise Summaries:
{chr(10).join(positive_samples) if positive_samples else "None"}

Customer Complaint Summaries:
{chr(10).join(negative_samples) if negative_samples else "None"}
"""

    try:
        raw_output = await llm_client.generate_json(TOP_INSIGHTS_SYSTEM_PROMPT, user_prompt)
        data = json.loads(raw_output)
        praises = data.get("top_praises", [])
        complaints = data.get("top_complaints", [])

        if len(praises) >= 3 and len(complaints) >= 3:
            result = {
                "top_praises": praises[:3],
                "top_complaints": complaints[:3],
            }
            if cache_key:
                set_cached_analysis(
                    cache_key=cache_key,
                    normalized_text="job_insights",
                    prompt_version=settings.PROMPT_VERSION,
                    model_name=settings.MODEL_NAME,
                    sentiment="neutral",
                    topics=["top_praises", "top_complaints"],
                    summary=json.dumps(result),
                    needs_reply=False,
                    db_path=db_path,
                )
            return result
    except Exception as err:
        logger.warning("Failed to generate LLM insights (%s), using deterministic fallback.", err)

    return fallback


async def get_job_insights(job_id: str, db_path: str = DEFAULT_DB_PATH, client: LLMClient | None = None) -> dict[str, Any]:
    """Compile complete insights object for a job."""
    job = get_job(job_id, db_path=db_path)
    reviews = get_job_reviews(job_id, db_path=db_path)

    stats = compute_summary_stats(reviews)
    product_insights = compute_per_product_insights(reviews)
    top_highlights = await generate_top_insights(reviews, llm_client=client, job_id=job_id, db_path=db_path)

    metrics = {
        "llm_calls": job["llm_calls"] if job else 0,
        "cache_hits": job["cache_hits"] if job else 0,
        "duration_seconds": job["duration_seconds"] if job else None,
        "total_reviews": job["total_reviews"] if job else len(reviews),
    }

    return {
        "job_id": job_id,
        "summary": stats,
        "per_product": product_insights,
        "top_praises": top_highlights["top_praises"],
        "top_complaints": top_highlights["top_complaints"],
        "metrics": metrics,
    }
