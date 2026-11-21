"""FastAPI application for ReviewLens.

Exposes REST endpoints for:
- Uploading CSV files and initiating background async processing.
- Polling job progress and metrics.
- Fetching processed review results from SQLite.
- Generating on-demand suggested customer replies.
- Serving the frontend dashboard.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.analysis import (
    generate_reply_for_review,
    is_rating_mismatch,
    process_batch,
    ReviewAnalysis,
)
from app.cache import (
    create_job,
    get_job,
    get_job_reviews,
    init_db,
    save_review_result,
    update_job_progress,
    update_suggested_reply,
)
from app.cleaning import clean_csv, CleanedReview
from app.config import settings
from app.llm_client import LLMClient

logger = logging.getLogger("reviewlens.api")
STATIC_DIR = Path(__file__).parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database tables on application startup."""
    init_db(settings.DB_PATH)
    logger.info("Database initialized at %s", settings.DB_PATH)
    yield


app = FastAPI(
    title="ReviewLens API",
    description="AI-powered customer review insights and analysis engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global LLM client instance
llm_client = LLMClient()


# ---------------------------------------------------------------------------
# Background Async Review Processing Worker
# ---------------------------------------------------------------------------

async def process_reviews_job(job_id: str, cleaned_reviews: list[CleanedReview], db_path: str) -> None:
    """Background worker that analyzes reviews in batches with bounded concurrency."""
    # Why: A concurrency semaphore limits simultaneous outgoing LLM requests
    # to protect against exceeding Gemini free tier rate limits (RPM caps).
    semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_CALLS)

    total_reviews = len(cleaned_reviews)
    processed_count = 0
    total_llm_calls = 0
    total_cache_hits = 0

    # Separate reviews by clean_status
    valid_reviews: list[CleanedReview] = []
    duplicate_reviews: list[CleanedReview] = []
    skipped_reviews: list[CleanedReview] = []

    for r in cleaned_reviews:
        if r.clean_status == "ok":
            valid_reviews.append(r)
        elif r.clean_status.startswith("duplicate_of:"):
            duplicate_reviews.append(r)
        else:
            skipped_reviews.append(r)

    # 1. Immediately persist skipped/empty reviews
    for r in skipped_reviews:
        save_review_result(
            job_id=job_id,
            review_id=r.review_id,
            product=r.product,
            rating=r.rating,
            rating_status=r.rating_status,
            date=r.date,
            raw_text=r.raw_text,
            cleaned_text=r.cleaned_text,
            clean_status=r.clean_status,
            injection_suspected=r.injection_suspected,
            sentiment=None,
            topics=None,
            summary="Skipped: empty review content",
            needs_reply=False,
            db_path=db_path,
        )
        processed_count += 1

    update_job_progress(
        job_id=job_id,
        processed_reviews=processed_count,
        llm_calls=total_llm_calls,
        cache_hits=total_cache_hits,
        status="processing",
        db_path=db_path,
    )

    # 2. Process valid reviews in batches of BATCH_SIZE
    batch_size = settings.BATCH_SIZE
    batches = [valid_reviews[i : i + batch_size] for i in range(0, len(valid_reviews), batch_size)]

    # Store analysis results indexed by review_id
    analysis_lookup: dict[str, ReviewAnalysis | str] = {}

    for batch in batches:
        async with semaphore:
            batch_res, calls, hits = await process_batch(
                batch=batch,
                llm_client=llm_client,
                prompt_version=settings.PROMPT_VERSION,
                model_name=settings.MODEL_NAME,
                db_path=db_path,
            )
            total_llm_calls += calls
            total_cache_hits += hits

            for r in batch:
                res = batch_res.get(r.review_id, "analysis_failed")
                analysis_lookup[r.review_id] = res

                if isinstance(res, ReviewAnalysis):
                    mismatch = is_rating_mismatch(r.rating, res.sentiment)
                    save_review_result(
                        job_id=job_id,
                        review_id=r.review_id,
                        product=r.product,
                        rating=r.rating,
                        rating_status=r.rating_status,
                        date=r.date,
                        raw_text=r.raw_text,
                        cleaned_text=r.cleaned_text,
                        clean_status="ok",
                        injection_suspected=r.injection_suspected,
                        sentiment=res.sentiment,
                        topics=res.topics,
                        summary=res.summary,
                        needs_reply=res.needs_reply,
                        mismatch_flag=mismatch,
                        db_path=db_path,
                    )
                else:
                    save_review_result(
                        job_id=job_id,
                        review_id=r.review_id,
                        product=r.product,
                        rating=r.rating,
                        rating_status=r.rating_status,
                        date=r.date,
                        raw_text=r.raw_text,
                        cleaned_text=r.cleaned_text,
                        clean_status="analysis_failed",
                        injection_suspected=r.injection_suspected,
                        sentiment=None,
                        topics=None,
                        summary="Analysis failed due to repeated validation or provider errors",
                        needs_reply=None,
                        mismatch_flag=False,
                        db_path=db_path,
                    )
                processed_count += 1

            update_job_progress(
                job_id=job_id,
                processed_reviews=processed_count,
                llm_calls=total_llm_calls,
                cache_hits=total_cache_hits,
                status="processing",
                db_path=db_path,
            )

    # 3. Resolve duplicates (reuse first review's result)
    for r in duplicate_reviews:
        first_id = r.clean_status.split("duplicate_of:")[1]
        parent_res = analysis_lookup.get(first_id)

        if isinstance(parent_res, ReviewAnalysis):
            mismatch = is_rating_mismatch(r.rating, parent_res.sentiment)
            save_review_result(
                job_id=job_id,
                review_id=r.review_id,
                product=r.product,
                rating=r.rating,
                rating_status=r.rating_status,
                date=r.date,
                raw_text=r.raw_text,
                cleaned_text=r.cleaned_text,
                clean_status=r.clean_status,
                injection_suspected=r.injection_suspected,
                sentiment=parent_res.sentiment,
                topics=parent_res.topics,
                summary=f"(Duplicate of {first_id}) {parent_res.summary}",
                needs_reply=parent_res.needs_reply,
                mismatch_flag=mismatch,
                db_path=db_path,
            )
        else:
            save_review_result(
                job_id=job_id,
                review_id=r.review_id,
                product=r.product,
                rating=r.rating,
                rating_status=r.rating_status,
                date=r.date,
                raw_text=r.raw_text,
                cleaned_text=r.cleaned_text,
                clean_status=r.clean_status,
                injection_suspected=r.injection_suspected,
                sentiment=None,
                topics=None,
                summary=f"Duplicate of {first_id}",
                needs_reply=False,
                db_path=db_path,
            )
        processed_count += 1

    # Mark job complete
    update_job_progress(
        job_id=job_id,
        processed_reviews=processed_count,
        llm_calls=total_llm_calls,
        cache_hits=total_cache_hits,
        status="completed",
        db_path=db_path,
    )
    logger.info("Job %s completed successfully: %d reviews processed", job_id, processed_count)


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_reviews_file(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload reviews CSV, parse and clean rows, and trigger async background processing."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a .csv file.")

    content = await file.read()
    if not content.strip():
        raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

    cleaned_reviews = clean_csv(content)
    if not cleaned_reviews:
        raise HTTPException(status_code=400, detail="No review rows found in CSV.")

    job_id = str(uuid.uuid4())
    create_job(job_id=job_id, total_reviews=len(cleaned_reviews), db_path=settings.DB_PATH)

    # Spawn background task
    asyncio.create_task(
        process_reviews_job(
            job_id=job_id,
            cleaned_reviews=cleaned_reviews,
            db_path=settings.DB_PATH,
        )
    )

    return {
        "job_id": job_id,
        "total_reviews": len(cleaned_reviews),
        "status": "processing",
    }


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict[str, Any]:
    """Poll job status, progress percentage, LLM calls made, and cache hits."""
    job = get_job(job_id, db_path=settings.DB_PATH)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    progress_pct = 0.0
    if job["total_reviews"] > 0:
        progress_pct = round((job["processed_reviews"] / job["total_reviews"]) * 100, 1)

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "total_reviews": job["total_reviews"],
        "processed_reviews": job["processed_reviews"],
        "progress_percentage": progress_pct,
        "llm_calls": job["llm_calls"],
        "cache_hits": job["cache_hits"],
        "duration_seconds": job["duration_seconds"],
        "error_message": job["error_message"],
    }


@app.get("/api/jobs/{job_id}/results")
async def get_job_results(job_id: str) -> dict[str, Any]:
    """Retrieve all processed review rows for a job."""
    job = get_job(job_id, db_path=settings.DB_PATH)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    reviews = get_job_reviews(job_id, db_path=settings.DB_PATH)
    return {
        "job_id": job_id,
        "status": job["status"],
        "reviews": reviews,
    }


@app.post("/api/reply/{job_id}/{review_id}")
async def generate_reply_endpoint(job_id: str, review_id: str) -> dict[str, Any]:
    """Generate an on-demand merchant reply for a review marked needs_reply."""
    reviews = get_job_reviews(job_id, db_path=settings.DB_PATH)
    target_review = next((r for r in reviews if r["review_id"] == review_id), None)

    if not target_review:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found in job {job_id}.")

    reply = await generate_reply_for_review(
        review_text=target_review["cleaned_text"],
        product=target_review["product"],
        summary=target_review.get("summary") or "",
        llm_client=llm_client,
    )

    # Persist generated reply
    update_suggested_reply(job_id, review_id, reply, db_path=settings.DB_PATH)

    return {
        "review_id": review_id,
        "suggested_reply": reply,
    }


# ---------------------------------------------------------------------------
# Static Files & Dashboard UI Route
# ---------------------------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    """Serve the single-page application dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "ReviewLens API is running. Dashboard assets will be loaded in Step 8."}
