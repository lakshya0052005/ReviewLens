"""SQLite cache and persistence layer for ReviewLens.

Uses standard library sqlite3 with parameterized queries only.
Handles:
1. LLM response caching keyed by SHA-256(normalized_text + prompt_version + model_name).
2. Upload job state tracking (progress, calls made, cache hits, duration).
3. Persisting cleaned and analyzed review rows.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from typing import Any

# Default database file path
DEFAULT_DB_PATH = "reviewlens.db"


def compute_cache_key(normalized_text: str, prompt_version: str, model_name: str) -> str:
    """Compute a unique SHA-256 hash for a review and its LLM configuration.

    Why: Including prompt_version and model_name ensures that changing the prompt
    or upgrading the model automatically invalidates stale cached responses without
    needing manual database wipes.
    """
    payload = f"{normalized_text}|{prompt_version}|{model_name}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create a connection configured with Row factory and WAL mode for concurrency."""
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    # Enable Write-Ahead Logging for fast concurrent reads and writes
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initialize database tables with parameterized schema definitions."""
    with get_connection(db_path) as conn:
        # Table 1: analyses (LLM response cache)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                cache_key TEXT PRIMARY KEY,
                normalized_text TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                model_name TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                topics TEXT NOT NULL,       -- JSON array of topics
                summary TEXT NOT NULL,
                needs_reply INTEGER NOT NULL, -- 1 for True, 0 for False
                created_at REAL NOT NULL
            );
            """
        )

        # Table 2: jobs (Tracks upload background jobs)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,       -- 'pending', 'processing', 'completed', 'failed'
                total_reviews INTEGER NOT NULL,
                processed_reviews INTEGER NOT NULL,
                llm_calls INTEGER NOT NULL,
                cache_hits INTEGER NOT NULL,
                error_message TEXT,
                created_at REAL NOT NULL,
                completed_at REAL,
                duration_seconds REAL
            );
            """
        )

        # Table 3: reviews (Cleaned & analyzed rows for dashboard display)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                review_id TEXT NOT NULL,
                product TEXT NOT NULL,
                rating INTEGER,
                rating_status TEXT NOT NULL,  -- 'valid' or 'unknown'
                date TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                cleaned_text TEXT NOT NULL,
                clean_status TEXT NOT NULL,   -- 'ok', 'skipped_empty', 'duplicate_of:<id>', 'analysis_failed'
                injection_suspected INTEGER NOT NULL, -- 1 or 0
                sentiment TEXT,               -- 'positive', 'neutral', 'negative' or NULL
                topics TEXT,                  -- JSON array or NULL
                summary TEXT,
                needs_reply INTEGER,          -- 1 or 0 or NULL
                suggested_reply TEXT,
                mismatch_flag INTEGER DEFAULT 0, -- 1 if rating disagrees with sentiment
                FOREIGN KEY (job_id) REFERENCES jobs (job_id)
            );
            """
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Cache Operations (analyses table)
# ---------------------------------------------------------------------------

def get_cached_analysis(cache_key: str, db_path: str = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Retrieve cached LLM analysis if available."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT sentiment, topics, summary, needs_reply
            FROM analyses
            WHERE cache_key = ?;
            """,
            (cache_key,),
        )
        row = cursor.fetchone()
        if not row:
            return None

        return {
            "sentiment": row["sentiment"],
            "topics": json.loads(row["topics"]),
            "summary": row["summary"],
            "needs_reply": bool(row["needs_reply"]),
        }


def set_cached_analysis(
    cache_key: str,
    normalized_text: str,
    prompt_version: str,
    model_name: str,
    sentiment: str,
    topics: list[str],
    summary: str,
    needs_reply: bool,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Save an analysis result into the cache. Never stores failures."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO analyses (
                cache_key, normalized_text, prompt_version, model_name,
                sentiment, topics, summary, needs_reply, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                cache_key,
                normalized_text,
                prompt_version,
                model_name,
                sentiment,
                json.dumps(topics),
                summary,
                1 if needs_reply else 0,
                time.time(),
            ),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Job Operations (jobs table)
# ---------------------------------------------------------------------------

def create_job(job_id: str, total_reviews: int, db_path: str = DEFAULT_DB_PATH) -> None:
    """Register a new upload job."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, status, total_reviews, processed_reviews,
                llm_calls, cache_hits, created_at
            ) VALUES (?, 'pending', ?, 0, 0, 0, ?);
            """,
            (job_id, total_reviews, time.time()),
        )
        conn.commit()


def update_job_progress(
    job_id: str,
    processed_reviews: int,
    llm_calls: int,
    cache_hits: int,
    status: str = "processing",
    error_message: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Update job progress stats."""
    with get_connection(db_path) as conn:
        now = time.time()
        completed_at = now if status in ("completed", "failed") else None

        # Calculate duration if finishing
        duration = None
        if completed_at:
            cur = conn.execute("SELECT created_at FROM jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            if row:
                duration = round(completed_at - row["created_at"], 2)

        conn.execute(
            """
            UPDATE jobs
            SET status = ?,
                processed_reviews = ?,
                llm_calls = ?,
                cache_hits = ?,
                error_message = ?,
                completed_at = COALESCE(?, completed_at),
                duration_seconds = COALESCE(?, duration_seconds)
            WHERE job_id = ?;
            """,
            (
                status,
                processed_reviews,
                llm_calls,
                cache_hits,
                error_message,
                completed_at,
                duration,
                job_id,
            ),
        )
        conn.commit()


def get_job(job_id: str, db_path: str = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Fetch status and metrics for a job."""
    with get_connection(db_path) as conn:
        cursor = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)


# ---------------------------------------------------------------------------
# Review Persistence Operations (reviews table)
# ---------------------------------------------------------------------------

def save_review_result(
    job_id: str,
    review_id: str,
    product: str,
    rating: int | None,
    rating_status: str,
    date: str,
    raw_text: str,
    cleaned_text: str,
    clean_status: str,
    injection_suspected: bool,
    sentiment: str | None,
    topics: list[str] | None,
    summary: str | None,
    needs_reply: bool | None,
    suggested_reply: str | None = None,
    mismatch_flag: bool = False,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Insert or update a review row associated with a job."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO reviews (
                job_id, review_id, product, rating, rating_status, date,
                raw_text, cleaned_text, clean_status, injection_suspected,
                sentiment, topics, summary, needs_reply, suggested_reply,
                mismatch_flag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                job_id,
                review_id,
                product,
                rating,
                rating_status,
                date,
                raw_text,
                cleaned_text,
                clean_status,
                1 if injection_suspected else 0,
                sentiment,
                json.dumps(topics) if topics is not None else None,
                summary,
                1 if needs_reply else (0 if needs_reply is False else None),
                suggested_reply,
                1 if mismatch_flag else 0,
            ),
        )
        conn.commit()


def get_job_reviews(job_id: str, db_path: str = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Retrieve all review records for a completed job."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM reviews WHERE job_id = ? ORDER BY id ASC;",
            (job_id,),
        )
        rows = cursor.fetchall()
        results: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            if d.get("topics"):
                d["topics"] = json.loads(d["topics"])
            else:
                d["topics"] = []
            d["injection_suspected"] = bool(d.get("injection_suspected"))
            d["mismatch_flag"] = bool(d.get("mismatch_flag"))
            d["needs_reply"] = bool(d.get("needs_reply")) if d.get("needs_reply") is not None else None
            results.append(d)
        return results


def update_suggested_reply(
    job_id: str,
    review_id: str,
    suggested_reply: str,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """Save an on-demand generated reply for a review row."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE reviews
            SET suggested_reply = ?
            WHERE job_id = ? AND review_id = ?;
            """,
            (suggested_reply, job_id, review_id),
        )
        conn.commit()
