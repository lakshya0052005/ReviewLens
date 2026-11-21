"""API integration tests using FastAPI TestClient with mocked LLM."""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from app.main import app, process_reviews_job
from app.cache import init_db
from app.cleaning import clean_csv


@pytest.fixture
def test_env(tmp_path, monkeypatch):
    """Fixture to isolate SQLite DB and mock LLM calls."""
    db_file = str(tmp_path / "test_api.db")
    init_db(db_file)
    monkeypatch.setattr("app.config.settings.DB_PATH", db_file)
    return db_file


@pytest.fixture
def client(test_env):
    return TestClient(app)


def test_root_endpoint(client):
    """Verify the root endpoint responds successfully."""
    response = client.get("/")
    assert response.status_code == 200


def test_upload_invalid_file_extension(client):
    """Verify non-csv files are rejected with 400."""
    response = client.post(
        "/api/upload",
        files={"file": ("test.txt", b"some text", "text/plain")},
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_upload_empty_csv(client):
    """Verify empty CSV upload is rejected with 400."""
    response = client.post(
        "/api/upload",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_end_to_end_job_processing(test_env):
    """Verify full upload -> background async processing -> results flow with mocked LLM."""
    csv_content = """review_id,product,rating,date,review_text
R001,Bedsheet,5,2026-10-02,"Super soft fabric"
R002,Bedsheet,1,2026-10-03,"Torn sheets"
R005,Bedsheet,5,2026-10-07,""
R015,Bedsheet,1,2026-10-03,"Torn sheets"
"""
    cleaned = clean_csv(csv_content)
    job_id = "test-job-e2e"

    from app.cache import create_job, get_job, get_job_reviews

    create_job(job_id, len(cleaned), db_path=test_env)

    mock_llm = AsyncMock()
    # Mock LLM returning valid analysis for R001 and R002
    mock_llm.generate_json.return_value = """{
      "reviews": [
        {
          "review_id": "R001",
          "sentiment": "positive",
          "topics": ["quality"],
          "summary": "Great soft fabric.",
          "needs_reply": false
        },
        {
          "review_id": "R002",
          "sentiment": "negative",
          "topics": ["quality"],
          "summary": "Sheets arrived torn.",
          "needs_reply": true
        }
      ]
    }"""

    with patch("app.main.llm_client", mock_llm):
        await process_reviews_job(job_id, cleaned, db_path=test_env)

    # 1. Job completed
    job = get_job(job_id, db_path=test_env)
    assert job["status"] == "completed"
    assert job["processed_reviews"] == 4

    # 2. Verify results
    reviews = get_job_reviews(job_id, db_path=test_env)
    assert len(reviews) == 4
    rev_map = {r["review_id"]: r for r in reviews}

    # R001: positive
    assert rev_map["R001"]["sentiment"] == "positive"
    assert rev_map["R001"]["clean_status"] == "ok"

    # R002: negative, needs_reply=True
    assert rev_map["R002"]["sentiment"] == "negative"
    assert rev_map["R002"]["needs_reply"] is True

    # R005: skipped empty
    assert rev_map["R005"]["clean_status"] == "skipped_empty"

    # R015: duplicate of R002, reuses sentiment and summary
    assert rev_map["R015"]["clean_status"] == "duplicate_of:R002"
    assert rev_map["R015"]["sentiment"] == "negative"
    assert "Duplicate of R002" in rev_map["R015"]["summary"]


@pytest.mark.asyncio
async def test_suggested_reply_flow(test_env):
    """Verify on-demand reply endpoint generates and persists reply."""
    from app.cache import create_job, save_review_result, get_job_reviews

    job_id = "job-reply-test"
    create_job(job_id, 1, db_path=test_env)
    save_review_result(
        job_id=job_id,
        review_id="R007",
        product="Mug",
        rating=1,
        rating_status="valid",
        date="2026-10-04",
        raw_text="Broken mug",
        cleaned_text="Broken mug",
        clean_status="ok",
        injection_suspected=False,
        sentiment="negative",
        topics=["packaging"],
        summary="Arrived broken",
        needs_reply=True,
        db_path=test_env,
    )

    mock_llm = AsyncMock()
    mock_llm.generate_json.return_value = "We are deeply sorry for the broken mug! A replacement has been dispatched."

    with patch("app.main.llm_client", mock_llm):
        client = TestClient(app)
        res = client.post(f"/api/reply/{job_id}/R007")

    assert res.status_code == 200
    data = res.json()
    assert data["review_id"] == "R007"
    assert "replacement has been dispatched" in data["suggested_reply"]

    # Verify persisted in SQLite
    reviews = get_job_reviews(job_id, db_path=test_env)
    assert reviews[0]["suggested_reply"] == "We are deeply sorry for the broken mug! A replacement has been dispatched."
