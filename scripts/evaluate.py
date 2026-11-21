#!/usr/bin/env python3
"""Evaluation harness for ReviewLens.

Runs live LLM inference against hand-labeled benchmark edge cases in tests/golden.json
and prints an evaluation accuracy table.

Usage:
  python scripts/evaluate.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.analysis import _execute_batch_call
from app.cleaning import CleanedReview
from app.config import settings
from app.llm_client import LLMClient


async def run_evaluation():
    golden_path = ROOT_DIR / "tests" / "golden.json"
    if not golden_path.exists():
        print(f"Error: {golden_path} not found.")
        sys.exit(1)

    api_key = settings.GEMINI_API_KEY
    if not api_key:
        print("\n[!] Warning: GEMINI_API_KEY is not set in your environment or .env file.")
        print("    To run live evaluation against the Gemini API, set your key:")
        print("    GEMINI_API_KEY=your_key_here python scripts/evaluate.py\n")
        sys.exit(1)

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_cases = json.load(f)

    print("\n" + "=" * 80)
    print(" ReviewLens Model Evaluation Harness ")
    print(f" Model: {settings.MODEL_NAME} | Test cases: {len(golden_cases)}")
    print("=" * 80)

    # Convert golden test cases into CleanedReview objects
    cleaned_items = [
        CleanedReview(
            review_id=c["review_id"],
            product=c["product"],
            rating=c["rating"],
            rating_status="valid",
            date="2026-10-01",
            raw_text=c["review_text"],
            cleaned_text=c["review_text"],
            normalized_text=c["review_text"].strip().lower(),
            clean_status="ok",
            injection_suspected=False,
        )
        for c in golden_cases
    ]

    llm_client = LLMClient(api_key=api_key, model_name=settings.MODEL_NAME)

    print("\nSending batch request to LLM API...")
    try:
        actual_results = await _execute_batch_call(cleaned_items, llm_client)
    except Exception as err:
        print(f"Evaluation failed during API call: {err}")
        sys.exit(1)

    # Format table output
    headers = ["ID", "Case Type", "Exp Sent", "Act Sent", "Sent OK?", "Topics Match", "Reply OK?", "Result"]
    row_fmt = "{:<6} {:<18} {:<10} {:<10} {:<10} {:<14} {:<11} {:<8}"

    print("-" * 92)
    print(row_fmt.format(*headers))
    print("-" * 92)

    passed_count = 0

    for case in golden_cases:
        rid = case["review_id"]
        exp_sent = case["expected_sentiment"]
        exp_topics = set(case["expected_topics"])
        exp_reply = case["expected_needs_reply"]

        actual = actual_results.get(rid)
        if not actual:
            print(row_fmt.format(rid, case["case_type"], exp_sent, "MISSING", "NO", "NO", "NO", "FAIL"))
            continue

        act_sent = actual.sentiment
        act_topics = set(actual.topics)
        act_reply = actual.needs_reply

        sent_ok = act_sent == exp_sent
        # Topic is considered matched if there is at least an overlap with expected core topics
        topics_ok = bool(exp_topics.intersection(act_topics))
        reply_ok = act_reply == exp_reply

        all_ok = sent_ok and reply_ok
        if all_ok:
            passed_count += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(
            row_fmt.format(
                rid,
                case["case_type"][:18],
                exp_sent,
                act_sent,
                "YES" if sent_ok else "NO",
                "YES" if topics_ok else "NO",
                "YES" if reply_ok else "NO",
                status,
            )
        )

    accuracy = round((passed_count / len(golden_cases)) * 100, 1)
    print("-" * 92)
    print(f"Summary: {passed_count}/{len(golden_cases)} passed ({accuracy}% accuracy)")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
