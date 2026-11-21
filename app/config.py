"""Application configuration loaded from environment variables and .env file."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load variables from .env file if present
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Centralized configuration for ReviewLens."""

    # LLM Provider configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemini-1.5-flash")
    PROMPT_VERSION: str = os.getenv("PROMPT_VERSION", "v1")

    # Batching and Concurrency
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "5"))
    MAX_CONCURRENT_CALLS: int = int(os.getenv("MAX_CONCURRENT_CALLS", "2"))

    # Resilience: Retries and Circuit Breaker
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    INITIAL_BACKOFF_SECONDS: float = float(os.getenv("INITIAL_BACKOFF_SECONDS", "1.0"))
    MAX_BACKOFF_SECONDS: float = float(os.getenv("MAX_BACKOFF_SECONDS", "8.0"))
    CIRCUIT_BREAKER_THRESHOLD: int = int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "3"))
    CIRCUIT_BREAKER_COOLDOWN_SECONDS: float = float(os.getenv("CIRCUIT_BREAKER_COOLDOWN_SECONDS", "30.0"))

    # Server configuration
    PORT: int = int(os.getenv("PORT", "8000"))
    DB_PATH: str = os.getenv("DB_PATH", "reviewlens.db")


settings = Settings()
