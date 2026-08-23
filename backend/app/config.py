from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

# Load .env from the repo root before any setting is read, so `make api`,
# `make eval` and pytest all see the same configuration without the caller
# having to export anything.
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR.parent / ".env")
except ImportError:  # optional dependency; env vars still work
    pass


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path = BASE_DIR / "data" / "corpus"
    index_path: Path = BASE_DIR / "data" / "index.json"
    audit_log_path: Path = Path(os.getenv("AUDIT_LOG_PATH", BASE_DIR / "data" / "audit.jsonl"))
    top_k: int = int(os.getenv("TOP_K", "6"))
    redact_pii: bool = os.getenv("REDACT_PII", "1") != "0"
    cors_origins: tuple[str, ...] = tuple(
        os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    )


settings = Settings()
