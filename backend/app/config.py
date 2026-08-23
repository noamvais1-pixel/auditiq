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


def env(name: str, default: str | None = None) -> str | None:
    """Read an environment variable, treating blank as unset.

    A .env file written from the template has keys like ``AUDIT_LOG_PATH=`` with
    nothing after the ``=``. ``os.getenv`` reports those as set-and-empty, so the
    default never applies: the audit log tried to open ``Path("")`` -- the current
    directory -- and every request died on IsADirectoryError. Blank means "not
    configured" everywhere in this app.
    """
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path = BASE_DIR / "data" / "corpus"
    index_path: Path = BASE_DIR / "data" / "index.json"
    audit_log_path: Path = Path(env("AUDIT_LOG_PATH") or BASE_DIR / "data" / "audit.jsonl")
    top_k: int = int(env("TOP_K", "6"))
    redact_pii: bool = env("REDACT_PII", "1") != "0"
    cors_origins: tuple[str, ...] = tuple(
        o.strip()
        for o in (env("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173") or "").split(",")
        if o.strip()
    )


settings = Settings()
