from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_environment(path: Path | None = None) -> bool:
    """Load project-local credentials without overriding the current shell."""
    return load_dotenv(path or PROJECT_ROOT / ".env", override=False)


def model_credentials(scope: str) -> tuple[str | None, str | None]:
    prefix = scope.upper()
    return (
        os.getenv(f"{prefix}_LLM_API_KEY") or os.getenv("LLM_API_KEY"),
        os.getenv(f"{prefix}_LLM_BASE_URL") or os.getenv("LLM_BASE_URL"),
    )
