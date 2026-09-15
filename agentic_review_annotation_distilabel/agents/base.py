from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AgentConfig(BaseModel):
    """Runtime configuration shared by the supported agent harnesses."""

    harness: Literal["mini_swe_agent", "openhands", "opencollab"]
    model: str
    runtime: Literal["local", "docker"] = "local"
    workspace: Path = Path(".")
    benchmark_path: Path | None = None
    instance: int | str = 0
    instance_id: str | None = None
    base_commit: str | None = None
    docker_image: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    step_limit: int = Field(default=50, ge=0)
    cost_limit: float = Field(default=3.0, ge=0)
    command_timeout: int = Field(default=30, gt=0)
    output_dir: Path = PROJECT_ROOT / "output" / "traj"
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    environment_kwargs: dict[str, Any] = Field(default_factory=dict)
    harness_kwargs: dict[str, Any] = Field(default_factory=dict)
    benchmark_instance: dict[str, Any] | None = None

    @model_validator(mode="after")
    def require_docker_image(self) -> "AgentConfig":
        if self.runtime == "docker" and not (self.docker_image or self.benchmark_path):
            raise ValueError("docker runtime requires docker_image or benchmark_path")
        return self


class Agent(ABC):
    harness_name: str

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    @abstractmethod
    def run(self, problem: str) -> dict[str, Any]:
        """Run one problem and return the saved trajectory."""

    def save(self, problem: str, result: dict[str, Any]) -> Path:
        output_dir = self.config.output_dir
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / self.output_name(problem)
        path.write_text(
            json.dumps(drop_secrets(result), ensure_ascii=False, indent=2, default=str) + "\n"
        )
        return path

    def output_name(self, problem: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", problem).strip("-").lower()[:48] or "task"
        digest = hashlib.sha256(problem.encode()).hexdigest()[:8]
        model = re.sub(r"[^a-zA-Z0-9._-]+", "-", self.config.model).strip("-")
        return f"{self.harness_name}__{model}__{slug}-{digest}.json"


SENSITIVE_KEYS = {"api_key", "authorization", "access_token", "password", "secret"}


def drop_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: drop_secrets(item)
            for key, item in value.items()
            if key.lower() not in SENSITIVE_KEYS
        }
    if isinstance(value, list):
        return [drop_secrets(item) for item in value]
    return value
