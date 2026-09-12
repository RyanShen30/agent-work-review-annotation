from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .base import AgentConfig
from .mini_swe_agent import MiniSWEAgent
from .openhands import OpenHandsAgent

AGENTS = {"mini_swe_agent": MiniSWEAgent, "openhands": OpenHandsAgent}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one coding-agent task")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--problem")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    raw = yaml.safe_load(args.config.read_text()) or {}
    config = AgentConfig.model_validate(raw)
    config.api_key = os.getenv("LLM_API_KEY")
    if not config.api_key:
        parser.error("set LLM_API_KEY in the project .env")
    instance = load_instance(config.benchmark_path, config.instance) if config.benchmark_path else None
    if instance:
        config.instance_id = str(instance["instance_id"])
        config.base_commit = instance.get("base_commit")
        config.docker_image = config.docker_image or instance.get("image") or swebench_image(instance)
    problem = args.problem or (instance and instance.get("problem_statement"))
    if not problem:
        parser.error("provide --problem or benchmark_path in YAML")
    result = AGENTS[config.harness](config).run(problem)
    print(result["output_path"])


def load_instance(path: Path | str, selector: int | str) -> dict[str, Any]:
    import pandas as pd

    path = Path(path)
    files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {path}")
    rows = pd.concat(pd.read_parquet(file) for file in files)
    if isinstance(selector, int) or str(selector).isdigit():
        return rows.iloc[int(selector)].to_dict()
    matches = rows[rows["instance_id"] == selector]
    if matches.empty:
        raise KeyError(f"benchmark instance not found: {selector}")
    return matches.iloc[0].to_dict()


def swebench_image(instance: dict[str, Any]) -> str:
    instance_id = str(instance["instance_id"]).replace("__", "_1776_").lower()
    return f"swebench/sweb.eval.x86_64.{instance_id}:latest"


if __name__ == "__main__":
    main()
