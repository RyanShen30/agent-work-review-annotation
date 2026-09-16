from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from .base import AgentConfig, PROJECT_ROOT, trajectory_filename
from .mini_swe_agent import MiniSWEAgent
from .opencollab import OpenCollabAgent
from .openhands import OpenHandsAgent

AGENTS = {
    "mini_swe_agent": MiniSWEAgent,
    "openhands": OpenHandsAgent,
    "opencollab": OpenCollabAgent,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one coding-agent task")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--problem")
    parser.add_argument("--instance")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text()) or {}
    config = AgentConfig.model_validate(raw.get("generation", raw))
    config.api_key = os.getenv("LLM_API_KEY")
    config.base_url = os.getenv("LLM_BASE_URL")
    if args.instance is not None:
        config.instance = (
            int(args.instance) if args.instance.isdigit() else args.instance
        )
    if args.output_dir:
        config.output_dir = args.output_dir
    instances = (
        load_instances(config.benchmark_path, config.instance)
        if config.benchmark_path
        else [None]
    )
    configured_image = config.docker_image
    for instance in instances:
        if instance:
            config.instance_id = str(instance["instance_id"])
            config.base_commit = instance.get("base_commit")
            config.docker_image = (
                configured_image or instance.get("image") or swebench_image(instance)
            )
            config.benchmark_instance = instance
        problem = args.problem or (instance and instance.get("problem_statement"))
        if not problem:
            parser.error("provide --problem or benchmark_path in YAML")
        output_dir = config.output_dir
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        path = output_dir / trajectory_filename(config.harness, config.model, problem)
        if reusable_trajectory(path, config, problem):
            print(f"reusing trajectory: {path}")
            print(path)
            continue
        if not config.api_key:
            parser.error("export LLM_API_KEY before running")
        result = AGENTS[config.harness](config).run(problem)
        print(result["output_path"])


def reusable_trajectory(path: Path, config: AgentConfig, problem: str) -> bool:
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return False
    return (
        isinstance(saved, dict)
        and saved.get("harness") == config.harness
        and saved.get("instance_id") == config.instance_id
        and saved.get("problem") == problem
        and any(isinstance(saved.get(key), list) for key in ("messages", "events", "trajectory"))
        and (
            config.runtime != "docker"
            or (
                isinstance(saved.get("review_workspace"), dict)
                and saved["review_workspace"].get("image") == config.docker_image
                and saved["review_workspace"].get("base_commit") == config.base_commit
            )
        )
    )


def load_instances(path: Path | str, selector: int | str) -> list[dict[str, Any]]:
    if str(selector) == "-1":
        return load_benchmark(path)
    return [load_instance(path, selector)]


def load_benchmark(path: Path | str) -> list[dict[str, Any]]:
    import pandas as pd

    path = Path(path)
    files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {path}")
    rows = pd.concat(pd.read_parquet(file) for file in files)
    return [rows.iloc[index].to_dict() for index in range(len(rows))]


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
