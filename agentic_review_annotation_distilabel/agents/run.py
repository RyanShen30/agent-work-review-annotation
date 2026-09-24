from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from .base import AgentConfig, PROJECT_ROOT, trajectory_filename
from agentic_review_annotation_distilabel.environment import (
    load_environment,
    model_credentials,
)
from .mini_swe_agent import MiniSWEAgent
from .opencollab import OpenCollabAgent
from .openhands import OpenHandsAgent

AGENTS = {
    "mini_swe_agent": MiniSWEAgent,
    "openhands": OpenHandsAgent,
    "opencollab": OpenCollabAgent,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run coding-agent tasks")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--problem")
    parser.add_argument("--instance")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text()) or {}
    config = AgentConfig.model_validate(raw.get("generation", raw))
    config.api_key, config.base_url = model_credentials("GENERATION")
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

    def run_instance(instance: dict[str, Any] | None) -> tuple[Path, bool]:
        worker_config = config.model_copy(deep=True)
        if instance:
            worker_config.instance_id = str(instance["instance_id"])
            worker_config.base_commit = instance.get("base_commit")
            worker_config.docker_image = (
                configured_image or instance.get("image") or swebench_image(instance)
            )
            worker_config.benchmark_instance = instance
        problem = args.problem or (instance and instance.get("problem_statement"))
        if not problem:
            raise ValueError("provide --problem or benchmark_path in YAML")
        output_dir = worker_config.output_dir
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        path = output_dir / trajectory_filename(
            worker_config.harness, worker_config.model, problem
        )
        if reusable_trajectory(path, worker_config, problem):
            return path, True
        if not worker_config.api_key:
            raise RuntimeError(
                "set GENERATION_LLM_API_KEY or the legacy LLM_API_KEY before running"
            )
        result = AGENTS[worker_config.harness](worker_config).run(problem)
        return Path(result["output_path"]), False

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=config.n_workers) as executor:
        futures = {
            executor.submit(run_instance, instance): instance for instance in instances
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Generating trajectories",
            unit="task",
            disable=len(futures) == 1,
        ):
            instance = futures[future]
            instance_id = str((instance or {}).get("instance_id") or "manual-task")
            try:
                path, reused = future.result()
            except Exception as exc:
                failures.append(f"{instance_id}: {type(exc).__name__}: {exc}")
                print(f"generation failed: {failures[-1]}", file=sys.stderr)
                continue
            if reused:
                print(f"reusing trajectory: {path}")
            print(path, flush=True)

    if failures:
        raise SystemExit(
            f"generation incomplete: {len(failures)}/{len(instances)} case(s) failed; "
            "rerun the same command to resume"
        )


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
    load_environment(PROJECT_ROOT / ".env")
    main()
