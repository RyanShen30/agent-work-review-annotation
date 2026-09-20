from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from ..datasets import get_dataset_adapter, load_instances
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
    parser = argparse.ArgumentParser(description="Run coding-agent tasks")
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

    def run_instance(instance: dict[str, Any] | None) -> tuple[Path, bool]:
        worker_config = config.model_copy(deep=True)
        if instance:
            adapter = get_dataset_adapter(
                config.dataset, config.benchmark_path, instance
            )
            worker_config.instance_id = str(instance["instance_id"])
            worker_config.base_commit = instance.get("base_commit")
            worker_config.docker_image = configured_image or adapter.image(instance)
            worker_config.environment_kwargs.setdefault("cwd", adapter.cwd)
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
            raise RuntimeError("export LLM_API_KEY before running")
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
                detail = getattr(exc, "stderr", None) or str(exc)
                failures.append(
                    f"{instance_id}: {type(exc).__name__}: {str(detail).strip()}"
                )
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


if __name__ == "__main__":
    main()
