from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from agentic_review_annotation_distilabel.environment import load_environment

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trajectory generation, review, or both."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "example.yaml")
    parser.add_argument("--mode", choices=["traj-only", "review-only", "full"])
    parser.add_argument(
        "--input", type=Path, help="Trajectory file/directory for review-only mode."
    )
    parser.add_argument(
        "--instance", help="Benchmark row index or instance_id override."
    )
    parser.add_argument(
        "--runner", choices=["llm", "mock"], help="Review runner override."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run review for existing annotations.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        help="Maximum workers used inside each pipeline phase.",
    )
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    apply_overrides(config, args.set)
    workers_changed = configure_n_workers(config, args.n_workers)
    cleanup_policy = config.get("cleanup_policy", "on_success")
    validate_cleanup_policy(cleanup_policy)
    config_path = effective_config(
        config, args.config, bool(args.set) or workers_changed
    )
    mode = args.mode or config.get("mode", "full")
    env = model_env()

    trajectory: Path | None = None
    generated_trajectories: list[Path] = []
    if mode in {"traj-only", "full"}:
        generation = config.get("generation", {})
        harness = generation.get("harness", "mini_swe_agent")
        thirdparty_names = {
            "mini_swe_agent": "mini-swe-agent",
            "openhands": "openhands",
            "opencollab": "opencollab",
        }
        try:
            thirdparty_name = thirdparty_names[harness]
        except KeyError as exc:
            raise ValueError(f"unsupported generation harness: {harness}") from exc
        python = (
            ROOT
            / "agentic_review_annotation_distilabel"
            / "thirdparty"
            / thirdparty_name
            / ".venv"
            / "bin"
            / "python"
        )
        output_dir = (
            ROOT / "output" / ("traj" if mode == "traj-only" else "pipeline/traj")
        )
        command = [
            str(python),
            "-m",
            "agentic_review_annotation_distilabel.agents.run",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
        if args.instance is not None:
            command += ["--instance", args.instance]
        output = run(command, env)
        generated_trajectories = trajectory_paths_from_output(output, output_dir)
        selector = (
            args.instance if args.instance is not None else generation.get("instance")
        )
        if str(selector) == "-1":
            trajectory = output_dir
        else:
            trajectory = Path(output.splitlines()[-1]) if output else None
            if (
                trajectory is None
                or not trajectory.is_file()
                or trajectory.parent != output_dir
            ):
                raise FileNotFoundError(
                    f"trajectory was not created under {output_dir}"
                )

    if mode in {"review-only", "full"}:
        base = ROOT / "output" / ("annotation" if mode == "review-only" else "pipeline")
        review_input = (
            args.input
            or trajectory
            or Path(config.get("review", {}).get("input", ROOT / "output/traj"))
        )
        command = [
            sys.executable,
            "-m",
            "agentic_review_annotation_distilabel.run",
            "--config",
            str(config_path),
            "--input",
            str(review_input),
            "--output-dir",
            str(base / "annotation"),
            "--normalized-dir",
            str(base / "normalized"),
            "--normalized-preview-dir",
            str(base / "preview"),
            "--public-dir",
            str(base / "public"),
            "--private-dir",
            str(base / "private"),
            "--cache-dir",
            str(base / "cache"),
        ]
        if mode == "full" or args.overwrite:
            command += ["--overwrite"]
        if args.runner:
            command += ["--runner", args.runner]
        pipeline_succeeded = False
        try:
            evaluation = config.get("evaluation", {})
            if mode == "full" and bool(evaluation.get("enabled", False)):
                evaluation_command = [
                    sys.executable,
                    "-m",
                    "agentic_review_annotation_distilabel.evaluation.swebench",
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(base / "evaluation"),
                    "--input",
                    *(str(path) for path in generated_trajectories),
                ]
                run(evaluation_command, env)
            run(command, env)
            pipeline_succeeded = True
        finally:
            if mode == "full" and should_cleanup(cleanup_policy, pipeline_succeeded):
                cleanup_docker_images(generated_trajectories)


def apply_overrides(config: dict, overrides: list[str]) -> None:
    for override in overrides:
        path, separator, raw_value = override.partition("=")
        if not separator:
            raise ValueError(f"invalid --set {override!r}; expected PATH=VALUE")
        target = config
        parts = path.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = yaml.safe_load(raw_value)


def configure_n_workers(config: dict, override: int | None) -> bool:
    if override is None and "n_workers" not in config:
        return False
    n_workers = int(override if override is not None else config["n_workers"])
    if n_workers <= 0:
        raise ValueError("n_workers must be greater than zero")
    config["n_workers"] = n_workers
    config.setdefault("generation", {})["n_workers"] = n_workers
    config.setdefault("evaluation", {})["max_workers"] = n_workers
    config.setdefault("review", {})["n_workers"] = n_workers
    return True


def effective_config(config: dict, original: Path, changed: bool) -> Path:
    if not changed:
        return original
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        path = Path(handle.name)
    atexit.register(path.unlink, missing_ok=True)
    return path


def model_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MSWEA_SILENT_STARTUP", "1")
    env.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
    return env


def trajectory_paths_from_output(output: str, output_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for line in output.splitlines():
        candidate = Path(line.strip())
        if candidate.is_file() and candidate.parent == output_dir:
            paths.append(candidate)
    return list(dict.fromkeys(paths))


def should_cleanup(policy: str, succeeded: bool) -> bool:
    validate_cleanup_policy(policy)
    return policy == "always" or (policy == "on_success" and succeeded)


def validate_cleanup_policy(policy: str) -> None:
    if policy not in {"always", "on_success", "never"}:
        raise ValueError("cleanup_policy must be one of: always, on_success, never")


def cleanup_docker_images(trajectories: list[Path]) -> None:
    snapshot_images: list[str] = []
    other_images: list[str] = []
    base_images: list[str] = []
    for path in trajectories:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            continue
        workspace = payload.get("review_workspace")
        if not isinstance(workspace, dict):
            continue
        managed = workspace.get("cleanup_images")
        if not isinstance(managed, list):
            continue
        managed_images = [
            image for image in managed if isinstance(image, str) and image
        ]
        snapshot_image = workspace.get("snapshot_image")
        base_image = workspace.get("image")
        for image in managed_images:
            if image == snapshot_image:
                snapshot_images.append(image)
            elif image == base_image:
                base_images.append(image)
            else:
                other_images.append(image)

    images = dict.fromkeys(snapshot_images + other_images + base_images)
    for image in images:
        result = subprocess.run(
            ["docker", "image", "rm", "-f", image],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode and "No such image" not in result.stderr:
            print(f"warning: could not remove image {image}: {result.stderr.strip()}")


def run(command: list[str], env: dict[str, str]) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode:
        raise SystemExit(result.stderr or result.returncode)
    return result.stdout.strip()


if __name__ == "__main__":
    load_environment(ROOT / ".env")
    main()
