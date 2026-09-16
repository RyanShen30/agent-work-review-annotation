from __future__ import annotations

import argparse
import atexit
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trajectory generation, review, or both.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "example.yaml")
    parser.add_argument("--mode", choices=["traj-only", "review-only", "full"])
    parser.add_argument("--input", type=Path, help="Trajectory file/directory for review-only mode.")
    parser.add_argument("--instance", help="Benchmark row index or instance_id override.")
    parser.add_argument("--runner", choices=["llm", "mock"], help="Review runner override.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run review for existing annotations.")
    parser.add_argument("--set", action="append", default=[], metavar="PATH=VALUE")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    apply_overrides(config, args.set)
    config_path = effective_config(config, args.config, bool(args.set))
    mode = args.mode or config.get("mode", "full")
    env = model_env()

    trajectory: Path | None = None
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
        output_dir = ROOT / "output" / ("traj" if mode == "traj-only" else "pipeline/traj")
        command = [
            str(python), "-m", "agentic_review_annotation_distilabel.agents.run",
            "--config", str(config_path), "--output-dir", str(output_dir),
        ]
        if args.instance is not None:
            command += ["--instance", args.instance]
        output = run(command, env)
        selector = args.instance if args.instance is not None else generation.get("instance")
        if str(selector) == "-1":
            trajectory = output_dir
        else:
            trajectory = Path(output.splitlines()[-1]) if output else None
            if trajectory is None or not trajectory.is_file() or trajectory.parent != output_dir:
                raise FileNotFoundError(f"trajectory was not created under {output_dir}")

    if mode in {"review-only", "full"}:
        base = ROOT / "output" / ("annotation" if mode == "review-only" else "pipeline")
        review_input = args.input or trajectory or Path(config.get("review", {}).get("input", ROOT / "output/traj"))
        command = [
            sys.executable, "-m", "agentic_review_annotation_distilabel.run",
            "--config", str(config_path), "--input", str(review_input),
            "--output-dir", str(base / "annotation"),
            "--normalized-dir", str(base / "normalized"),
            "--normalized-preview-dir", str(base / "preview"),
            "--public-dir", str(base / "public"),
            "--private-dir", str(base / "private"),
            "--cache-dir", str(base / "cache"),
        ]
        if mode == "full" or args.overwrite:
            command += ["--overwrite"]
        if args.runner:
            command += ["--runner", args.runner]
        run(command, env)


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


def effective_config(config: dict, original: Path, changed: bool) -> Path:
    if not changed:
        return original
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(config, handle)
    handle.close()
    path = Path(handle.name)
    atexit.register(path.unlink, missing_ok=True)
    return path


def model_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MSWEA_SILENT_STARTUP", "1")
    env.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
    return env


def run(command: list[str], env: dict[str, str]) -> str:
    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode:
        raise SystemExit(result.stderr or result.returncode)
    return result.stdout.strip()


if __name__ == "__main__":
    main()
