from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agentic_review_annotation_distilabel.agents.run import load_benchmark

REQUIRED_INSTANCE_FIELDS = {
    "instance_id",
    "image",
    "repo",
    "version",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "log_parser",
    "eval_type",
    "eval_script",
}


@dataclass(frozen=True)
class EvaluationConfig:
    timeout: int = 1800
    max_workers: int = 1
    open_file_limit: int = 4096

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> EvaluationConfig:
        runner = value.get("runner", "official_swebench")
        if runner != "official_swebench":
            raise ValueError(f"unsupported evaluation runner: {runner}")
        config = cls(
            timeout=int(value.get("timeout", 1800)),
            max_workers=int(value.get("max_workers", 1)),
            open_file_limit=int(value.get("open_file_limit", 4096)),
        )
        if config.timeout <= 0:
            raise ValueError("evaluation.timeout must be greater than zero")
        if config.max_workers <= 0:
            raise ValueError("evaluation.max_workers must be greater than zero")
        if config.open_file_limit <= 0:
            raise ValueError("evaluation.open_file_limit must be greater than zero")
        return config


def evaluate_trajectories(
    trajectory_paths: Iterable[Path],
    *,
    benchmark_path: Path,
    output_dir: Path,
    config: EvaluationConfig,
    python_executable: str = sys.executable,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[Path]:
    paths = [Path(path).resolve() for path in trajectory_paths]
    if not paths:
        raise ValueError("official evaluation requires at least one trajectory")
    if importlib.util.find_spec("swebench") is None:
        raise RuntimeError(
            "The official SWE-bench harness is not installed; run `uv sync` or "
            "install the project dependencies before evaluation."
        )
    runner_version = importlib.metadata.version("swebench")

    trajectories = [_load_json(path) for path in paths]
    instance_ids = [_required_text(raw, "instance_id") for raw in trajectories]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("official evaluation received duplicate instance_id values")

    benchmark_rows = {
        str(row.get("instance_id")): row for row in load_benchmark(benchmark_path)
    }
    missing = [
        instance_id for instance_id in instance_ids if instance_id not in benchmark_rows
    ]
    if missing:
        raise KeyError(
            "trajectory instances are missing from the configured benchmark: "
            + ", ".join(missing)
        )

    selected_rows = [
        _jsonable(benchmark_rows[instance_id]) for instance_id in instance_ids
    ]
    _validate_instances(selected_rows)
    predictions = [
        _prediction(raw, instance_id)
        for raw, instance_id in zip(trajectories, instance_ids, strict=True)
    ]
    empty = [pred["instance_id"] for pred in predictions if not pred["model_patch"]]
    if empty:
        raise ValueError(
            "official SWE-bench evaluation cannot grade an empty generated patch: "
            + ", ".join(empty)
        )

    run_id = _run_id(selected_rows, predictions, runner_version)
    run_dir = output_dir.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = run_dir / "dataset.json"
    predictions_path = run_dir / "predictions.jsonl"
    report_dir = run_dir / "reports"
    harness_log = run_dir / "harness.log"
    _write_json(dataset_path, selected_rows)
    predictions_path.write_text(
        "".join(json.dumps(pred, ensure_ascii=False) + "\n" for pred in predictions),
        encoding="utf-8",
    )

    command = [
        python_executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(dataset_path),
        "--split",
        "test",
        "--instance_ids",
        *instance_ids,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(config.max_workers),
        "--open_file_limit",
        str(config.open_file_limit),
        "--timeout",
        str(config.timeout),
        "--run_id",
        run_id,
        "--report_dir",
        str(report_dir),
    ]
    result = command_runner(
        command,
        cwd=run_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    combined_output = (result.stdout or "") + (result.stderr or "")
    harness_log.write_text(combined_output, encoding="utf-8")
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode:
        raise RuntimeError(
            f"official SWE-bench evaluation exited with code {result.returncode}; "
            f"see {harness_log}"
        )

    run_report_path = (
        report_dir
        / f"{predictions[0]['model_name_or_path'].replace('/', '__')}.{run_id}.json"
    )
    run_report = _load_json(run_report_path) if run_report_path.is_file() else {}
    pending_updates: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for path, raw, prediction in zip(paths, trajectories, predictions, strict=True):
        instance_id = prediction["instance_id"]
        artifact_dir = (
            run_dir
            / "logs"
            / "run_evaluation"
            / run_id
            / prediction["model_name_or_path"].replace("/", "__")
            / instance_id
        )
        report_path = artifact_dir / "report.json"
        test_output_path = artifact_dir / "test_output.txt"
        instance_log_path = artifact_dir / "run_instance.log"
        report = _instance_report(
            instance_id=instance_id,
            report_path=report_path,
            test_output_path=test_output_path,
            instance_log_path=instance_log_path,
            harness_log=harness_log,
            run_report=run_report,
        )

        evaluation = {
            "status": report.pop("evaluation_status", "completed"),
            "runner": "swebench.harness.run_evaluation",
            "runner_version": runner_version,
            "run_id": run_id,
            "resolved": report["resolved"],
            "per_test_results": _per_test_results(report),
            "official_report": report,
            "eval_logs": {
                "report_path": _relative_to_output(report_path, output_dir),
                "test_output_path": _relative_to_output(test_output_path, output_dir),
                "instance_log_path": _relative_to_output(instance_log_path, output_dir),
                "harness_log_path": _relative_to_output(harness_log, output_dir),
            },
        }
        pending_updates.append((path, raw, evaluation))

    updated: list[Path] = []
    for path, raw, evaluation in pending_updates:
        raw["evaluation"] = evaluation
        _write_json_atomic(path, raw)
        updated.append(path)
        print(
            f"official evaluation: {raw['instance_id']} "
            f"resolved={evaluation['resolved']}"
        )
    return updated


def _prediction(raw: dict[str, Any], instance_id: str) -> dict[str, str]:
    patch = _first_present(
        raw,
        "generated_patch",
        "model_patch",
        "submission",
        "info.submission",
        "run.generated_patch",
        "patch",
    )
    if patch is not None and not isinstance(patch, str):
        raise TypeError(f"generated patch for {instance_id} must be a string")
    model = _first_present(raw, "model", "model_name", "info.model_name") or "unknown"
    harness = str(raw.get("harness") or "agent")
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{harness}__{model}").strip("-.")
    return {
        "instance_id": instance_id,
        "model_name_or_path": label or "agent-work-review",
        "model_patch": patch or "",
    }


def _validate_instances(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        missing = sorted(
            field for field in REQUIRED_INSTANCE_FIELDS if row.get(field) is None
        )
        if missing:
            instance_id = row.get("instance_id", "<unknown>")
            raise ValueError(
                f"{instance_id} is not compatible with the official SWE-bench harness; "
                f"missing fields: {', '.join(missing)}"
            )


def _instance_report(
    *,
    instance_id: str,
    report_path: Path,
    test_output_path: Path,
    instance_log_path: Path,
    harness_log: Path,
    run_report: dict[str, Any],
) -> dict[str, Any]:
    if report_path.is_file():
        report_map = _load_json(report_path)
        report = report_map.get(instance_id)
        if not isinstance(report, dict) or not isinstance(report.get("resolved"), bool):
            raise TypeError(f"invalid official report for {instance_id}: {report_path}")
        return dict(report)

    if instance_id in run_report.get("infra_failure_ids", []):
        reason = run_report.get("failure_reasons", {}).get(instance_id, "unknown")
        raise RuntimeError(
            f"official SWE-bench evaluation hit an infrastructure failure for "
            f"{instance_id} ({reason}); see {instance_log_path}"
        )

    log_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (instance_log_path, test_output_path)
        if path.is_file()
    )
    if ">>>>> Patch Apply Failed" in log_text:
        return {
            "evaluation_status": "patch_apply_failed",
            "report_kind": "derived_from_official_logs",
            "patch_is_None": False,
            "patch_exists": True,
            "patch_successfully_applied": False,
            "resolved": False,
            "infra_failure": False,
            "failure_reason": "patch_apply_failed",
        }
    if "Timeout error:" in log_text:
        return {
            "evaluation_status": "test_timeout",
            "report_kind": "derived_from_official_logs",
            "patch_is_None": False,
            "patch_exists": True,
            "patch_successfully_applied": True,
            "resolved": False,
            "infra_failure": None,
            "failure_reason": "tests_timed_out",
            "failure_attribution": "ambiguous",
        }
    raise RuntimeError(
        f"official SWE-bench evaluation produced no report for {instance_id}; "
        f"see {instance_log_path if instance_log_path.exists() else harness_log}"
    )


def _per_test_results(report: dict[str, Any]) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    tests_status = report.get("tests_status")
    if not isinstance(tests_status, dict):
        return results
    for category in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        category_result = tests_status.get(category)
        if not isinstance(category_result, dict):
            continue
        for bucket, status in (("success", "passed"), ("failure", "failed")):
            tests = category_result.get(bucket)
            if not isinstance(tests, list):
                continue
            results.extend(
                {"test": str(test), "category": category, "status": status}
                for test in tests
            )
    return results


def _run_id(
    rows: list[dict[str, Any]],
    predictions: list[dict[str, str]],
    runner_version: str,
) -> str:
    payload = {
        "dataset": rows,
        "predictions": predictions,
        "runner_version": runner_version,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return f"agent-work-review-{digest}"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    return str(value)


def _first_present(value: dict[str, Any], *paths: str) -> Any | None:
    for path in paths:
        current: Any = value
        for part in path.split("."):
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(part)
        if current is not None and current != "":
            return current
    return None


def _required_text(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"trajectory is missing required {key}")
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, value)
    os.replace(temporary, path)


def _relative_to_output(path: Path, output_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(output_dir.resolve().parent))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate generated patches with the official SWE-bench harness."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    args = parser.parse_args()

    raw_config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    generation = raw_config.get("generation") or {}
    evaluation = raw_config.get("evaluation") or {}
    benchmark_path = generation.get("benchmark_path")
    if not benchmark_path:
        parser.error("generation.benchmark_path is required for official evaluation")
    evaluate_trajectories(
        args.input,
        benchmark_path=Path(benchmark_path),
        output_dir=args.output_dir,
        config=EvaluationConfig.from_mapping(evaluation),
    )


if __name__ == "__main__":
    main()
