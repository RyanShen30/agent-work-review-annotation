from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from agentic_review_annotation_distilabel.adapters.base import (
    DatasetAdapter,
    Sample,
    require_mapping,
)


class OpenCollabAdapter(DatasetAdapter):
    """Normalize an OpenCollab SDK or OpenCollab-Eval run record."""

    dataset_name = "opencollab"

    def adapt(self, raw: Mapping[str, Any]) -> Sample:
        raw = require_mapping(raw)
        trajectory = _extract_trajectory(raw)
        instance_id = _first_present(
            raw,
            "instance_id",
            "task_id",
            "id",
            "run.instance_id",
            "result.instance_id",
            "swebench.instance_id",
        )
        if not instance_id:
            raise ValueError(
                "OpenCollab sample is missing required instance_id or task_id."
            )

        task = _first_present(
            raw,
            "problem",
            "task",
            "problem_statement",
            "description",
            "prompt",
            "run.problem",
            "swebench.problem_statement",
        )
        patch = _first_present(
            raw,
            "generated_patch",
            "model_patch",
            "submission",
            "run.generated_patch",
            "result.patch",
        )
        if patch is None and not _has_oracle_patch(raw):
            patch = raw.get("patch")

        environment = _mapping_or_none(
            _first_present(raw, "environment", "run.environment")
        )
        source = _compact(
            {
                "benchmark": _first_present(
                    raw, "benchmark", "dataset", "source.benchmark"
                ),
                "split": _first_present(raw, "split", "source.split"),
                "repo": _repository_name(raw),
                "base_commit": _first_present(
                    raw, "base_commit", "source.base_commit", "swebench.base_commit"
                ),
                "problem_statement": task,
                "difficulty": _first_present(raw, "difficulty", "source.difficulty"),
            }
        )
        if (
            source.get("benchmark") is None
            and _first_present(raw, "swebench") is not None
        ):
            source["benchmark"] = "SWE-bench"

        result = _mapping_or_none(raw.get("result")) or {}
        evaluation = _compact(
            {
                "status": _first_present(raw, "status", "result.status", "run.status"),
                "reason": _first_present(raw, "reason", "result.reason", "run.reason"),
                "resolved": _first_present(
                    raw, "resolved", "evaluation.resolved", "result.resolved"
                ),
                "metrics": _first_present(raw, "metrics", "result.metrics"),
                "agent_failures": _first_present(
                    raw, "agent_failures", "result.agent_failures"
                ),
                "eval_logs": _first_present(raw, "eval_logs", "evaluation.eval_logs"),
            }
        )
        run = _compact(
            {
                "run_id": _first_present(raw, "run_id", "run.run_id"),
                "harness": "opencollab",
                "harness_version": _first_present(
                    raw, "opencollab_version", "run.harness_version"
                ),
                "model": _first_present(raw, "model", "run.model"),
                "config": _compact(
                    {
                        "mode": _first_present(raw, "mode", "run.mode") or "team",
                        "provider": _first_present(raw, "provider", "run.provider"),
                        "reason": evaluation.get("reason"),
                        "tokens": _first_present(
                            raw, "tokens", "result.tokens", "run.tokens"
                        ),
                        "metrics": evaluation.get("metrics") or result.get("metrics"),
                        "output": _first_present(raw, "output", "result.output"),
                        "trajectory_format": _first_present(raw, "trajectory_format")
                        or "opencollab.trajectory.jsonl",
                    }
                ),
                "exit_status": evaluation.get("status"),
                "generated_patch": patch,
            }
        )
        oracle = _compact(
            {
                "gold_patch": _first_present(
                    raw, "gold_patch", "oracle.gold_patch", "swebench.patch"
                )
                or (raw.get("patch") if _has_oracle_patch(raw) else None),
                "test_patch": _first_present(
                    raw, "test_patch", "oracle.test_patch", "swebench.test_patch"
                ),
                "fail_to_pass": _string_list(
                    _first_present(
                        raw, "FAIL_TO_PASS", "fail_to_pass", "swebench.FAIL_TO_PASS"
                    )
                ),
                "pass_to_pass": _string_list(
                    _first_present(
                        raw, "PASS_TO_PASS", "pass_to_pass", "swebench.PASS_TO_PASS"
                    )
                ),
                "eval_image": _first_present(
                    raw, "image", "eval_image", "docker_image", "swebench.image"
                ),
            }
        )

        repository = _compact(
            {
                "repo": _repository_name(raw),
                "instance_id": str(instance_id),
                "cwd": _first_present(
                    raw,
                    "repo_path",
                    "workspace",
                    "environment.cwd",
                    "environment.workspace",
                ),
            }
        )
        return Sample(
            instance_id=str(instance_id),
            task=task,
            trajectory=trajectory,
            patch=patch,
            evaluation=evaluation,
            raw=raw,
            repository=repository or None,
            environment=environment,
            source=source,
            run=run,
            oracle=oracle,
        )


def _extract_trajectory(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidate = _first_present(
        raw,
        "trajectory",
        "trace",
        "events",
        "artifacts.trajectory",
        "result.trajectory",
    )
    if isinstance(candidate, str):
        try:
            candidate = [
                json.loads(line) for line in candidate.splitlines() if line.strip()
            ]
        except json.JSONDecodeError as exc:
            raise ValueError("OpenCollab trajectory string must be JSONL.") from exc
    if not isinstance(candidate, list) or not candidate:
        raise ValueError("OpenCollab sample must contain a non-empty trajectory list.")
    trajectory: list[dict[str, Any]] = []
    for index, record in enumerate(candidate):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"OpenCollab trajectory record {index} must be a JSON object."
            )
        trajectory.append(dict(record))
    return trajectory


def _repository_name(raw: Mapping[str, Any]) -> Any | None:
    value = _first_present(
        raw,
        "repo",
        "repo_name",
        "repository",
        "source.repo",
        "swebench.repo",
    )
    if isinstance(value, Mapping):
        return value.get("repo") or value.get("name") or value.get("repository")
    return value


def _has_oracle_patch(raw: Mapping[str, Any]) -> bool:
    return any(
        _get_path(raw, path) is not None
        for path in ("gold_patch", "oracle.gold_patch", "swebench.patch", "test_patch")
    )


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _compact(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, [], {})}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return _string_list(json.loads(value))
        except json.JSONDecodeError:
            return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _first_present(raw: Mapping[str, Any], *paths: str) -> Any | None:
    for path in paths:
        value = _get_path(raw, path)
        if value is not None and value != "":
            return value
    return None


def _get_path(value: Any, path: str) -> Any | None:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current
