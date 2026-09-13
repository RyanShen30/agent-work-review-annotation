from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentic_review_annotation_distilabel.adapters.base import (
    DatasetAdapter,
    Sample,
    require_mapping,
)


class MiniSWEAgentAdapter(DatasetAdapter):
    dataset_name = "mini_swe_agent"

    def adapt(self, raw: Mapping[str, Any]) -> Sample:
        raw = require_mapping(raw)
        messages = _extract_messages(raw)
        instance_id = _first_present(
            raw,
            "instance_id",
            "info.instance_id",
            "info.swebench.instance_id",
            "info.benchmark.instance_id",
            "info.task_id",
        )
        if not instance_id:
            raise ValueError("mini-swe-agent sample is missing required instance_id.")

        info = raw.get("info") if isinstance(raw.get("info"), Mapping) else {}
        config = info.get("config") if isinstance(info.get("config"), Mapping) else {}
        environment = (
            config.get("environment")
            if isinstance(config.get("environment"), Mapping)
            else None
        )

        return Sample(
            instance_id=str(instance_id),
            task=_first_present(
                raw,
                "problem",
                "task",
                "problem_statement",
                "info.problem",
                "info.problem_statement",
            )
            or _extract_first_user_text(messages),
            trajectory=messages,
            patch=_first_present(
                raw,
                "info.submission",
                "submission",
                "generated_patch",
                "model_patch",
                "patch",
            ),
            evaluation=_extract_evaluation(raw),
            raw=raw,
            repository=_extract_repository(raw, environment),
            environment=environment,
        )


def _extract_messages(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidate = raw.get("messages")
    if candidate is None:
        candidate = raw.get("trajectory")
    if not isinstance(candidate, list):
        raise ValueError("mini-swe-agent sample must contain a messages list.")

    messages: list[dict[str, Any]] = []
    for index, message in enumerate(candidate):
        if not isinstance(message, Mapping):
            raise ValueError(f"Message {index} must be a JSON object.")
        message = dict(message)
        message.pop("provider_specific_fields", None)
        if isinstance(message.get("extra"), Mapping):
            message["extra"] = dict(message["extra"])
            message["extra"].pop("response", None)
        messages.append(message)
    if not messages:
        raise ValueError("mini-swe-agent messages list must not be empty.")
    return messages


def _extract_first_user_text(messages: list[dict[str, Any]]) -> str | None:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = _message_text(message)
        if content:
            return content
    return None


def _extract_evaluation(raw: Mapping[str, Any]) -> dict[str, Any]:
    info = raw.get("info") if isinstance(raw.get("info"), Mapping) else {}
    outcome = raw.get("outcome") if isinstance(raw.get("outcome"), Mapping) else {}
    evaluation = {
        "exit_status": _first_present(
            raw,
            "outcome.exit_status",
            "info.exit_status",
            "exit_status",
        ),
        "outcome": outcome or None,
        "target": raw.get("target"),
        "resolved": raw.get("resolved"),
        "eval_result": raw.get("eval_result"),
        "eval_logs": raw.get("eval_logs"),
        "model_name": _first_present(raw, "info.model_name", "model_name"),
        "model_stats": info.get("model_stats") if isinstance(info, Mapping) else None,
        "trajectory_format": raw.get("trajectory_format"),
    }
    return {key: value for key, value in evaluation.items() if value is not None}


def _extract_repository(
    raw: Mapping[str, Any],
    environment: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    repository = {
        "repo": _first_present(
            raw,
            "repo",
            "repository",
            "repo_name",
            "info.repo",
            "info.repository",
            "info.swebench.repo",
        ),
        "instance_id": _first_present(raw, "instance_id", "info.instance_id"),
    }
    if environment:
        for key in ("repo", "repository", "repo_name", "cwd", "working_dir"):
            if key in environment and repository.get(key) is None:
                repository[key] = environment[key]
    repository = {key: value for key, value in repository.items() if value is not None}
    return repository or None


def _first_present(raw: Mapping[str, Any], *paths: str) -> Any | None:
    for path in paths:
        value = _get_path(raw, path)
        if value is not None and value != "":
            return value
    return None


def _get_path(value: Any, path: str) -> Any | None:
    current = value
    for part in path.split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        else:
            return None
    return current


def _message_text(message: Mapping[str, Any]) -> str | None:
    content = message.get("content")
    if content is None:
        content = message.get("text")
    if isinstance(content, str) and content != "None":
        return content
    return None
