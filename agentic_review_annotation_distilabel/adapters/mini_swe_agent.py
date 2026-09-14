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
        source = _extract_source(raw)
        run = _extract_run(raw, config, environment)
        oracle = _extract_oracle(raw)
        generated_patch = _extract_generated_patch(raw, has_oracle=bool(oracle))

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
            patch=generated_patch,
            evaluation=_extract_evaluation(raw),
            raw=raw,
            repository=_extract_repository(raw, environment),
            environment=environment,
            source=source,
            run={
                **run,
                "generated_patch": generated_patch,
            },
            oracle=oracle,
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


def _extract_source(raw: Mapping[str, Any]) -> dict[str, Any]:
    benchmark = _first_present(
        raw,
        "benchmark",
        "dataset",
        "source.benchmark",
        "swebench.benchmark",
        "info.benchmark.name",
        "info.swebench.benchmark",
    )
    if isinstance(benchmark, Mapping):
        benchmark = benchmark.get("name")

    repo = _first_present(
        raw,
        "repo",
        "repository",
        "swebench.repo",
        "info.swebench.repo",
    )
    if isinstance(repo, Mapping):
        repo = repo.get("repo") or repo.get("repository") or repo.get("name")

    source = {
        "benchmark": benchmark,
        "split": _first_present(raw, "split", "source.split", "swebench.split"),
        "repo": repo,
        "base_commit": _first_present(raw, "base_commit", "swebench.base_commit"),
        "environment_setup_commit": _first_present(
            raw,
            "environment_setup_commit",
            "swebench.environment_setup_commit",
        ),
        "problem_statement": _first_present(
            raw,
            "problem_statement",
            "problem",
            "task",
            "swebench.problem_statement",
        ),
        "hints_text": _first_present(raw, "hints_text", "swebench.hints_text"),
        "created_at": _first_present(raw, "created_at", "swebench.created_at"),
        "version": _first_present(raw, "version", "swebench.version"),
        "difficulty": _first_present(raw, "difficulty", "swebench.difficulty"),
    }
    if source.get("benchmark") is None and _has_swebench_fields(raw):
        source["benchmark"] = (
            "SWE-bench_Verified" if source.get("difficulty") is not None else "SWE-bench"
        )
    return {key: value for key, value in source.items() if value is not None}


def _extract_run(
    raw: Mapping[str, Any],
    config: Mapping[str, Any],
    environment: Mapping[str, Any] | None,
) -> dict[str, Any]:
    info = raw.get("info") if isinstance(raw.get("info"), Mapping) else {}
    model_config = config.get("model") if isinstance(config.get("model"), Mapping) else {}
    run = {
        "run_id": _first_present(raw, "run_id", "id", "info.run_id", "output_path"),
        "harness": _first_present(raw, "harness", "info.harness") or "mini_swe_agent",
        "harness_version": _first_present(raw, "info.mini_version", "mini_version"),
        "model": _first_present(raw, "model", "model_name", "info.model_name")
        or model_config.get("model_name"),
        "config": dict(config) if isinstance(config, Mapping) else {},
        "environment": dict(environment) if isinstance(environment, Mapping) else {},
        "exit_status": _first_present(
            raw,
            "outcome.exit_status",
            "info.exit_status",
            "exit_status",
        ),
        "cost": _first_present(
            raw,
            "cost",
            "info.cost",
            "info.model_stats.instance_cost",
            "info.model_stats.cost",
        ),
        "api_calls": _first_present(
            raw,
            "api_calls",
            "info.api_calls",
            "info.model_stats.api_calls",
        ),
    }
    if run["cost"] is not None:
        run["cost"] = float(run["cost"])
    if run["api_calls"] is not None:
        run["api_calls"] = int(run["api_calls"])
    return {key: value for key, value in run.items() if value not in (None, {}, [])}


def _extract_oracle(raw: Mapping[str, Any]) -> dict[str, Any]:
    oracle = {
        "gold_patch": _extract_gold_patch(raw),
        "test_patch": _first_present(raw, "test_patch", "swebench.test_patch"),
        "fail_to_pass": _string_list(
            _first_present(
                raw,
                "FAIL_TO_PASS",
                "fail_to_pass",
                "swebench.FAIL_TO_PASS",
                "swebench.fail_to_pass",
            )
        ),
        "pass_to_pass": _string_list(
            _first_present(
                raw,
                "PASS_TO_PASS",
                "pass_to_pass",
                "swebench.PASS_TO_PASS",
                "swebench.pass_to_pass",
            )
        ),
        "eval_type": _first_present(raw, "eval_type", "swebench.eval_type"),
        "eval_image": _first_present(
            raw,
            "image",
            "eval_image",
            "swebench.image",
            "swebench.eval_image",
        ),
        "eval_script": _first_present(raw, "eval_script", "swebench.eval_script"),
        "log_parser": _first_present(raw, "log_parser", "swebench.log_parser"),
    }
    return {key: value for key, value in oracle.items() if value not in (None, [], {})}


def _extract_generated_patch(raw: Mapping[str, Any], *, has_oracle: bool) -> Any | None:
    generated_patch = _first_present(
        raw,
        "info.submission",
        "submission",
        "generated_patch",
        "model_patch",
        "run.generated_patch",
    )
    if generated_patch is not None:
        return generated_patch
    if isinstance(raw.get("swebench"), Mapping):
        return raw.get("patch")
    if not has_oracle:
        return raw.get("patch")
    return None


def _extract_gold_patch(raw: Mapping[str, Any]) -> Any | None:
    gold_patch = _first_present(raw, "gold_patch", "oracle.gold_patch", "swebench.patch")
    if gold_patch is not None:
        return gold_patch
    if _has_swebench_fields(raw):
        return raw.get("patch")
    return None


def _has_swebench_fields(raw: Mapping[str, Any]) -> bool:
    return any(
        _get_path(raw, path) is not None
        for path in (
            "base_commit",
            "problem_statement",
            "test_patch",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
            "swebench",
        )
    )


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            import json

            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return _string_list(parsed)
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


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
            "swebench.repo",
        ),
        "instance_id": _first_present(
            raw,
            "instance_id",
            "info.instance_id",
            "swebench.instance_id",
        ),
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
