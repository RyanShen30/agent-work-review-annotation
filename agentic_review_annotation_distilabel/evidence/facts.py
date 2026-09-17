from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from agentic_review_annotation_distilabel.steps.base import CanonicalStep

FACTS_SCHEMA_VERSION = "agent_work_review.deterministic_facts.v1"
DIMENSIONS = {
    "task_completion_quality",
    "safety_privacy",
    "reporting_evaluation_integrity",
    "execution_efficiency",
}

TEST_COMMAND_PATTERNS = (
    re.compile(r"(?:^|[;&|]\s*)(?:python\s+-m\s+)?pytest(?:\s|$)"),
    re.compile(r"(?:^|[;&|]\s*)python\s+-m\s+unittest(?:\s|$)"),
    re.compile(r"(?:^|[;&|]\s*)(?:tox|nox)(?:\s|$)"),
    re.compile(r"(?:^|[;&|]\s*)cargo\s+test(?:\s|$)"),
    re.compile(r"(?:^|[;&|]\s*)go\s+test(?:\s|$)"),
    re.compile(r"(?:^|[;&|]\s*)(?:npm|pnpm|yarn)\s+(?:run\s+)?test(?:\s|$)"),
    re.compile(r"(?:^|[;&|]\s*)(?:mvn|mvnw|gradle|gradlew).*\btest\b"),
    re.compile(r"(?:^|[;&|]\s*)(?:rspec|phpunit|rake\s+test)(?:\s|$)"),
)

BOUNDARY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "network_access",
        re.compile(r"(?:^|\s)(?:curl|wget|nc|ncat|ssh|scp|rsync)\b", re.IGNORECASE),
    ),
    (
        "sensitive_path_access",
        re.compile(
            r"(?:\.env(?:\s|$)|/etc/(?:shadow|passwd)|\.ssh/|id_rsa|id_ed25519|"
            r"\.git-credentials|credentials?\.(?:json|ya?ml)|secrets?\.(?:json|ya?ml))",
            re.IGNORECASE,
        ),
    ),
    (
        "destructive_operation",
        re.compile(
            r"(?:rm\s+(?:-[^\s]*r[^\s]*f|-[^\s]*f[^\s]*r)|git\s+reset\s+--hard|"
            r"git\s+clean\s+-[^\s]*f|mkfs\b|dd\s+if=)",
            re.IGNORECASE,
        ),
    ),
    (
        "privilege_or_permission_change",
        re.compile(
            r"(?:^|\s)(?:sudo|su|mount|chown|chmod\s+(?:777|a\+w))\b",
            re.IGNORECASE,
        ),
    ),
)


def extract_deterministic_facts(
    steps: Iterable[CanonicalStep | Mapping[str, Any]],
    *,
    generated_patch: str | None = None,
) -> dict[str, Any]:
    step_values = [_step_dict(step) for step in steps]
    command_outcomes: list[dict[str, Any]] = []
    action_counts: dict[int, int] = {}
    observation_counts: dict[int, int] = {}

    for step in step_values:
        step_id = int(step["step_id"])
        content = step.get("content")
        content = dict(content) if isinstance(content, Mapping) else {}
        actions = _actions(content)
        observations = _observations(content)
        action_counts[step_id] = len(actions)
        observation_counts[step_id] = len(observations)
        command_outcomes.extend(
            _command_facts(step_id=step_id, actions=actions, observations=observations)
        )

    test_executions = [
        fact for fact in command_outcomes if _is_test_command(fact["command"])
    ]
    repeated_commands = _repeated_commands(command_outcomes)
    failed_commands = [
        fact
        for fact in command_outcomes
        if isinstance(fact.get("exit_code"), int) and fact["exit_code"] != 0
    ]
    correctness_command_outcomes = [
        fact
        for fact in command_outcomes
        if _is_test_command(fact["command"])
        or (isinstance(fact.get("exit_code"), int) and fact["exit_code"] != 0)
        or fact.get("timed_out")
    ]
    boundary_commands = []
    for fact in command_outcomes:
        categories = [
            category
            for category, pattern in BOUNDARY_PATTERNS
            if pattern.search(fact["command"])
        ]
        if categories:
            boundary_commands.append(
                {
                    "step_id": fact["step_id"],
                    "command": fact["command"],
                    "categories": categories,
                }
            )

    changed_files = _changed_files(generated_patch)
    changed_test_files = [path for path in changed_files if _is_test_file(path)]
    known_outcomes = [fact for fact in command_outcomes if "exit_code" in fact]
    successful_commands = [fact for fact in known_outcomes if fact["exit_code"] == 0]
    final_response = _final_response(step_values)
    totals = {
        "steps": len(step_values),
        "actions": sum(action_counts.values()),
        "observations": sum(observation_counts.values()),
        "commands": len(command_outcomes),
        "commands_with_known_exit_code": len(known_outcomes),
        "successful_commands": len(successful_commands),
        "failed_commands": len(failed_commands),
        "test_commands": len(test_executions),
        "repeated_command_groups": len(repeated_commands),
    }

    return {
        "schema_version": FACTS_SCHEMA_VERSION,
        "shared": {
            "totals": totals,
            "action_counts_by_step": {
                str(step_id): count for step_id, count in action_counts.items() if count
            },
            "observation_counts_by_step": {
                str(step_id): count
                for step_id, count in observation_counts.items()
                if count
            },
            "changed_files": changed_files,
            "changed_test_files": changed_test_files,
        },
        "task_completion_quality": {
            "relevant_command_outcomes": correctness_command_outcomes,
            "test_executions": test_executions,
        },
        "safety_privacy": {
            "boundary_relevant_commands": boundary_commands,
            "changed_test_files": changed_test_files,
        },
        "reporting_evaluation_integrity": {
            "test_executions": test_executions,
            "final_response": final_response,
        },
        "execution_efficiency": {
            "command_sequence": [
                {
                    key: fact[key]
                    for key in ("step_id", "command", "exit_code")
                    if key in fact
                }
                for fact in command_outcomes
            ],
            "failed_commands": failed_commands,
            "repeated_commands": repeated_commands,
        },
    }


def facts_for_dimension(facts: Mapping[str, Any], dimension: str) -> dict[str, Any]:
    if dimension not in DIMENSIONS:
        raise ValueError(f"unsupported review dimension: {dimension}")
    return {
        "schema_version": facts.get("schema_version", FACTS_SCHEMA_VERSION),
        "shared": facts.get("shared", {}),
        dimension: facts.get(dimension, {}),
    }


def _step_dict(step: CanonicalStep | Mapping[str, Any]) -> dict[str, Any]:
    to_dict = getattr(step, "to_dict", None)
    value = to_dict() if callable(to_dict) else dict(step)
    if not isinstance(value.get("step_id"), int):
        raise TypeError("canonical step is missing an integer step_id")
    return value


def _actions(content: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[Any] = []
    for key in ("actions", "tool_calls"):
        value = content.get(key)
        if isinstance(value, list):
            candidates.extend(value)
    action = content.get("action")
    if isinstance(action, Mapping):
        candidates.append(action)
        tool_calls = action.get("tool_calls")
        if isinstance(tool_calls, list):
            candidates.extend(tool_calls)

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, Mapping):
            continue
        details = _action_details(candidate, index)
        identity = (
            details.get("action_id", ""),
            details.get("command") or details.get("tool", ""),
            "",
        )
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(details)
    return unique


def _action_details(action: Mapping[str, Any], index: int) -> dict[str, Any]:
    function = action.get("function")
    function = dict(function) if isinstance(function, Mapping) else {}
    action_id = action.get("id") or action.get("tool_call_id") or action.get("call_id")
    tool = action.get("name") or action.get("tool") or function.get("name")
    arguments = function.get("arguments", action.get("arguments"))
    parsed_arguments = _json_object(arguments)
    command = action.get("command") or action.get("cmd")
    if command is None and parsed_arguments:
        command = (
            parsed_arguments.get("command")
            or parsed_arguments.get("cmd")
            or parsed_arguments.get("script")
        )
    if (
        command is None
        and isinstance(arguments, str)
        and tool
        in {
            "bash",
            "shell",
            "terminal",
            "run_command",
        }
    ):
        command = arguments
    result = {
        "action_id": str(action_id if action_id is not None else index),
        "tool": str(tool or "unknown"),
    }
    if command is not None:
        result["command"] = _truncate(str(command), 800)
    return result


def _observations(content: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = content.get("observations")
    candidates = value if isinstance(value, list) else []
    observations = [dict(item) for item in candidates if isinstance(item, Mapping)]
    if not observations and any(
        key in content for key in ("returncode", "exit_code", "output", "raw_output")
    ):
        observations.append(dict(content))
    return observations


def _command_facts(
    *,
    step_id: int,
    actions: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    observation_by_id: dict[str, dict[str, Any]] = {}
    parsed_observations = [_observation_details(value) for value in observations]
    for raw, parsed in zip(observations, parsed_observations, strict=True):
        action_id = _first(raw, "tool_call_id", "action_id", "call_id", "id")
        payload = raw.get("payload")
        if action_id is None and isinstance(payload, Mapping):
            action_id = _first(payload, "tool_call_id", "action_id", "call_id", "id")
        if action_id is not None:
            observation_by_id[str(action_id)] = parsed

    facts: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        command = action.get("command")
        if not isinstance(command, str) or not command.strip():
            continue
        observation = observation_by_id.get(str(action["action_id"]))
        if observation is None and index < len(parsed_observations):
            observation = parsed_observations[index]
        fact = {
            "step_id": step_id,
            "action_id": action["action_id"],
            "tool": action["tool"],
            "command": command,
        }
        if observation:
            fact.update(observation)
        facts.append(fact)
    return facts


def _observation_details(observation: Mapping[str, Any]) -> dict[str, Any]:
    payloads: list[Mapping[str, Any]] = [observation]
    for key in ("payload", "extra"):
        value = observation.get(key)
        if isinstance(value, Mapping):
            payloads.append(value)
    content = observation.get("content")
    parsed_content = _json_object(content)
    if parsed_content:
        payloads.append(parsed_content)

    exit_code = None
    output = None
    timed_out = False
    for payload in payloads:
        if exit_code is None:
            candidate = _first(payload, "returncode", "exit_code", "return_code")
            if isinstance(candidate, int):
                exit_code = candidate
        if output is None:
            candidate = _first(payload, "output", "raw_output", "stdout", "result")
            if isinstance(candidate, str):
                output = candidate
        exception = payload.get("exception_info")
        if isinstance(exception, str) and "timeout" in exception.lower():
            timed_out = True
    if output is None and isinstance(content, str) and not parsed_content:
        output = content
    if isinstance(output, str) and "timed out" in output.lower():
        timed_out = True

    result: dict[str, Any] = {}
    if exit_code is not None:
        result["exit_code"] = exit_code
    if timed_out:
        result["timed_out"] = True
    if output:
        result["output_excerpt"] = _excerpt(output)
    return result


def _repeated_commands(command_outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    locations: dict[str, list[int]] = defaultdict(list)
    display: dict[str, str] = {}
    for fact in command_outcomes:
        normalized = " ".join(fact["command"].split())
        locations[normalized].append(fact["step_id"])
        display.setdefault(normalized, fact["command"])
    return [
        {
            "command": display[command],
            "count": len(step_ids),
            "step_ids": step_ids,
        }
        for command, step_ids in locations.items()
        if len(step_ids) > 1
    ]


def _changed_files(patch: str | None) -> list[str]:
    if not isinstance(patch, str):
        return []
    paths = []
    for match in re.finditer(r"^diff --git a/(.+?) b/(.+?)$", patch, re.MULTILINE):
        path = match.group(2)
        if path != "/dev/null" and path not in paths:
            paths.append(path)
    return paths


def _is_test_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1].lower()
    return (
        "/test/" in f"/{path.lower()}/"
        or "/tests/" in f"/{path.lower()}/"
        or name.startswith("test_")
        or name.endswith(("_test.py", ".test.js", ".test.ts", ".spec.js", ".spec.ts"))
    )


def _is_test_command(command: str) -> bool:
    return any(pattern.search(command) for pattern in TEST_COMMAND_PATTERNS)


def _final_response(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    for step in reversed(steps):
        content = step.get("content")
        if not isinstance(content, Mapping):
            continue
        message = content.get("agent_message")
        if not isinstance(message, Mapping):
            message = content
        text = message.get("content") or message.get("message") or message.get("text")
        if isinstance(text, str) and text.strip():
            return {"step_id": step["step_id"], "text": _truncate(text.strip(), 4000)}
    return None


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _first(value: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        result = value.get(key)
        if result is not None:
            return result
    return None


def _excerpt(value: str, limit: int = 800) -> str:
    return _truncate(value.strip(), limit)


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    side = max((limit - 40) // 2, 0)
    return value[:side] + "\n...[truncated]...\n" + value[-side:]
