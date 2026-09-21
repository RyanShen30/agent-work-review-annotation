from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from agentic_review_annotation_distilabel.adapters.base import Sample
from agentic_review_annotation_distilabel.annotation.annotators import ANNOTATION_AGENTS
from agentic_review_annotation_distilabel.annotation.schema import (
    annotation_json_schema,
)
from agentic_review_annotation_distilabel.evidence import (
    extract_deterministic_facts,
    facts_for_dimension,
)
from agentic_review_annotation_distilabel.steps.base import CanonicalStep

PROMPT_VERSION = "annotation_v8_deduplicated_model_input"
DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "annotation_v1.md"
)


class PromptBuilder:
    def __init__(
        self,
        template_path: Path = DEFAULT_PROMPT_PATH,
        *,
        compact_for_model: bool = False,
        max_task_chars: int = 12000,
        max_patch_chars: int = 20000,
        max_step_chars: int = 6000,
        max_total_step_chars: int = 60000,
    ) -> None:
        self.template_path = template_path
        self.compact_for_model = compact_for_model
        self.max_task_chars = max_task_chars
        self.max_patch_chars = max_patch_chars
        self.max_step_chars = max_step_chars
        self.max_total_step_chars = max_total_step_chars

    def build_instruction(self, sample: Sample, steps: list[CanonicalStep]) -> str:
        template = self.template_path.read_text(encoding="utf-8")
        payload = self.build_model_payload(sample, steps)
        return template.replace(
            "{{payload_json}}",
            json.dumps(payload, ensure_ascii=False, indent=2),
        ).replace(
            "{{json_schema}}",
            json.dumps(annotation_json_schema(), ensure_ascii=False, indent=2),
        )

    def build_annotator_instructions(
        self,
        sample: Sample,
        steps: list[CanonicalStep],
    ) -> dict[str, str]:
        return {
            agent.name: agent.build_instruction(
                self.build_specialized_payload(
                    sample,
                    steps,
                    include_evaluation=agent.include_evaluation,
                    dimension=agent.dimension,
                )
            )
            for agent in ANNOTATION_AGENTS
        }

    def build_specialized_payload(
        self,
        sample: Sample,
        steps: list[CanonicalStep],
        *,
        include_evaluation: bool,
        dimension: str,
    ) -> dict[str, Any]:
        payload = self.build_model_payload(sample, steps)
        if not include_evaluation:
            payload.pop("evaluation", None)
        payload["deterministic_facts"] = facts_for_dimension(
            payload["deterministic_facts"], dimension
        )
        return payload

    def build_payload(
        self, sample: Sample, steps: list[CanonicalStep]
    ) -> dict[str, Any]:
        deterministic_facts = extract_deterministic_facts(
            steps, generated_patch=sample.patch
        )
        return {
            "instance_id": sample.instance_id,
            "repository": sample.repository,
            "environment": sample.environment,
            "task": sample.task,
            "evaluation": sample.evaluation,
            "generated_patch": sample.patch,
            "canonical_steps": [step.to_dict() for step in steps],
            "deterministic_facts": deterministic_facts,
        }

    def build_model_payload(
        self, sample: Sample, steps: list[CanonicalStep]
    ) -> dict[str, Any]:
        deterministic_facts = extract_deterministic_facts(
            steps, generated_patch=sample.patch
        )
        model_steps = [model_step_payload(step, task=sample.task) for step in steps]

        if not self.compact_for_model:
            return {
                "instance_id": sample.instance_id,
                "repository": sample.repository,
                "environment": sample.environment,
                "task": sample.task,
                "evaluation": sample.evaluation,
                "generated_patch": sample.patch,
                "canonical_steps": model_steps,
                "deterministic_facts": deterministic_facts,
            }

        compact_steps = []
        used_step_chars = 0
        for step in model_steps:
            compact_content = compact_step_content(step["content"])
            step_text = json.dumps(compact_content, ensure_ascii=False)
            if len(step_text) > self.max_step_chars:
                compact_content = {
                    "truncated": True,
                    "summary": truncate_text(step_text, self.max_step_chars),
                }
                step_text = json.dumps(compact_content, ensure_ascii=False)

            if used_step_chars + len(step_text) > self.max_total_step_chars:
                remaining = max(self.max_total_step_chars - used_step_chars, 0)
                compact_content = {
                    "truncated": True,
                    "summary": truncate_text(step_text, remaining),
                }
                step_text = json.dumps(compact_content, ensure_ascii=False)

            compact_step = dict(step)
            compact_step["content"] = compact_content
            compact_steps.append(compact_step)
            used_step_chars += len(step_text)

        return {
            "instance_id": sample.instance_id,
            "repository": truncate_data(sample.repository, 4000),
            "environment": truncate_data(sample.environment, 4000),
            "task": truncate_data(sample.task, self.max_task_chars),
            "evaluation": sample.evaluation,
            "generated_patch": truncate_text(sample.patch or "", self.max_patch_chars),
            "canonical_steps": compact_steps,
            "deterministic_facts": deterministic_facts,
            "model_input_note": (
                "This is a compact model payload. Full task, generated patch, raw trajectory, "
                "and full canonical steps are saved in output/*/normalized for human audit."
            ),
        }


def model_step_payload(step: CanonicalStep, *, task: Any) -> dict[str, Any]:
    """Project an audit step into a non-redundant model-facing step."""
    return {
        "step_id": step.step_id,
        "content": model_step_content(step.content, task=task),
    }


def model_step_content(content: Any, *, task: Any) -> Any:
    if not isinstance(content, Mapping) or content.get("type") != "agent_turn":
        return content

    projected = {
        key: value
        for key, value in content.items()
        if key
        not in {
            "agent_message",
            "actions",
            "observations",
            "messages",
            "context_messages",
        }
    }
    projected["type"] = "agent_turn"

    agent_message = model_agent_message(content.get("agent_message"))
    if agent_message:
        projected["agent_message"] = agent_message

    actions = model_actions(content.get("actions"))
    if actions:
        projected["actions"] = actions

    observations = content.get("observations")
    if observations:
        projected["observations"] = observations

    context_messages = model_context_messages(
        content.get("context_messages"), task=task
    )
    if context_messages:
        projected["context_messages"] = context_messages

    return projected


def model_agent_message(message: Any) -> Any:
    if not isinstance(message, Mapping):
        return message

    projected = {
        key: value
        for key, value in message.items()
        if key not in {"tool_calls", "extra"}
    }
    extra = message.get("extra")
    if isinstance(extra, Mapping):
        projected_extra = {
            key: value for key, value in extra.items() if key != "actions"
        }
        if projected_extra:
            projected["extra"] = projected_extra
    return projected


def model_actions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    actions: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    anonymous: set[str] = set()
    for raw_action in value:
        if not isinstance(raw_action, Mapping):
            continue
        action = model_action(raw_action)
        action_id = action.get("action_id")
        if isinstance(action_id, str) and action_id:
            if action_id in positions:
                actions[positions[action_id]] = merge_model_actions(
                    actions[positions[action_id]], action
                )
            else:
                positions[action_id] = len(actions)
                actions.append(action)
            continue

        identity = json.dumps(action, ensure_ascii=False, sort_keys=True)
        if identity not in anonymous:
            anonymous.add(identity)
            actions.append(action)
    return actions


def model_action(action: Mapping[str, Any]) -> dict[str, Any]:
    function = action.get("function")
    function = dict(function) if isinstance(function, Mapping) else {}
    action_id = action.get("id") or action.get("tool_call_id") or action.get("call_id")
    tool = action.get("name") or action.get("tool") or function.get("name")
    arguments = function.get("arguments", action.get("arguments"))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass

    command = action.get("command") or action.get("cmd")
    if command is not None:
        if isinstance(arguments, Mapping):
            arguments = dict(arguments)
            arguments.setdefault("command", command)
        elif arguments is None:
            arguments = {"command": command}

    ignored = {
        "id",
        "tool_call_id",
        "call_id",
        "name",
        "tool",
        "function",
        "arguments",
        "command",
        "cmd",
        "index",
        "type",
    }
    details = {key: value for key, value in action.items() if key not in ignored}

    projected: dict[str, Any] = {}
    if action_id is not None:
        projected["action_id"] = str(action_id)
    if tool is not None:
        projected["tool"] = str(tool)
    if arguments is not None:
        projected["arguments"] = arguments
    if details:
        projected["details"] = details
    return projected


def merge_model_actions(
    existing: dict[str, Any], incoming: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        current = merged.get(key)
        if current is None:
            merged[key] = value
        elif (
            key in {"arguments", "details"}
            and isinstance(current, Mapping)
            and isinstance(value, Mapping)
        ):
            combined = dict(current)
            for nested_key, nested_value in value.items():
                combined.setdefault(nested_key, nested_value)
            merged[key] = combined
    return merged


def model_context_messages(value: Any, *, task: Any) -> list[Any]:
    if not isinstance(value, list):
        return []

    task_text = task if isinstance(task, str) and task else None
    projected = []
    for message in value:
        if not isinstance(message, Mapping):
            projected.append(message)
            continue
        item = dict(message)
        content = item.get("content")
        if (
            task_text
            and item.get("role") == "user"
            and isinstance(content, str)
            and task_text in content
        ):
            item["content"] = content.replace(
                task_text,
                "[Task text omitted here; see top-level `task`.]",
                1,
            )
        projected.append(item)
    return projected


def compact_step_content(content: Any) -> Any:
    if not isinstance(content, dict):
        return truncate_data(content, 6000)

    compact: dict[str, Any] = {}
    if content.get("type") == "agent_turn":
        compact["type"] = "agent_turn"
        compact["agent_message"] = truncate_data(content.get("agent_message"), 2500)
        compact["actions"] = truncate_data(content.get("actions"), 1800)
        compact["observations"] = truncate_data(content.get("observations"), 2200)
        if content.get("context_messages"):
            compact["context_messages"] = truncate_data(
                content.get("context_messages"),
                2000,
            )
        return compact

    if "step" in content:
        compact["step"] = content["step"]

    action = content.get("action")
    if isinstance(action, dict):
        compact_action = {
            key: action.get(key)
            for key in ["type", "content", "tool_calls", "reasoning_text"]
            if key in action
        }
        compact["action"] = truncate_data(compact_action, 3500)
    elif action is not None:
        compact["action"] = truncate_data(action, 3500)

    observations = content.get("observations")
    if observations is not None:
        compact["observations"] = truncate_data(observations, 2000)

    return compact


def truncate_data(value: Any, max_chars: int) -> Any:
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= max_chars:
        return value
    return {
        "truncated": True,
        "chars_original": len(text),
        "text": truncate_text(text, max_chars),
    }


def truncate_text(value: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    half = max((max_chars - 80) // 2, 0)
    return (
        value[:half]
        + f"\n...[truncated {len(value) - (2 * half)} chars]...\n"
        + value[-half:]
    )
