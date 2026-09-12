from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentic_review_annotation_distilabel.adapters.base import Sample
from agentic_review_annotation_distilabel.annotation.schema import annotation_json_schema
from agentic_review_annotation_distilabel.steps.base import CanonicalStep

PROMPT_VERSION = "annotation_v1"
DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "annotation_v1.md"


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

    def build_payload(self, sample: Sample, steps: list[CanonicalStep]) -> dict[str, Any]:
        return {
            "instance_id": sample.instance_id,
            "repository": sample.repository,
            "environment": sample.environment,
            "task": sample.task,
            "evaluation": sample.evaluation,
            "patch": sample.patch,
            "canonical_steps": [
                {
                    "step_id": step.step_id,
                    "content": step.content,
                }
                for step in steps
            ],
        }

    def build_model_payload(self, sample: Sample, steps: list[CanonicalStep]) -> dict[str, Any]:
        if not self.compact_for_model:
            return self.build_payload(sample, steps)

        compact_steps = []
        used_step_chars = 0
        for step in steps:
            compact_content = compact_step_content(step.content)
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

            compact_steps.append({"step_id": step.step_id, "content": compact_content})
            used_step_chars += len(step_text)

        return {
            "instance_id": sample.instance_id,
            "repository": truncate_data(sample.repository, 4000),
            "environment": truncate_data(sample.environment, 4000),
            "task": truncate_data(sample.task, self.max_task_chars),
            "evaluation": sample.evaluation,
            "patch": truncate_text(sample.patch or "", self.max_patch_chars),
            "canonical_steps": compact_steps,
            "model_input_note": (
                "This is a compact model payload. Full task, patch, raw trajectory, "
                "and full canonical steps are saved in data/normalized for human audit."
            ),
        }


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
