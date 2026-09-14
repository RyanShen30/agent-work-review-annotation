from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentic_review_annotation_distilabel.adapters.base import Sample
from agentic_review_annotation_distilabel.steps.base import CanonicalStep, StepParser


class MiniSWEAgentStepParser(StepParser):
    dataset_name = "mini_swe_agent"
    assistant_roles = {"assistant", "ai", "model"}
    observation_roles = {"user", "tool"}
    context_roles = {"system"}

    def parse(self, sample: Sample) -> list[CanonicalStep]:
        if not isinstance(sample.trajectory, list):
            raise ValueError("mini-swe-agent trajectory must be a message list.")

        messages = []
        for index, message in enumerate(sample.trajectory):
            if not isinstance(message, Mapping):
                raise ValueError(f"Trajectory message {index} must be a JSON object.")
            messages.append(dict(message))

        context_messages: list[dict[str, Any]] = []
        steps: list[CanonicalStep] = []
        index = 0
        seen_agent_turn = False

        while index < len(messages):
            message = messages[index]
            role = message.get("role")

            if role in self.context_roles or (
                role == "user" and not seen_agent_turn and not steps
            ):
                context_messages.append(message)
                index += 1
                continue

            if role in self.assistant_roles:
                seen_agent_turn = True
                step_messages = [message]
                observations: list[dict[str, Any]] = []
                raw_message_indices = [index]
                observation_indices: list[int] = []
                index += 1
                while index < len(messages):
                    next_message = messages[index]
                    next_role = next_message.get("role")
                    if next_role in self.assistant_roles:
                        break
                    if next_role in self.observation_roles:
                        observations.append(next_message)
                        observation_indices.append(index)
                    else:
                        observations.append(next_message)
                        observation_indices.append(index)
                    step_messages.append(next_message)
                    raw_message_indices.append(index)
                    index += 1

                actions = _extract_actions(message)
                steps.append(
                    CanonicalStep(
                        step_id=len(steps) + 1,
                        raw_message_indices=raw_message_indices,
                        action_ids=_extract_action_ids(actions),
                        observation_indices=observation_indices,
                        content={
                            "type": "agent_turn",
                            "agent_message": message,
                            "actions": actions,
                            "observations": observations,
                            "messages": step_messages,
                        },
                    )
                )
                continue

            context_messages.append(message)
            index += 1

        if not steps:
            raise ValueError("mini-swe-agent trajectory contains no assistant turns.")

        if context_messages:
            steps[0].content["context_messages"] = context_messages

        return steps


def _extract_actions(message: Mapping[str, Any]) -> list[Any]:
    actions: list[Any] = []

    extra = message.get("extra")
    if isinstance(extra, Mapping):
        extra_actions = extra.get("actions")
        if isinstance(extra_actions, list):
            actions.extend(extra_actions)

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        actions.extend(tool_calls)

    return actions


def _extract_action_ids(actions: list[Any]) -> list[str]:
    action_ids: list[str] = []
    for index, action in enumerate(actions, start=1):
        if isinstance(action, Mapping):
            action_id = action.get("id") or action.get("tool_call_id") or action.get("name")
            function = action.get("function")
            if action_id is None and isinstance(function, Mapping):
                action_id = function.get("name")
            if action_id is not None:
                action_ids.append(str(action_id))
                continue
        action_ids.append(str(index))
    return action_ids
