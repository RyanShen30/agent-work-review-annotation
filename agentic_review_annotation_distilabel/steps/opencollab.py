from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentic_review_annotation_distilabel.adapters.base import Sample
from agentic_review_annotation_distilabel.steps.base import CanonicalStep, StepParser


class OpenCollabStepParser(StepParser):
    """Use each completed OpenCollab model turn as one review step."""

    dataset_name = "opencollab"

    def parse(self, sample: Sample) -> list[CanonicalStep]:
        records = _validate_records(sample.trajectory)
        steps: list[CanonicalStep] = []
        latest_by_agent: dict[str, CanonicalStep] = {}
        pending_by_response: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        pending_by_agent: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        pending_response_by_agent: dict[str, str] = {}
        context_events: list[dict[str, Any]] = []

        for index, record in enumerate(records):
            event_type = _event_type(record)
            payload = _payload(record)
            agent_key = _agent_key(payload)

            if event_type == "llm_call_started":
                response_id = payload.get("response_session_id")
                if response_id is not None:
                    response_key = str(response_id)
                    pending_by_response.setdefault(response_key, []).append(
                        (index, record)
                    )
                    if agent_key is not None:
                        pending_response_by_agent[agent_key] = response_key
                elif agent_key is not None:
                    pending_by_agent.setdefault(agent_key, []).append((index, record))
                else:
                    context_events.append(record)
                continue

            if event_type == "llm_call":
                lifecycle: list[dict[str, Any]] = []
                raw_indices = [index]
                response_id = payload.get("response_session_id")
                pending: list[tuple[int, dict[str, Any]]] = []
                if response_id is not None:
                    pending.extend(pending_by_response.pop(str(response_id), []))
                    if agent_key is not None:
                        pending_response_by_agent.pop(agent_key, None)
                if not pending and agent_key is not None:
                    pending.extend(pending_by_agent.pop(agent_key, []))
                for pending_index, pending_record in pending:
                    raw_indices.append(pending_index)
                    lifecycle.append(pending_record)

                actions = _actions(payload)
                step = CanonicalStep(
                    step_id=len(steps) + 1,
                    raw_message_indices=sorted(raw_indices),
                    action_ids=_action_ids(actions),
                    content={
                        "type": "opencollab_agent_turn",
                        "agent": {
                            key: value
                            for key, value in {
                                "aid": payload.get("aid"),
                                "role": payload.get("role") or payload.get("agent"),
                                "session_step": payload.get("session_step"),
                            }.items()
                            if value is not None
                        },
                        "agent_message": {
                            "role": "assistant",
                            "content": payload.get("content"),
                            "reasoning": payload.get("reasoning"),
                            "finish_reason": payload.get("finish_reason"),
                        },
                        "actions": actions,
                        "observations": [],
                        "orchestration_events": [],
                        "runtime_events": lifecycle,
                        "llm_call": record,
                    },
                )
                steps.append(step)
                if agent_key is not None:
                    latest_by_agent[agent_key] = step
                continue

            pending_response = (
                pending_response_by_agent.get(agent_key)
                if agent_key is not None
                else None
            )
            if pending_response is not None:
                pending_by_response[pending_response].append((index, record))
                continue
            if agent_key is not None and agent_key in pending_by_agent:
                pending_by_agent[agent_key].append((index, record))
                continue

            target = _target_step(payload, latest_by_agent)
            if target is None:
                context_events.append(record)
                continue
            target.raw_message_indices.append(index)
            if event_type == "tool_exec":
                target.observation_indices.append(index)
                target.content["observations"].append(record)
            elif _is_orchestration_event(event_type, payload):
                target.content["orchestration_events"].append(record)
            else:
                target.content["runtime_events"].append(record)

        for pending in (*pending_by_response.values(), *pending_by_agent.values()):
            for index, record in pending:
                target = _target_step(_payload(record), latest_by_agent)
                if target is None:
                    context_events.append(record)
                    continue
                target.raw_message_indices.append(index)
                target.content["runtime_events"].append(record)

        if not steps:
            raise ValueError(
                "OpenCollab trajectory contains no completed llm_call records."
            )

        if context_events:
            steps[0].content["context_events"] = context_events
        for step in steps:
            step.raw_message_indices.sort()
            step.observation_indices.sort()
            _drop_empty_step_sections(step.content)
        return steps


def _validate_records(trajectory: Any) -> list[dict[str, Any]]:
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("OpenCollab trajectory must be a non-empty list.")
    records: list[dict[str, Any]] = []
    for index, record in enumerate(trajectory):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"OpenCollab trajectory record {index} must be a JSON object."
            )
        records.append(dict(record))
    return records


def _event_type(record: Mapping[str, Any]) -> str:
    return str(
        record.get("type")
        or record.get("event")
        or record.get("step_type")
        or "unknown"
    )


def _payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = record.get("payload")
    return dict(payload) if isinstance(payload, Mapping) else dict(record)


def _agent_key(payload: Mapping[str, Any]) -> str | None:
    aid = payload.get("aid")
    if aid is not None:
        return f"aid:{aid}"
    role = payload.get("role") or payload.get("agent")
    return f"role:{role}" if role is not None else None


def _target_step(
    payload: Mapping[str, Any], latest_by_agent: Mapping[str, CanonicalStep]
) -> CanonicalStep | None:
    key = _agent_key(payload)
    if key is not None and key in latest_by_agent:
        return latest_by_agent[key]
    from_aid = payload.get("from_aid")
    if from_aid is not None:
        return latest_by_agent.get(f"aid:{from_aid}")
    requester_aid = payload.get("requester_aid")
    if requester_aid is None:
        requester_aid = payload.get("parent_aid")
    if requester_aid is not None:
        return latest_by_agent.get(f"aid:{requester_aid}")
    to_aid = payload.get("to_aid")
    if to_aid is not None:
        return latest_by_agent.get(f"aid:{to_aid}")
    return None


def _actions(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    tool_calls = payload.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [dict(call) for call in tool_calls if isinstance(call, Mapping)]


def _action_ids(actions: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for index, action in enumerate(actions, start=1):
        value = action.get("id") or action.get("tool_call_id") or action.get("name")
        ids.append(str(value if value is not None else index))
    return ids


def _is_orchestration_event(event_type: str, payload: Mapping[str, Any]) -> bool:
    return event_type.startswith(
        ("assigned.", "spawn", "message", "agent_", "worktree")
    ) or any(
        key in payload for key in ("from_aid", "to_aid", "requester_aid", "parent_aid")
    )


def _drop_empty_step_sections(content: dict[str, Any]) -> None:
    message = content.get("agent_message")
    if isinstance(message, dict):
        content["agent_message"] = {
            key: value for key, value in message.items() if value is not None
        }
    for key in ("actions", "observations", "orchestration_events", "runtime_events"):
        if not content.get(key):
            content.pop(key, None)
