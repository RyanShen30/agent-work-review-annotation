from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentic_review_annotation_distilabel.adapters.base import Sample


@dataclass(frozen=True)
class CanonicalStep:
    step_id: int
    content: Any
    raw_message_indices: list[int] = field(default_factory=list)
    action_ids: list[str] = field(default_factory=list)
    observation_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "step_id": self.step_id,
            "content": self.content,
        }
        if self.raw_message_indices:
            payload["raw_message_indices"] = self.raw_message_indices
        if self.action_ids:
            payload["action_ids"] = self.action_ids
        if self.observation_indices:
            payload["observation_indices"] = self.observation_indices
        return payload


class StepParser:
    dataset_name: str

    def parse(self, sample: Sample) -> list[CanonicalStep]:
        raise NotImplementedError
