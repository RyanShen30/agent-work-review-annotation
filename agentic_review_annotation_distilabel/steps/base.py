from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentic_review_annotation_distilabel.adapters.base import Sample


@dataclass(frozen=True)
class CanonicalStep:
    step_id: int
    content: Any


class StepParser:
    dataset_name: str

    def parse(self, sample: Sample) -> list[CanonicalStep]:
        raise NotImplementedError

