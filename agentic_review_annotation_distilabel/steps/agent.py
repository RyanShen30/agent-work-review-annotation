from __future__ import annotations

from collections.abc import Mapping

from agentic_review_annotation_distilabel.adapters.base import Sample
from agentic_review_annotation_distilabel.steps.base import CanonicalStep, StepParser


class AgentStepParser(StepParser):
    def parse(self, sample: Sample) -> list[CanonicalStep]:
        if not isinstance(sample.trajectory, list) or not sample.trajectory:
            raise ValueError("Agent trajectory must be a non-empty list.")
        if not all(isinstance(item, Mapping) for item in sample.trajectory):
            raise ValueError("Agent trajectory items must be JSON objects.")
        return [
            CanonicalStep(
                step_id=index + 1,
                content=dict(item),
                raw_message_indices=[index],
            )
            for index, item in enumerate(sample.trajectory)
        ]
