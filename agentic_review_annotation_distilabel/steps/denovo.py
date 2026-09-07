from __future__ import annotations

from typing import Mapping

from agentic_review_annotation_distilabel.adapters.base import Sample
from agentic_review_annotation_distilabel.steps.base import CanonicalStep, StepParser


class DeNovoSWEStepParser(StepParser):
    dataset_name = "denovo"

    def parse(self, sample: Sample) -> list[CanonicalStep]:
        if not isinstance(sample.trajectory, list):
            raise ValueError("DeNovoSWE trajectory must be a list of step objects.")

        steps: list[CanonicalStep] = []
        seen: set[int] = set()
        for index, raw_step in enumerate(sample.trajectory):
            if not isinstance(raw_step, Mapping):
                raise ValueError(f"Trajectory item {index} must be a JSON object.")

            raw_step_id = raw_step.get("step", index)
            if not isinstance(raw_step_id, int):
                raise ValueError(f"Trajectory item {index} has non-integer step id.")
            if raw_step_id in seen:
                raise ValueError(f"Duplicate step id in trajectory: {raw_step_id}")
            if raw_step_id != index:
                raise ValueError(
                    "DeNovoSWE step ids must be continuous. "
                    f"Expected {index}, got {raw_step_id}."
                )

            seen.add(raw_step_id)
            steps.append(CanonicalStep(step_id=raw_step_id, content=dict(raw_step)))

        if not steps:
            raise ValueError("Trajectory must contain at least one step.")
        return steps
