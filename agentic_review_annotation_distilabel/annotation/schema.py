from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


QualityRating = Literal["pass", "warning", "fail", "unknown"]
EfficiencyRating = Literal["high", "normal", "low", "unknown"]
RecoveryStatus = Literal["not_applicable", "unrecovered", "self_corrected", "unknown"]


class DimensionReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: QualityRating
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_blank(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("reason must not be blank")
        return normalized


class TaskCompletionQualityReview(DimensionReview):
    recovery: RecoveryStatus = "not_applicable"


class ExecutionEfficiencyReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: EfficiencyRating
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_blank(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("reason must not be blank")
        return normalized


class StepReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    task_completion_quality: TaskCompletionQualityReview
    safety_privacy: DimensionReview
    reporting_evaluation_integrity: DimensionReview
    execution_efficiency: ExecutionEfficiencyReview


class AnnotationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1)
    step_reviews: list[StepReview] = Field(default_factory=list)


def annotation_to_dict(annotation: AnnotationResult) -> dict[str, Any]:
    return annotation.model_dump()


def annotation_json_schema() -> dict[str, Any]:
    return AnnotationResult.model_json_schema()


def parse_annotation(value: Any) -> AnnotationResult:
    if isinstance(value, AnnotationResult):
        return value
    if isinstance(value, dict):
        return AnnotationResult.model_validate(value)
    if not isinstance(value, str):
        raise TypeError(f"Unsupported annotation value type: {type(value).__name__}")
    return AnnotationResult.model_validate(_extract_json_object(value))


def _extract_json_object(text: str) -> Any:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            repaired = _repair_truncated_annotation(stripped)
            if repaired is not None:
                return repaired
            raise
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            repaired = _repair_truncated_annotation(stripped[start:])
            if repaired is not None:
                return repaired
            raise


def _repair_truncated_annotation(text: str) -> dict[str, Any] | None:
    instance_id = _extract_string_field(text, "instance_id")
    if not instance_id:
        return None

    step_reviews_start = text.find('"step_reviews"')
    if step_reviews_start == -1:
        return None

    array_start = text.find("[", step_reviews_start)
    if array_start == -1:
        return None

    step_reviews = []
    for object_text in _iter_complete_json_objects(text[array_start + 1 :]):
        try:
            step_reviews.append(json.loads(object_text))
        except json.JSONDecodeError:
            continue

    if not step_reviews and '"step"' in text[step_reviews_start:]:
        return None

    return {
        "instance_id": instance_id,
        "step_reviews": step_reviews,
    }


def _extract_string_field(text: str, field: str) -> str | None:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"([^"]*)"', text)
    if not match:
        return None
    return match.group(1)


def _iter_complete_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    depth = 0
    start: int | None = None
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
        elif char == "]" and depth == 0:
            break

    return objects


def validate_annotation_against_steps(
    annotation: AnnotationResult,
    instance_id: str,
    valid_step_ids: list[int],
) -> None:
    if annotation.instance_id != instance_id:
        raise ValueError(
            f"Annotation instance_id {annotation.instance_id!r} does not match "
            f"sample instance_id {instance_id!r}."
        )

    expected = set(valid_step_ids)
    seen: list[int] = [review.step for review in annotation.step_reviews]
    seen_set = set(seen)
    step_counts = Counter(seen)

    duplicate_steps = sorted(
        step_id for step_id, count in step_counts.items() if count > 1
    )
    if duplicate_steps:
        raise ValueError(
            "Annotation contains duplicate step reviews: "
            + ", ".join(str(step_id) for step_id in duplicate_steps)
        )

    invalid_steps = sorted(seen_set - expected)
    if invalid_steps:
        raise ValueError(
            "Annotation contains reviewed steps not present in trajectory: "
            + ", ".join(str(step_id) for step_id in invalid_steps)
        )

    missing_steps = sorted(expected - seen_set)
    if missing_steps:
        raise ValueError(
            "Annotation is missing step reviews for trajectory steps: "
            + ", ".join(str(step_id) for step_id in missing_steps)
        )
