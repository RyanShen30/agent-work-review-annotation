from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class FailureAnnotation(BaseModel):
    step: int
    reason: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    recovery: Literal["unrecovered", "self_corrected", "unknown"]

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_certainty(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if "confidence" in value or "certainty" not in value:
            return value

        migrated = dict(value)
        certainty = migrated.pop("certainty")
        if certainty == "certain":
            migrated["confidence"] = 0.9
        elif certainty == "unclear":
            migrated["confidence"] = 0.5
        return migrated

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_blank(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("reason must not be blank")
        return normalized


class AnnotationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1)
    final_outcome: Literal["correct", "incorrect"]
    failures: list[FailureAnnotation] = Field(default_factory=list)

    @model_validator(mode="after")
    def merge_duplicate_failure_steps(self) -> AnnotationResult:
        merged: dict[int, FailureAnnotation] = {}
        for failure in self.failures:
            current = merged.get(failure.step)
            if current is None or failure.confidence > current.confidence:
                merged[failure.step] = failure
            elif current is not None and failure.recovery == "unrecovered":
                current.recovery = "unrecovered"
        self.failures = list(merged.values())
        return self


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
    final_outcome = _extract_string_field(text, "final_outcome")
    if not instance_id or final_outcome not in {"correct", "incorrect"}:
        return None

    failures_start = text.find('"failures"')
    if failures_start == -1:
        return {
            "instance_id": instance_id,
            "final_outcome": final_outcome,
            "failures": [],
        }

    array_start = text.find("[", failures_start)
    if array_start == -1:
        return None

    failures = []
    for object_text in _iter_complete_json_objects(text[array_start + 1 :]):
        try:
            failures.append(json.loads(object_text))
        except json.JSONDecodeError:
            continue

    if not failures and '"step"' in text[failures_start:]:
        return None

    return {
        "instance_id": instance_id,
        "final_outcome": final_outcome,
        "failures": failures,
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

    valid = set(valid_step_ids)
    invalid_steps = [
        failure.step for failure in annotation.failures if failure.step not in valid
    ]
    if invalid_steps:
        raise ValueError(
            "Annotation contains failure steps not present in trajectory: "
            + ", ".join(str(step_id) for step_id in sorted(set(invalid_steps)))
        )
