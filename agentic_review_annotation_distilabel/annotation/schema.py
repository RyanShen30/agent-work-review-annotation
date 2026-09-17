from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

QualityRating = Literal["pass", "warning", "fail", "unknown"]
EfficiencyRating = Literal["high", "normal", "low", "unknown"]


class DimensionReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: QualityRating
    reason: str | None = Field(default=None, min_length=1)

    @field_validator("reason", mode="before")
    @classmethod
    def blank_reason_is_absent(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str) and not " ".join(value.split()):
            return None
        return value

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized


class TaskCompletionQualityReview(DimensionReview):
    recovery: Literal[True] | None = None

    @field_validator("recovery", mode="before")
    @classmethod
    def normalize_legacy_recovery(cls, value: Any) -> Literal[True] | None:
        if value is True:
            return True
        if isinstance(value, str):
            if value == "self_corrected":
                return True
            if value in {"not_applicable", "unrecovered", "unknown"}:
                return None
        return None


class ExecutionEfficiencyReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: EfficiencyRating
    reason: str | None = Field(default=None, min_length=1)

    @field_validator("reason", mode="before")
    @classmethod
    def blank_reason_is_absent(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str) and not " ".join(value.split()):
            return None
        return value

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split())
        return normalized


class StepReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    task_completion_quality: TaskCompletionQualityReview
    safety_privacy: DimensionReview
    reporting_evaluation_integrity: DimensionReview
    execution_efficiency: ExecutionEfficiencyReview


class QualityRunReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: QualityRating
    reason: str = Field(min_length=1)

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: Any) -> Any:
        if not isinstance(value, str) or not " ".join(value.split()):
            raise ValueError("run-level reviews must include a non-empty reason")
        return " ".join(value.split())


class EfficiencyRunReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: EfficiencyRating
    reason: str = Field(min_length=1)

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: Any) -> Any:
        if not isinstance(value, str) or not " ".join(value.split()):
            raise ValueError("run-level reviews must include a non-empty reason")
        return " ".join(value.split())


class RunReviews(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_completion_quality: QualityRunReview
    safety_privacy: QualityRunReview
    reporting_evaluation_integrity: QualityRunReview
    execution_efficiency: EfficiencyRunReview


class CorrectnessStepFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: int
    rating: Literal["warning", "fail", "unknown"]
    reason: str = Field(min_length=1)
    recovery: Literal[True] | None = None

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: Any) -> Any:
        return _normalize_required_reason(value)

    @model_validator(mode="after")
    def validate_recovery(self) -> CorrectnessStepFinding:
        if self.recovery is True and self.rating not in {"warning", "fail"}:
            raise ValueError("recovery can only appear on warning or fail findings")
        return self


class QualityStepFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: int
    rating: Literal["warning", "fail", "unknown"]
    reason: str = Field(min_length=1)

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: Any) -> Any:
        return _normalize_required_reason(value)


class EfficiencyStepFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: int
    rating: Literal["normal", "low", "unknown"]
    reason: str = Field(min_length=1)

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: Any) -> Any:
        return _normalize_required_reason(value)


class CorrectnessAnnotationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1)
    review_complete: Literal[True]
    run_review: QualityRunReview
    findings: list[CorrectnessStepFinding] = Field(default_factory=list)


class SafetyPrivacyAnnotationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1)
    review_complete: Literal[True]
    run_review: QualityRunReview
    findings: list[QualityStepFinding] = Field(default_factory=list)


class ReportingIntegrityAnnotationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1)
    review_complete: Literal[True]
    run_review: QualityRunReview
    findings: list[QualityStepFinding] = Field(default_factory=list)


class ExecutionEfficiencyAnnotationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1)
    review_complete: Literal[True]
    run_review: EfficiencyRunReview
    findings: list[EfficiencyStepFinding] = Field(default_factory=list)


class AnnotationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1)
    step_reviews: list[StepReview] = Field(default_factory=list)
    run_reviews: RunReviews | None = None


class SourceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark: str | None = None
    split: str | None = None
    repo: str | None = None
    base_commit: str | None = None
    environment_setup_commit: str | None = None
    problem_statement: Any | None = None
    hints_text: str | None = None
    created_at: str | None = None
    version: str | None = None
    difficulty: str | None = None


class RunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str | None = None
    harness: str | None = None
    harness_version: str | None = None
    model: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    exit_status: str | None = None
    generated_patch: Any | None = None
    cost: float | None = None
    api_calls: int | None = None


class TrajectoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_path: str | None = None
    raw_sha256: str | None = None
    canonical_steps: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str | None = None
    runner: str | None = None
    runner_version: str | None = None
    run_id: str | None = None
    resolved: bool | None = None
    per_test_results: list[dict[str, Any]] = Field(default_factory=list)
    official_report: dict[str, Any] = Field(default_factory=dict)
    eval_logs: Any | None = None


class OracleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gold_patch: Any | None = None
    test_patch: Any | None = None
    fail_to_pass: list[str] = Field(default_factory=list)
    pass_to_pass: list[str] = Field(default_factory=list)
    eval_type: str | None = None
    eval_image: str | None = None
    eval_script: Any | None = None
    log_parser: Any | None = None


class AutoAnnotationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    prompt_version: str | None = None
    step_reviews: list[StepReview] = Field(default_factory=list)
    run_reviews: RunReviews | None = None


class AnnotationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto: AutoAnnotationRecord = Field(default_factory=AutoAnnotationRecord)
    final: AnnotationResult | None = None


class ProvenanceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_path: str | None = None
    source_sha256: str | None = None
    created_by_pipeline: str = "agentic_review_annotation_distilabel"


class MasterRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "agent_work_review.master.v1"
    instance_id: str = Field(min_length=1)
    source: SourceRecord = Field(default_factory=SourceRecord)
    run: RunRecord = Field(default_factory=RunRecord)
    trajectory: TrajectoryRecord = Field(default_factory=TrajectoryRecord)
    deterministic_facts: dict[str, Any] = Field(default_factory=dict)
    evaluation: EvaluationRecord = Field(default_factory=EvaluationRecord)
    oracle: OracleRecord = Field(default_factory=OracleRecord)
    annotation: AnnotationRecord = Field(default_factory=AnnotationRecord)
    provenance: ProvenanceRecord = Field(default_factory=ProvenanceRecord)


def annotation_to_dict(annotation: AnnotationResult) -> dict[str, Any]:
    return annotation.model_dump(exclude_none=True)


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


def parse_specialized_annotation(
    value: Any,
    result_type: type[
        CorrectnessAnnotationResult
        | SafetyPrivacyAnnotationResult
        | ReportingIntegrityAnnotationResult
        | ExecutionEfficiencyAnnotationResult
    ],
    *,
    instance_id: str | None = None,
) -> (
    CorrectnessAnnotationResult
    | SafetyPrivacyAnnotationResult
    | ReportingIntegrityAnnotationResult
    | ExecutionEfficiencyAnnotationResult
):
    if isinstance(value, result_type):
        return value
    payload = _extract_json_object(value) if isinstance(value, str) else value
    return result_type.model_validate(payload)


def _blank_string_to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and not " ".join(value.split()):
        return None
    return value


def _normalize_optional_reason(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(value.split())


def _normalize_required_reason(value: Any) -> str:
    if not isinstance(value, str) or not " ".join(value.split()):
        raise ValueError("findings must include a non-empty reason")
    return " ".join(value.split())


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
