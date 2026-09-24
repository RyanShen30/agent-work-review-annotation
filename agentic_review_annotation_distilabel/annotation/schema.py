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
    reason: str | None = Field(default=None, min_length=1)

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: Any) -> Any:
        return _normalize_optional_reason(value)

    @model_validator(mode="after")
    def validate_reason(self) -> QualityRunReview:
        _require_reason_matching_rating(self.rating, "pass", self.reason, "run")
        return self


class EfficiencyRunReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: EfficiencyRating
    reason: str | None = Field(default=None, min_length=1)

    @field_validator("reason", mode="before")
    @classmethod
    def normalize_reason(cls, value: Any) -> Any:
        return _normalize_optional_reason(value)

    @model_validator(mode="after")
    def validate_reason(self) -> EfficiencyRunReview:
        _require_reason_matching_rating(self.rating, "high", self.reason, "run")
        return self


class RunReviews(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_completion_quality: QualityRunReview
    safety_privacy: QualityRunReview
    reporting_evaluation_integrity: QualityRunReview
    execution_efficiency: EfficiencyRunReview


class CorrectnessStepReview(TaskCompletionQualityReview):
    step_id: int

    @model_validator(mode="before")
    @classmethod
    def reject_false_recovery(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("recovery") is False:
            raise ValueError("recovery must be true when present")
        return value

    @model_validator(mode="after")
    def validate_step_review(self) -> CorrectnessStepReview:
        _require_reason_matching_rating(self.rating, "pass", self.reason, "step")
        if self.recovery is True and self.rating not in {"warning", "fail"}:
            raise ValueError("recovery can only appear on warning or fail step reviews")
        return self


class QualityStepReview(DimensionReview):
    step_id: int

    @model_validator(mode="after")
    def validate_step_review(self) -> QualityStepReview:
        _require_reason_matching_rating(self.rating, "pass", self.reason, "step")
        return self


class EfficiencyStepReview(ExecutionEfficiencyReview):
    step_id: int

    @model_validator(mode="after")
    def validate_step_review(self) -> EfficiencyStepReview:
        _require_reason_matching_rating(self.rating, "high", self.reason, "step")
        return self


class CorrectnessAnnotationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1)
    review_complete: Literal[True]
    run_review: QualityRunReview
    step_reviews: list[CorrectnessStepReview]


class SafetyPrivacyAnnotationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1)
    review_complete: Literal[True]
    run_review: QualityRunReview
    step_reviews: list[QualityStepReview]


class ReportingIntegrityAnnotationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1)
    review_complete: Literal[True]
    run_review: QualityRunReview
    step_reviews: list[QualityStepReview]


class ExecutionEfficiencyAnnotationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1)
    review_complete: Literal[True]
    run_review: EfficiencyRunReview
    step_reviews: list[EfficiencyStepReview]


class CorrectnessDimensionAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[CorrectnessStepReview] = Field(default_factory=list)
    run_review: QualityRunReview

    @model_validator(mode="after")
    def only_problems(self) -> CorrectnessDimensionAnnotation:
        _require_non_default_findings(self.findings, "pass")
        return self


class QualityDimensionAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[QualityStepReview] = Field(default_factory=list)
    run_review: QualityRunReview

    @model_validator(mode="after")
    def only_problems(self) -> QualityDimensionAnnotation:
        _require_non_default_findings(self.findings, "pass")
        return self


class EfficiencyDimensionAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[EfficiencyStepReview] = Field(default_factory=list)
    run_review: EfficiencyRunReview

    @model_validator(mode="after")
    def only_problems(self) -> EfficiencyDimensionAnnotation:
        _require_non_default_findings(self.findings, "high")
        return self


class DimensionAnnotations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_completion_quality: CorrectnessDimensionAnnotation
    safety_privacy: QualityDimensionAnnotation
    reporting_evaluation_integrity: QualityDimensionAnnotation
    execution_efficiency: EfficiencyDimensionAnnotation


class AnnotationResult(DimensionAnnotations):
    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(min_length=1)


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


class AutoAnnotationRecord(DimensionAnnotations):
    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    prompt_version: str | None = None


class AnnotationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    auto: AutoAnnotationRecord | None = None
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


def _normalize_optional_reason(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return " ".join(value.split()) or None


def _require_reason_matching_rating(
    rating: str,
    default_rating: str,
    reason: str | None,
    level: str,
) -> None:
    if rating != default_rating and reason is None:
        raise ValueError(f"non-default {level} reviews must include a non-empty reason")
    if rating == default_rating and reason is not None:
        raise ValueError(f"default {level} reviews must not include a reason")


def _require_non_default_findings(findings: list[Any], default_rating: str) -> None:
    if any(finding.rating == default_rating for finding in findings):
        raise ValueError(f"findings must omit default {default_rating} step reviews")


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
    for name in (
        "task_completion_quality",
        "safety_privacy",
        "reporting_evaluation_integrity",
        "execution_efficiency",
    ):
        seen = [finding.step_id for finding in getattr(annotation, name).findings]
        duplicates = sorted(step_id for step_id, count in Counter(seen).items() if count > 1)
        if duplicates:
            raise ValueError(f"{name} contains duplicate findings: {duplicates}")
        invalid = sorted(set(seen) - expected)
        if invalid:
            raise ValueError(f"{name} references steps not present in trajectory: {invalid}")
        seen_set = set(seen)
        canonical_order = [step_id for step_id in valid_step_ids if step_id in seen_set]
        if seen != canonical_order:
            raise ValueError(f"{name} findings are not in canonical order")
