from __future__ import annotations

from collections import Counter
from typing import Any

from agentic_review_annotation_distilabel.annotation.schema import (
    AnnotationResult,
    CorrectnessAnnotationResult,
    ExecutionEfficiencyAnnotationResult,
    ReportingIntegrityAnnotationResult,
    SafetyPrivacyAnnotationResult,
)


def merge_specialized_annotations(
    *,
    instance_id: str,
    valid_step_ids: list[int],
    correctness: CorrectnessAnnotationResult,
    safety_privacy: SafetyPrivacyAnnotationResult,
    reporting_integrity: ReportingIntegrityAnnotationResult,
    execution_efficiency: ExecutionEfficiencyAnnotationResult,
) -> AnnotationResult:
    expected = list(valid_step_ids)
    if len(expected) != len(set(expected)):
        raise ValueError("valid_step_ids contains duplicates")

    results = {
        "task_completion_quality": correctness,
        "safety_privacy": safety_privacy,
        "reporting_evaluation_integrity": reporting_integrity,
        "execution_efficiency": execution_efficiency,
    }
    for name, result in results.items():
        _validate_result_identity(name, result, instance_id)
        _validate_step_reviews(
            name,
            [item.step_id for item in result.step_reviews],
            expected,
        )

    return AnnotationResult(
        instance_id=instance_id,
        **{
            name: {
                "findings": [
                    item.model_dump(exclude_none=True)
                    for item in result.step_reviews
                    if item.rating != ("high" if name == "execution_efficiency" else "pass")
                ],
                "run_review": result.run_review.model_dump(exclude_none=True),
            }
            for name, result in results.items()
        },
    )


def _validate_result_identity(name: str, result: Any, instance_id: str) -> None:
    if result.instance_id != instance_id:
        raise ValueError(
            f"{name} result instance_id {result.instance_id!r} does not match {instance_id!r}"
        )


def _validate_step_reviews(name: str, step_ids: list[int], expected: list[int]) -> None:
    expected_set = set(expected)
    counts = Counter(step_ids)
    duplicates = sorted(step_id for step_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"{name} result contains duplicate step ids: "
            + ", ".join(str(step_id) for step_id in duplicates)
        )
    seen = set(step_ids)
    invalid = sorted(seen - expected_set)
    if invalid:
        raise ValueError(
            f"{name} result references unknown step ids: "
            + ", ".join(str(step_id) for step_id in invalid)
        )
    missing = sorted(expected_set - seen)
    if missing:
        raise ValueError(
            f"{name} result is missing step ids: "
            + ", ".join(str(step_id) for step_id in missing)
        )
    if step_ids != expected:
        raise ValueError(f"{name} result step ids are not in canonical order")
