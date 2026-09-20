from __future__ import annotations

from collections import Counter
from typing import Any

from agentic_review_annotation_distilabel.annotation.schema import (
    AnnotationResult,
    CorrectnessAnnotationResult,
    CorrectnessStepReview,
    DimensionReview,
    EfficiencyStepReview,
    ExecutionEfficiencyAnnotationResult,
    ExecutionEfficiencyReview,
    QualityStepReview,
    ReportingIntegrityAnnotationResult,
    RunReviews,
    SafetyPrivacyAnnotationResult,
    TaskCompletionQualityReview,
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
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        raise ValueError("valid_step_ids contains duplicates")

    for name, result in {
        "task_completion_quality": correctness,
        "safety_privacy": safety_privacy,
        "reporting_evaluation_integrity": reporting_integrity,
        "execution_efficiency": execution_efficiency,
    }.items():
        _validate_result_identity(name, result, instance_id)
        _validate_step_reviews(
            name,
            [item.step_id for item in result.step_reviews],
            expected,
        )

    correctness_by_step = {item.step_id: item for item in correctness.step_reviews}
    safety_by_step = {item.step_id: item for item in safety_privacy.step_reviews}
    reporting_by_step = {item.step_id: item for item in reporting_integrity.step_reviews}
    efficiency_by_step = {item.step_id: item for item in execution_efficiency.step_reviews}

    return AnnotationResult(
        instance_id=instance_id,
        step_reviews=[
            {
                "step": step_id,
                "task_completion_quality": _merge_correctness(correctness_by_step[step_id]),
                "safety_privacy": _merge_quality_dimension(safety_by_step[step_id]),
                "reporting_evaluation_integrity": _merge_quality_dimension(
                    reporting_by_step[step_id]
                ),
                "execution_efficiency": _merge_efficiency(efficiency_by_step[step_id]),
            }
            for step_id in expected
        ],
        run_reviews=RunReviews(
            task_completion_quality=correctness.run_review,
            safety_privacy=safety_privacy.run_review,
            reporting_evaluation_integrity=reporting_integrity.run_review,
            execution_efficiency=execution_efficiency.run_review,
        ),
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


def _merge_correctness(item: CorrectnessStepReview) -> TaskCompletionQualityReview:
    return TaskCompletionQualityReview(
        rating=item.rating,
        reason=item.reason,
        recovery=item.recovery,
    )


def _merge_quality_dimension(item: QualityStepReview) -> DimensionReview:
    return DimensionReview(
        rating=item.rating,
        reason=item.reason,
    )


def _merge_efficiency(item: EfficiencyStepReview) -> ExecutionEfficiencyReview:
    return ExecutionEfficiencyReview(
        rating=item.rating,
        reason=item.reason,
    )
