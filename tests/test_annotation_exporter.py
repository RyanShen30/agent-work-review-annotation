import json

import pytest

from agentic_review_annotation_distilabel.annotation.exporter import (
    export_private,
    export_public,
)
from agentic_review_annotation_distilabel.annotation.prompt_builder import PROMPT_VERSION
from agentic_review_annotation_distilabel.annotation.schema import (
    AnnotationResult,
    EfficiencyRunReview,
    EfficiencyStepReview,
    MasterRecord,
    QualityRunReview,
    QualityStepReview,
    validate_annotation_against_steps,
)
from agentic_review_annotation_distilabel.run import (
    is_valid_existing_result,
    load_existing_annotation,
    master_with_auto_annotation,
)


def annotation_payload(instance_id="sample"):
    return {
        "instance_id": instance_id,
        "task_completion_quality": {
            "findings": [
                {"step_id": 1, "rating": "fail", "reason": "Wrong implementation."}
            ],
            "run_review": {"rating": "fail", "reason": "The defect remained."},
        },
        "safety_privacy": {"findings": [], "run_review": {"rating": "pass"}},
        "reporting_evaluation_integrity": {
            "findings": [],
            "run_review": {"rating": "pass"},
        },
        "execution_efficiency": {"findings": [], "run_review": {"rating": "high"}},
    }


def test_public_benchmark_task_hides_annotation_and_oracle():
    annotation = annotation_payload("django__django-11049")
    auto = {key: value for key, value in annotation.items() if key != "instance_id"}
    master = MasterRecord.model_validate(
        {
            "instance_id": "django__django-11049",
            "source": {
                "benchmark": "SWE-bench_Verified",
                "repo": "django/django",
                "base_commit": "abc123",
                "problem_statement": "Fix the regression.",
            },
            "run": {
                "harness": "mini_swe_agent",
                "model": "openai/gpt-5.4",
                "environment": {"image": "swebench/image:latest", "secret": "hidden"},
                "generated_patch": "agent patch",
            },
            "trajectory": {
                "raw_path": "raw.json",
                "raw_sha256": "sha",
                "canonical_steps": [{"step_id": 1, "content": {"type": "agent_turn"}}],
            },
            "deterministic_facts": {
                "schema_version": "agent_work_review.deterministic_facts.v1",
                "shared": {"totals": {"steps": 1}},
            },
            "evaluation": {"resolved": True, "eval_logs": "hidden logs"},
            "oracle": {
                "gold_patch": "gold patch",
                "test_patch": "test patch",
                "fail_to_pass": ["test_fail"],
                "pass_to_pass": ["test_pass"],
            },
            "annotation": {
                "auto": {"model": "annotator", "prompt_version": PROMPT_VERSION, **auto},
                "final": annotation,
            },
            "provenance": {"source_path": "raw.json", "source_sha256": "sha"},
        }
    )

    public = export_public(master)
    private = export_private(master)

    assert public["mode"] == "benchmark_task"
    assert public["generated_patch"] == "agent patch"
    assert public["environment"] == {"image": "swebench/image:latest"}
    assert "annotation" not in public
    assert "final_annotation" not in public
    assert "oracle" not in public
    assert "evaluation" not in public
    assert "deterministic_facts" not in public
    assert "gold patch" not in str(public)
    assert "test_fail" not in str(public)
    assert private["oracle"]["gold_patch"] == "gold patch"
    assert private["deterministic_facts"]["shared"]["totals"]["steps"] == 1
    assert private["annotation"]["final"]["task_completion_quality"]["findings"][0]["step_id"] == 1


def test_public_annotation_release_can_include_final_reviews():
    public = export_public(
        {"instance_id": "sample", "annotation": {"final": annotation_payload()}},
        mode="annotation_release",
    )

    assert public["mode"] == "annotation_release"
    assert public["final_annotation"]["task_completion_quality"]["findings"][0]["step_id"] == 1
    assert public["final_annotation"]["execution_efficiency"]["run_review"] == {
        "rating": "high"
    }


def test_master_persists_auto_dimension_reviews():
    annotation = AnnotationResult.model_validate(annotation_payload())

    updated = master_with_auto_annotation(
        master=MasterRecord(instance_id="sample"),
        annotation=annotation,
        model="review-model",
        prompt_version=PROMPT_VERSION,
    )

    assert updated.annotation.auto is not None
    assert updated.annotation.auto.execution_efficiency.run_review.rating == "high"
    assert updated.annotation.auto.task_completion_quality.findings[0].step_id == 1


def test_existing_annotation_requires_and_loads_current_dimension_reviews(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps(
            {**annotation_payload(), "metadata": {"prompt_version": PROMPT_VERSION}}
        ),
        encoding="utf-8",
    )

    annotation, _ = load_existing_annotation(path)

    assert annotation.task_completion_quality.findings[0].step_id == 1
    assert is_valid_existing_result(path, "sample", [1])

    stale = path.read_text(encoding="utf-8").replace(
        PROMPT_VERSION, "annotation_v8_deduplicated_model_input"
    )
    path.write_text(stale, encoding="utf-8")
    assert not is_valid_existing_result(path, "sample", [1])


@pytest.mark.parametrize(
    ("review_type", "rating", "level"),
    [
        (QualityStepReview, "pass", "step"),
        (EfficiencyStepReview, "high", "step"),
        (QualityRunReview, "pass", "run"),
        (EfficiencyRunReview, "high", "run"),
    ],
)
def test_default_rating_cannot_carry_reason(review_type, rating, level):
    payload = {"rating": rating, "reason": "A concrete problem occurred."}
    if level == "step":
        payload["step_id"] = 1
    with pytest.raises(ValueError, match=f"default {level} reviews must not include a reason"):
        review_type.model_validate(payload)


@pytest.mark.parametrize(
    ("review_type", "rating"),
    [(QualityRunReview, "fail"), (EfficiencyRunReview, "low")],
)
def test_non_default_run_requires_reason(review_type, rating):
    with pytest.raises(ValueError, match="non-default run reviews must include"):
        review_type.model_validate({"rating": rating})


def test_sparse_findings_must_be_non_default_and_reference_canonical_steps():
    payload = annotation_payload()
    annotation = AnnotationResult.model_validate(payload)
    validate_annotation_against_steps(annotation, "sample", [1, 2])

    payload["safety_privacy"]["findings"] = [{"step_id": 2, "rating": "pass"}]
    with pytest.raises(ValueError, match="findings must omit default pass"):
        AnnotationResult.model_validate(payload)

    payload["safety_privacy"]["findings"] = [
        {"step_id": 3, "rating": "warning", "reason": "Unsafe operation."}
    ]
    annotation = AnnotationResult.model_validate(payload)
    with pytest.raises(ValueError, match="not present in trajectory"):
        validate_annotation_against_steps(annotation, "sample", [1, 2])

    payload["safety_privacy"]["findings"] = [
        {"step_id": 2, "rating": "warning", "reason": "First issue."},
        {"step_id": 1, "rating": "fail", "reason": "Second issue."},
    ]
    annotation = AnnotationResult.model_validate(payload)
    with pytest.raises(ValueError, match="not in canonical order"):
        validate_annotation_against_steps(annotation, "sample", [1, 2])
