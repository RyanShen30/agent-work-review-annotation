import json
from types import SimpleNamespace

import pytest

from agentic_review_annotation_distilabel.annotation.annotators import ANNOTATION_AGENTS
from agentic_review_annotation_distilabel.annotation.merge import (
    _validate_step_reviews,
    merge_specialized_annotations,
)
from agentic_review_annotation_distilabel.annotation.prompt_builder import (
    PROMPT_VERSION,
    PromptBuilder,
)
from agentic_review_annotation_distilabel.annotation.schema import (
    CorrectnessAnnotationResult,
    ExecutionEfficiencyAnnotationResult,
    ReportingIntegrityAnnotationResult,
    SafetyPrivacyAnnotationResult,
)
from agentic_review_annotation_distilabel.pipelines import (
    DistilabelPipelineConfig,
    run_annotation_pipeline,
)
from agentic_review_annotation_distilabel.pipelines.distilabel_pipeline import (
    _merge_annotator_generations,
)
from agentic_review_annotation_distilabel.steps.base import CanonicalStep


def test_merges_full_specialized_results_into_step_and_run_reviews():
    merged = merge_specialized_annotations(
        instance_id="sample",
        valid_step_ids=[1, 2],
        correctness=CorrectnessAnnotationResult.model_validate(
            {
                "instance_id": "sample",
                "review_complete": True,
                "run_review": {
                    "rating": "warning",
                    "reason": "A correctness problem was later recovered.",
                },
                "step_reviews": [
                    {"step_id": 1, "rating": "pass"},
                    {
                        "step_id": 2,
                        "rating": "fail",
                        "reason": "The step changed the wrong API.",
                        "recovery": True,
                    },
                ],
            }
        ),
        safety_privacy=SafetyPrivacyAnnotationResult.model_validate(
            {
                "instance_id": "sample",
                "review_complete": True,
                "run_review": {"rating": "pass", "reason": "No safety issue."},
                "step_reviews": [
                    {"step_id": 1, "rating": "pass"},
                    {"step_id": 2, "rating": "pass"},
                ],
            }
        ),
        reporting_integrity=ReportingIntegrityAnnotationResult.model_validate(
            {
                "instance_id": "sample",
                "review_complete": True,
                "run_review": {
                    "rating": "warning",
                    "reason": "The final report contained an unsupported test claim.",
                },
                "step_reviews": [
                    {"step_id": 1, "rating": "pass"},
                    {
                        "step_id": 2,
                        "rating": "warning",
                        "reason": "The step claimed tests passed without running them.",
                    },
                ],
            }
        ),
        execution_efficiency=ExecutionEfficiencyAnnotationResult.model_validate(
            {
                "instance_id": "sample",
                "review_complete": True,
                "run_review": {
                    "rating": "low",
                    "reason": "The run repeated ineffective work.",
                },
                "step_reviews": [
                    {"step_id": 1, "rating": "high"},
                    {
                        "step_id": 2,
                        "rating": "low",
                        "reason": "The step repeated the same failed command.",
                    },
                ],
            }
        ),
    )

    dumped = merged.model_dump(exclude_none=True)

    assert dumped["step_reviews"][0] == {
        "step": 1,
        "task_completion_quality": {"rating": "pass"},
        "safety_privacy": {"rating": "pass"},
        "reporting_evaluation_integrity": {"rating": "pass"},
        "execution_efficiency": {"rating": "high"},
    }
    assert dumped["step_reviews"][1]["task_completion_quality"] == {
        "rating": "fail",
        "reason": "The step changed the wrong API.",
        "recovery": True,
    }
    assert dumped["step_reviews"][1]["execution_efficiency"]["rating"] == "low"
    assert dumped["run_reviews"]["task_completion_quality"] == {
        "rating": "warning",
        "reason": "A correctness problem was later recovered.",
    }
    assert dumped["run_reviews"]["execution_efficiency"]["rating"] == "low"


def test_rejects_invalid_specialized_outputs_before_merge():
    with pytest.raises(ValueError, match="non-default step reviews must include"):
        CorrectnessAnnotationResult.model_validate(
            {
                "instance_id": "sample",
                "review_complete": True,
                "run_review": {"rating": "pass", "reason": "No issue."},
                "step_reviews": [{"step_id": 1, "rating": "warning"}],
            }
        )

    with pytest.raises(ValueError, match="recovery must be true when present"):
        CorrectnessAnnotationResult.model_validate(
            {
                "instance_id": "sample",
                "review_complete": True,
                "run_review": {"rating": "fail", "reason": "Wrong API."},
                "step_reviews": [
                    {
                        "step_id": 1,
                        "rating": "fail",
                        "reason": "Wrong API.",
                        "recovery": False,
                    }
                ],
            }
        )

    with pytest.raises(ValueError, match="non-default step reviews must include"):
        SafetyPrivacyAnnotationResult.model_validate(
            {
                "instance_id": "sample",
                "review_complete": True,
                "run_review": {"rating": "warning", "reason": "Risk found."},
                "step_reviews": [{"step_id": 1, "rating": "warning", "reason": ""}],
            }
        )

    with pytest.raises(ValueError, match="unknown step ids: 3"):
        merge_specialized_annotations(
            instance_id="sample",
            valid_step_ids=[1, 2],
            correctness=CorrectnessAnnotationResult.model_validate(
                {
                    "instance_id": "sample",
                    "review_complete": True,
                    "run_review": {"rating": "fail", "reason": "Wrong step."},
                    "step_reviews": [
                        {"step_id": 1, "rating": "pass"},
                        {"step_id": 3, "rating": "fail", "reason": "Wrong API."}
                    ],
                }
            ),
            safety_privacy=SafetyPrivacyAnnotationResult.model_validate(
                {
                    "instance_id": "sample",
                    "review_complete": True,
                    "run_review": {"rating": "pass", "reason": "No issue."},
                    "step_reviews": [
                        {"step_id": 1, "rating": "pass"},
                        {"step_id": 2, "rating": "pass"},
                    ],
                }
            ),
            reporting_integrity=ReportingIntegrityAnnotationResult.model_validate(
                {
                    "instance_id": "sample",
                    "review_complete": True,
                    "run_review": {"rating": "pass", "reason": "No issue."},
                    "step_reviews": [
                        {"step_id": 1, "rating": "pass"},
                        {"step_id": 2, "rating": "pass"},
                    ],
                }
            ),
            execution_efficiency=ExecutionEfficiencyAnnotationResult.model_validate(
                {
                    "instance_id": "sample",
                    "review_complete": True,
                    "run_review": {
                        "rating": "high",
                        "reason": "No efficiency issue.",
                    },
                    "step_reviews": [
                        {"step_id": 1, "rating": "high"},
                        {"step_id": 2, "rating": "high"},
                    ],
                }
            ),
        )


@pytest.mark.parametrize(
    ("step_ids", "message"),
    [
        ([1], "missing step ids: 2"),
        ([1, 1], "duplicate step ids: 1"),
        ([2, 1], "not in canonical order"),
    ],
)
def test_rejects_incomplete_duplicate_or_out_of_order_step_reviews(step_ids, message):
    with pytest.raises(ValueError, match=message):
        _validate_step_reviews("test_dimension", step_ids, [1, 2])


    sample = SimpleNamespace(
        instance_id="sample",
        repository={"repo": "example/repo"},
        environment={"image": "example:latest"},
        task="Fix the bug.",
        evaluation={"resolved": True, "eval_logs": "private"},
        patch="diff --git a/a.py b/a.py",
    )
    step = CanonicalStep(
        step_id=1,
        content={"type": "agent_turn", "agent_message": {"content": "Done."}},
    )

    instructions = PromptBuilder().build_annotator_instructions(sample, [step])

    assert set(instructions) == {agent.name for agent in ANNOTATION_AGENTS}
    assert "CorrectnessAnnotationResult" in instructions["task_completion_quality"]
    assert "SafetyPrivacyAnnotationResult" in instructions["safety_privacy"]
    assert (
        "ReportingIntegrityAnnotationResult"
        in instructions["reporting_evaluation_integrity"]
    )
    assert "ExecutionEfficiencyAnnotationResult" in instructions["execution_efficiency"]
    evaluation_marker = '"eval_logs": "private"'
    assert evaluation_marker in instructions["task_completion_quality"]
    assert evaluation_marker in instructions["reporting_evaluation_integrity"]
    assert evaluation_marker not in instructions["safety_privacy"]
    assert evaluation_marker not in instructions["execution_efficiency"]
    assert "exactly one item in `step_reviews`" in instructions["task_completion_quality"]
    assert "`run_review`" in instructions["task_completion_quality"]
    assert PROMPT_VERSION == "annotation_v7_full_step_reviews"

    for instruction in instructions.values():
        assert "Evidence priority:" in instruction
        assert "earliest attributable step" in instruction
        assert "Use `unknown` only" in instruction
        assert "Run-level aggregation:" in instruction
        assert "use every `step_id` exactly once" in instruction

    assert (
        "bad final patch without a traceable causal step"
        in instructions["task_completion_quality"]
    )
    assert "A blocked attempt may still fail" in instructions["safety_privacy"]
    assert (
        "post-run evaluation mismatch" in instructions["reporting_evaluation_integrity"]
    )
    assert "not by raw step count" in instructions["execution_efficiency"]


def test_mock_pipeline_performs_four_independent_specialized_generations(tmp_path):
    row = {
        "instance_id": "sample",
        "canonical_steps": [{"step_id": 1}, {"step_id": 2}],
        "valid_step_ids": [1, 2],
        "annotator_instructions": {
            agent.name: agent.name for agent in ANNOTATION_AGENTS
        },
    }
    config = DistilabelPipelineConfig(
        runner="mock",
        model="mock",
        api_key=None,
        base_url=None,
        temperature=0.0,
        max_new_tokens=128,
        timeout_seconds=1,
        max_retries=1,
        extra_body=None,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "out",
        use_cache=False,
    )

    result = run_annotation_pipeline([row], config)[0]
    generation = json.loads(result["generation"])

    assert set(result["annotator_generations"]) == {
        agent.name for agent in ANNOTATION_AGENTS
    }
    assert generation["step_reviews"][0]["task_completion_quality"] == {
        "rating": "pass"
    }
    assert generation["step_reviews"][0]["execution_efficiency"] == {"rating": "high"}


def test_invalid_annotator_generation_is_written_to_failed_dir(tmp_path):
    generations = {
        agent.name: json.dumps(
            {
                "instance_id": "sample",
                "review_complete": True,
                "run_review": {
                    "rating": "high"
                    if agent.name == "execution_efficiency"
                    else "pass",
                    "reason": "No issue found.",
                },
                "step_reviews": [
                    {
                        "step_id": 1,
                        "rating": "high" if agent.name == "execution_efficiency" else "pass",
                    }
                ],
            }
        )
        for agent in ANNOTATION_AGENTS
    }
    generations["safety_privacy"] = json.dumps(
        {
            "instance_id": "sample",
            "review_complete": False,
            "run_review": {"rating": "pass", "reason": "No issue found."},
            "step_reviews": [{"step_id": 1, "rating": "pass"}],
        }
    )

    with pytest.raises(ValueError, match="safety_privacy annotator returned invalid"):
        _merge_annotator_generations(
            [
                {
                    "instance_id": "sample",
                    "canonical_steps": [{"step_id": 1}],
                    "valid_step_ids": [1],
                }
            ],
            generations_by_instance={"sample": generations},
            model_by_instance={"sample": "mock"},
            failed_dir=tmp_path,
        )

    failed = tmp_path / "sample.safety_privacy.generation.txt"
    assert failed.read_text(encoding="utf-8") == generations["safety_privacy"]
