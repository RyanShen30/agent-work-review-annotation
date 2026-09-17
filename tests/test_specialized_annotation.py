import json

import pytest

from agentic_review_annotation_distilabel.annotation.annotators import ANNOTATION_AGENTS
from agentic_review_annotation_distilabel.annotation.merge import (
    merge_specialized_annotations,
)
from agentic_review_annotation_distilabel.annotation.prompt_builder import PromptBuilder
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


def test_merges_sparse_specialized_results_into_dense_step_and_run_reviews():
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
                "findings": [
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
                "findings": [],
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
                "findings": [
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
                "findings": [
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
    with pytest.raises(
        ValueError, match="Input should be 'warning', 'fail' or 'unknown'"
    ):
        CorrectnessAnnotationResult.model_validate(
            {
                "instance_id": "sample",
                "review_complete": True,
                "run_review": {"rating": "pass", "reason": "No issue."},
                "findings": [{"step_id": 1, "rating": "pass", "reason": "Looks fine."}],
            }
        )

    with pytest.raises(ValueError, match="Input should be True"):
        CorrectnessAnnotationResult.model_validate(
            {
                "instance_id": "sample",
                "review_complete": True,
                "run_review": {"rating": "fail", "reason": "Wrong API."},
                "findings": [
                    {
                        "step_id": 1,
                        "rating": "fail",
                        "reason": "Wrong API.",
                        "recovery": False,
                    }
                ],
            }
        )

    with pytest.raises(ValueError, match="findings must include a non-empty reason"):
        SafetyPrivacyAnnotationResult.model_validate(
            {
                "instance_id": "sample",
                "review_complete": True,
                "run_review": {"rating": "warning", "reason": "Risk found."},
                "findings": [{"step_id": 1, "rating": "warning", "reason": ""}],
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
                    "findings": [
                        {"step_id": 3, "rating": "fail", "reason": "Wrong API."}
                    ],
                }
            ),
            safety_privacy=SafetyPrivacyAnnotationResult.model_validate(
                {
                    "instance_id": "sample",
                    "review_complete": True,
                    "run_review": {"rating": "pass", "reason": "No issue."},
                    "findings": [],
                }
            ),
            reporting_integrity=ReportingIntegrityAnnotationResult.model_validate(
                {
                    "instance_id": "sample",
                    "review_complete": True,
                    "run_review": {"rating": "pass", "reason": "No issue."},
                    "findings": [],
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
                    "findings": [],
                }
            ),
        )


def test_prompt_builder_creates_four_dedicated_instructions():
    class Sample:
        instance_id = "sample"
        repository = {"repo": "example/repo"}
        environment = {"image": "example:latest"}
        task = "Fix the bug."
        evaluation = {"resolved": True, "eval_logs": "private"}
        patch = "diff --git a/a.py b/a.py"

    class Step:
        step_id = 1
        content = {"type": "agent_turn", "agent_message": {"content": "Done."}}

        def to_dict(self):
            return {"step_id": self.step_id, "content": self.content}

    instructions = PromptBuilder().build_annotator_instructions(Sample(), [Step()])

    assert set(instructions) == {agent.name for agent in ANNOTATION_AGENTS}
    assert "CorrectnessAnnotationResult" in instructions["task_completion_quality"]
    assert "SafetyPrivacyAnnotationResult" in instructions["safety_privacy"]
    assert (
        "ReportingIntegrityAnnotationResult"
        in instructions["reporting_evaluation_integrity"]
    )
    assert "ExecutionEfficiencyAnnotationResult" in instructions["execution_efficiency"]
    assert "private" in instructions["task_completion_quality"]
    assert "private" in instructions["reporting_evaluation_integrity"]
    assert "private" not in instructions["safety_privacy"]
    assert "private" not in instructions["execution_efficiency"]
    assert "sparse `findings`" in instructions["task_completion_quality"]
    assert "`run_review`" in instructions["task_completion_quality"]


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
                "findings": [],
            }
        )
        for agent in ANNOTATION_AGENTS
    }
    generations["safety_privacy"] = json.dumps(
        {
            "instance_id": "sample",
            "review_complete": False,
            "run_review": {"rating": "pass", "reason": "No issue found."},
            "findings": [],
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
