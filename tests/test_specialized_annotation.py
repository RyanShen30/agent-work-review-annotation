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


def test_merges_four_specialized_results_into_legacy_step_review():
    merged = merge_specialized_annotations(
        instance_id="sample",
        valid_step_ids=[1, 2],
        correctness=CorrectnessAnnotationResult.model_validate(
            {
                "instance_id": "sample",
                "step_reviews": [
                    {"step_id": 1, "label": "pass"},
                    {
                        "step_id": 2,
                        "label": "error",
                        "reason": "The step changed the wrong API.",
                        "recovery": True,
                    },
                ],
            }
        ),
        safety_privacy=SafetyPrivacyAnnotationResult.model_validate(
            {
                "instance_id": "sample",
                "step_reviews": [
                    {"step_id": 1, "label": "pass"},
                    {"step_id": 2, "label": "pass"},
                ],
            }
        ),
        reporting_integrity=ReportingIntegrityAnnotationResult.model_validate(
            {
                "instance_id": "sample",
                "step_reviews": [
                    {"step_id": 1, "label": "pass"},
                    {
                        "step_id": 2,
                        "label": "issue",
                        "reason": "The step claimed tests passed without running them.",
                    },
                ],
            }
        ),
        execution_efficiency=ExecutionEfficiencyAnnotationResult.model_validate(
            {
                "instance_id": "sample",
                "step_reviews": [
                    {"step_id": 1, "label": "pass"},
                    {
                        "step_id": 2,
                        "label": "issue",
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
        "execution_efficiency": {"rating": "normal"},
    }
    assert dumped["step_reviews"][1]["task_completion_quality"] == {
        "rating": "fail",
        "reason": "The step changed the wrong API.",
        "recovery": True,
    }
    assert dumped["step_reviews"][1]["execution_efficiency"]["rating"] == "low"


def test_rejects_invalid_specialized_outputs_before_merge():
    with pytest.raises(ValueError, match="pass correctness annotations must omit reason"):
        CorrectnessAnnotationResult.model_validate(
            {
                "instance_id": "sample",
                "step_reviews": [
                    {"step_id": 1, "label": "pass", "reason": "Looks fine."}
                ],
            }
        )

    with pytest.raises(ValueError, match="recovery must be true when present"):
        CorrectnessAnnotationResult.model_validate(
            {
                "instance_id": "sample",
                "step_reviews": [
                    {"step_id": 1, "label": "error", "reason": "Wrong API.", "recovery": False}
                ],
            }
        )

    with pytest.raises(ValueError, match="pass annotations must omit reason"):
        SafetyPrivacyAnnotationResult.model_validate(
            {
                "instance_id": "sample",
                "step_reviews": [
                    {"step_id": 1, "label": "pass", "reason": None}
                ],
            }
        )

    with pytest.raises(ValueError, match="missing step ids: 2"):
        merge_specialized_annotations(
            instance_id="sample",
            valid_step_ids=[1, 2],
            correctness=CorrectnessAnnotationResult.model_validate(
                {
                    "instance_id": "sample",
                    "step_reviews": [{"step_id": 1, "label": "pass"}],
                }
            ),
            safety_privacy=SafetyPrivacyAnnotationResult.model_validate(
                {
                    "instance_id": "sample",
                    "step_reviews": [
                        {"step_id": 1, "label": "pass"},
                        {"step_id": 2, "label": "pass"},
                    ],
                }
            ),
            reporting_integrity=ReportingIntegrityAnnotationResult.model_validate(
                {
                    "instance_id": "sample",
                    "step_reviews": [
                        {"step_id": 1, "label": "pass"},
                        {"step_id": 2, "label": "pass"},
                    ],
                }
            ),
            execution_efficiency=ExecutionEfficiencyAnnotationResult.model_validate(
                {
                    "instance_id": "sample",
                    "step_reviews": [
                        {"step_id": 1, "label": "pass"},
                        {"step_id": 2, "label": "pass"},
                    ],
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
    assert "ReportingIntegrityAnnotationResult" in instructions[
        "reporting_evaluation_integrity"
    ]
    assert "ExecutionEfficiencyAnnotationResult" in instructions["execution_efficiency"]
    assert "private" in instructions["task_completion_quality"]
    assert "private" not in instructions["safety_privacy"]


def test_mock_pipeline_performs_four_independent_specialized_generations(tmp_path):
    row = {
        "instance_id": "sample",
        "canonical_steps": [{"step_id": 1}, {"step_id": 2}],
        "valid_step_ids": [1, 2],
        "annotator_instructions": {agent.name: agent.name for agent in ANNOTATION_AGENTS},
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
    assert generation["step_reviews"][0]["execution_efficiency"] == {
        "rating": "normal"
    }


def test_invalid_annotator_generation_is_written_to_failed_dir(tmp_path):
    generations = {
        agent.name: json.dumps(
            {
                "instance_id": "sample",
                "step_reviews": [{"step_id": 1, "label": "pass"}],
            }
        )
        for agent in ANNOTATION_AGENTS
    }
    generations["safety_privacy"] = json.dumps(
        {
            "instance_id": "sample",
            "step_reviews": [{"step_id": 1, "label": "pass", "reason": None}],
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
