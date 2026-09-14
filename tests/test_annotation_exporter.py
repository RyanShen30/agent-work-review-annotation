from agentic_review_annotation_distilabel.annotation.exporter import (
    export_private,
    export_public,
)
from agentic_review_annotation_distilabel.annotation.schema import MasterRecord


def test_public_benchmark_task_hides_annotation_and_oracle():
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
            "evaluation": {"resolved": True, "eval_logs": "hidden logs"},
            "oracle": {
                "gold_patch": "gold patch",
                "test_patch": "test patch",
                "fail_to_pass": ["test_fail"],
                "pass_to_pass": ["test_pass"],
            },
            "annotation": {
                "auto": {
                    "model": "annotator",
                    "prompt_version": "annotation_v1",
                    "step_reviews": [
                        {
                            "step": 1,
                            "task_completion_quality": {"rating": "pass"},
                            "safety_privacy": {"rating": "pass"},
                            "reporting_evaluation_integrity": {"rating": "pass"},
                            "execution_efficiency": {"rating": "high"},
                        }
                    ],
                },
                "final": {
                    "instance_id": "django__django-11049",
                    "step_reviews": [
                        {
                            "step": 1,
                            "task_completion_quality": {"rating": "pass"},
                            "safety_privacy": {"rating": "pass"},
                            "reporting_evaluation_integrity": {"rating": "pass"},
                            "execution_efficiency": {"rating": "high"},
                        }
                    ],
                },
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
    assert "gold_patch" not in str(public)
    assert "test_fail" not in str(public)
    assert private["oracle"]["gold_patch"] == "gold patch"
    assert private["annotation"]["final"]["step_reviews"][0]["step"] == 1


def test_public_annotation_release_can_include_final_reviews():
    master = {
        "instance_id": "sample",
        "annotation": {
            "final": {
                "instance_id": "sample",
                "step_reviews": [
                    {
                        "step": 1,
                        "task_completion_quality": {"rating": "pass"},
                        "safety_privacy": {"rating": "pass"},
                        "reporting_evaluation_integrity": {"rating": "pass"},
                        "execution_efficiency": {"rating": "high"},
                    }
                ],
            }
        },
    }

    public = export_public(master, mode="annotation_release")

    assert public["mode"] == "annotation_release"
    assert public["final_annotation"]["step_reviews"][0]["step"] == 1
