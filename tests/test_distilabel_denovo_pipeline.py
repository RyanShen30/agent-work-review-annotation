from __future__ import annotations

import unittest

from agentic_review_annotation_distilabel.adapters.denovo import DeNovoSWEAdapter
from agentic_review_annotation_distilabel.annotation.prompt_builder import PromptBuilder
from agentic_review_annotation_distilabel.annotation.schema import (
    parse_annotation,
    validate_annotation_against_steps,
)
from agentic_review_annotation_distilabel.steps.denovo import DeNovoSWEStepParser


class DistilabelDeNovoPipelineTests(unittest.TestCase):
    def test_denovo_adapter_step_parser_and_schema(self) -> None:
        raw = {
            "instance_id": "inline_denovo_sample",
            "success": True,
            "initial_messages": [{"role": "user", "content": "fix it"}],
            "trajectory": [
                {"step": 0, "action": {"type": "thought", "content": "inspect"}},
                {"step": 1, "action": {"type": "finish", "content": "done"}},
            ],
            "patch": "diff --git a/a.py b/a.py",
        }
        sample = DeNovoSWEAdapter().adapt(raw)
        steps = DeNovoSWEStepParser().parse(sample)
        prompt = PromptBuilder().build_instruction(sample, steps)
        annotation = parse_annotation(
            {
                "instance_id": sample.instance_id,
                "step_reviews": [
                    {
                        "step": step.step_id,
                        "task_completion_quality": {
                            "rating": "unknown",
                            "reason": "The inline fixture does not include enough evidence.",
                        },
                        "safety_privacy": {
                            "rating": "unknown",
                            "reason": "The inline fixture does not include safety evidence.",
                        },
                        "reporting_evaluation_integrity": {
                            "rating": "unknown",
                            "reason": "The inline fixture does not include reporting evidence.",
                        },
                        "execution_efficiency": {
                            "rating": "unknown",
                            "reason": "The inline fixture does not include efficiency evidence.",
                        },
                    }
                    for step in steps
                ],
            }
        )

        self.assertEqual(sample.instance_id, "inline_denovo_sample")
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0].step_id, 1)
        self.assertIn("action", steps[0].content)
        self.assertEqual(sample.task, raw["initial_messages"])
        self.assertIn("Review payload", prompt)
        validate_annotation_against_steps(
            annotation=annotation,
            instance_id=sample.instance_id,
            valid_step_ids=[step.step_id for step in steps],
        )


if __name__ == "__main__":
    unittest.main()
