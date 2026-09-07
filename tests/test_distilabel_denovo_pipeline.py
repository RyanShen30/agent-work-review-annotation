from __future__ import annotations

import json
import unittest
from pathlib import Path

from agentic_review_annotation_distilabel.adapters.denovo import DeNovoSWEAdapter
from agentic_review_annotation_distilabel.annotation.prompt_builder import PromptBuilder
from agentic_review_annotation_distilabel.annotation.schema import (
    parse_annotation,
    validate_annotation_against_steps,
)
from agentic_review_annotation_distilabel.steps.denovo import DeNovoSWEStepParser


class DistilabelDeNovoPipelineTests(unittest.TestCase):
    def test_denovo_adapter_step_parser_and_schema(self) -> None:
        raw = json.loads(Path("annotation/samples/sample_001.json").read_text(encoding="utf-8"))
        sample = DeNovoSWEAdapter().adapt(raw)
        steps = DeNovoSWEStepParser().parse(sample)
        prompt = PromptBuilder().build_instruction(sample, steps)
        annotation = parse_annotation(
            {
                "instance_id": sample.instance_id,
                "final_outcome": "correct",
                "failures": [],
            }
        )

        self.assertEqual(sample.instance_id, "mewwts_addict_pr130")
        self.assertEqual(len(steps), 43)
        self.assertEqual(steps[0].step_id, 0)
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
