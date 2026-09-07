from __future__ import annotations

from typing import Any, Mapping

from agentic_review_annotation_distilabel.adapters.base import (
    DatasetAdapter,
    Sample,
    require_mapping,
)


class DeNovoSWEAdapter(DatasetAdapter):
    dataset_name = "denovo"
    evaluation_fields = (
        "success",
        "score",
        "finish_reason",
        "error",
        "difficulty",
        "eval_result",
    )

    def adapt(self, raw: Mapping[str, Any]) -> Sample:
        raw = require_mapping(raw)
        instance_id = raw.get("instance_id")
        trajectory = raw.get("trajectory")
        if not instance_id:
            raise ValueError("DeNovoSWE sample is missing required field: instance_id")
        if trajectory is None:
            raise ValueError("DeNovoSWE sample is missing required field: trajectory")

        return Sample(
            instance_id=str(instance_id),
            task=raw.get("initial_messages"),
            trajectory=trajectory,
            patch=raw.get("patch"),
            evaluation={
                field: raw.get(field)
                for field in self.evaluation_fields
                if field in raw
            },
            raw=raw,
        )
