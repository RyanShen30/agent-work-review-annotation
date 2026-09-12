from __future__ import annotations

from typing import Any, Mapping

from agentic_review_annotation_distilabel.adapters.base import (
    DatasetAdapter,
    Sample,
    require_mapping,
)


class OpenHandsAdapter(DatasetAdapter):
    dataset_name = "openhands"

    def adapt(self, raw: Mapping[str, Any]) -> Sample:
        raw = require_mapping(raw)
        trajectory = raw.get("events")
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError("OpenHands sample is missing required field: events")

        instance_id = raw.get("instance_id")
        if not instance_id:
            raise ValueError("OpenHands sample is missing required field: instance_id")

        return Sample(
            instance_id=str(instance_id),
            task=raw.get("problem"),
            trajectory=trajectory,
            patch=raw.get("patch"),
            evaluation={"finish_reason": raw.get("status"), "metrics": raw.get("metrics")},
            raw=raw,
        )
