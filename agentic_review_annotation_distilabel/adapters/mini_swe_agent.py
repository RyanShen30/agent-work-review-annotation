from __future__ import annotations

from typing import Any, Mapping

from agentic_review_annotation_distilabel.adapters.base import (
    DatasetAdapter,
    Sample,
    require_mapping,
)


class MiniSWEAgentAdapter(DatasetAdapter):
    dataset_name = "mini_swe_agent"

    def adapt(self, raw: Mapping[str, Any]) -> Sample:
        raw = require_mapping(raw)
        trajectory = raw.get("messages")
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError("mini-swe-agent sample is missing required field: messages")

        instance_id = raw.get("instance_id")
        if not instance_id:
            raise ValueError("mini-swe-agent sample is missing required field: instance_id")

        info = raw.get("info") if isinstance(raw.get("info"), Mapping) else {}
        outcome = raw.get("outcome") if isinstance(raw.get("outcome"), Mapping) else {}
        return Sample(
            instance_id=str(instance_id),
            task=raw.get("problem"),
            trajectory=trajectory,
            patch=raw.get("patch"),
            evaluation={
                "finish_reason": outcome.get("exit_status") or info.get("exit_status"),
                "model_stats": info.get("model_stats"),
            },
            raw=raw,
        )
