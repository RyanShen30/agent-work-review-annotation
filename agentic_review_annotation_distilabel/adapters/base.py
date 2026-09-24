from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Sample:
    instance_id: str
    task: Any | None
    trajectory: Any
    patch: Any | None
    evaluation: Any | None
    raw: Mapping[str, Any]
    repository: Any | None = None
    environment: Any | None = None
    source: dict[str, Any] | None = None
    run: dict[str, Any] | None = None
    oracle: dict[str, Any] | None = None


class DatasetAdapter:
    dataset_name: str

    def adapt(self, raw: Mapping[str, Any]) -> Sample:
        raise NotImplementedError


def require_mapping(raw: Any) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("Raw dataset sample must be a JSON object.")
    return raw
