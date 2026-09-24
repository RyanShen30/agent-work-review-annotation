from __future__ import annotations

from typing import Any, Literal

from agentic_review_annotation_distilabel.annotation.schema import MasterRecord

PublicExportMode = Literal["benchmark_task", "annotation_release"]
SENSITIVE_ENV_KEY_PARTS = ("key", "token", "secret", "password", "authorization")


def export_master(master: MasterRecord | dict[str, Any]) -> dict[str, Any]:
    record = _as_master_record(master)
    payload = record.model_dump(exclude_none=True)
    payload.setdefault("annotation", {})
    payload["annotation"].setdefault("final", None)
    return payload


def export_public(
    master: MasterRecord | dict[str, Any],
    *,
    mode: PublicExportMode = "benchmark_task",
) -> dict[str, Any]:
    record = _as_master_record(master)
    payload: dict[str, Any] = {
        "schema_version": f"agent_work_review.public.{mode}.v1",
        "mode": mode,
        "instance_id": record.instance_id,
        "problem_statement": record.source.problem_statement,
        "repo": record.source.repo,
        "base_commit": record.source.base_commit,
        "environment": _public_environment(record.run.environment),
        "run": {
            key: value
            for key, value in {
                "run_id": record.run.run_id,
                "harness": record.run.harness,
                "harness_version": record.run.harness_version,
                "model": record.run.model,
            }.items()
            if value is not None
        },
        "canonical_steps": record.trajectory.canonical_steps,
        "generated_patch": record.run.generated_patch,
    }
    if mode == "annotation_release" and record.annotation.final:
        payload["final_annotation"] = record.annotation.final.model_dump(exclude_none=True)
    return _drop_none(payload)


def export_private(master: MasterRecord | dict[str, Any]) -> dict[str, Any]:
    record = _as_master_record(master)
    payload = _drop_none(
        {
            "schema_version": "agent_work_review.private.v1",
            "instance_id": record.instance_id,
            "source": record.source.model_dump(exclude_none=True),
            "run": record.run.model_dump(exclude_none=True),
            "trajectory": {
                "raw_path": record.trajectory.raw_path,
                "raw_sha256": record.trajectory.raw_sha256,
            },
            "deterministic_facts": record.deterministic_facts,
            "evaluation": record.evaluation.model_dump(exclude_none=True),
            "oracle": record.oracle.model_dump(exclude_none=True),
            "provenance": record.provenance.model_dump(exclude_none=True),
        }
    )
    payload["annotation"] = {
        "final": record.annotation.final.model_dump(exclude_none=True)
        if record.annotation.final
        else None
    }
    return payload


def _as_master_record(master: MasterRecord | dict[str, Any]) -> MasterRecord:
    if isinstance(master, MasterRecord):
        return master
    return MasterRecord.model_validate(master)


def _public_environment(environment: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "image",
        "cwd",
        "env",
        "runtime",
        "environment_class",
        "timeout",
        "container_timeout",
        "platform",
    }
    public: dict[str, Any] = {}
    for key, value in environment.items():
        if key not in keep:
            continue
        if key == "env" and isinstance(value, dict):
            public[key] = {
                env_key: env_value
                for env_key, env_value in value.items()
                if not _is_sensitive_env_key(str(env_key))
            }
            continue
        public[key] = value
    return public


def _is_sensitive_env_key(key: str) -> bool:
    normalized = key.lower()
    return any(part in normalized for part in SENSITIVE_ENV_KEY_PARTS)


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := _drop_none(item)) is not None
        }
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value
