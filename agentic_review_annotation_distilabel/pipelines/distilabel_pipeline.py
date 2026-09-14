import json
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from agentic_review_annotation_distilabel.annotation.annotators import (
    ANNOTATION_AGENTS,
    AnnotationAgentSpec,
)
from agentic_review_annotation_distilabel.annotation.merge import (
    merge_specialized_annotations,
)
from agentic_review_annotation_distilabel.annotation.schema import (
    annotation_to_dict,
    parse_specialized_annotation,
)


class DistilabelPipelineConfig:
    def __init__(
        self,
        *,
        runner: str,
        model: str,
        api_key: str | None,
        base_url: str | None,
        temperature: float,
        max_new_tokens: int,
        timeout_seconds: int,
        max_retries: int,
        extra_body: dict[str, Any] | None,
        cache_dir: Path,
        output_dir: Path,
        use_cache: bool,
    ) -> None:
        self.runner = runner
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.extra_body = extra_body
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.use_cache = use_cache


def run_annotation_pipeline(
    rows: list[dict[str, Any]],
    config: DistilabelPipelineConfig,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    if config.runner == "mock":
        return _run_mock_annotation(rows, config)

    generations_by_instance: dict[str, dict[str, str]] = {
        str(row["instance_id"]): {} for row in rows
    }
    model_by_instance: dict[str, str] = {}

    for agent in ANNOTATION_AGENTS:
        annotated_rows = _run_single_annotator_pipeline(rows, config, agent)
        for annotated_row in annotated_rows:
            instance_id = str(annotated_row["instance_id"])
            generation = annotated_row.get("generation")
            if not generation:
                raise ValueError(
                    f"{agent.name} annotator returned empty generation for {instance_id}"
                )
            generations_by_instance[instance_id][agent.name] = str(generation)
            model_by_instance[instance_id] = str(
                annotated_row.get("model_name") or config.model
            )

    return _merge_annotator_generations(
        rows,
        generations_by_instance=generations_by_instance,
        model_by_instance=model_by_instance,
        failed_dir=config.output_dir / "_failed",
    )


def _run_single_annotator_pipeline(
    rows: list[dict[str, Any]],
    config: DistilabelPipelineConfig,
    agent: AnnotationAgentSpec,
) -> list[dict[str, Any]]:
    task_rows = []
    for row in rows:
        instructions = row.get("annotator_instructions")
        if not isinstance(instructions, dict) or agent.name not in instructions:
            raise ValueError(f"missing {agent.name} annotator instruction")
        task_rows.append(
            {
                **row,
                "instruction": instructions[agent.name],
                "structured_output": {
                    "format": "json",
                    "schema": agent.json_schema(),
                    "max_retries": config.max_retries,
                },
            }
        )

    from distilabel.pipeline import Pipeline
    from distilabel.steps import LoadDataFromDicts

    config.cache_dir.mkdir(parents=True, exist_ok=True)

    with Pipeline(
        name=f"agentic-work-review-{agent.name}",
        description=f"Auto-annotate {agent.name} for SWE agent trajectories.",
        cache_dir=str(config.cache_dir),
    ) as pipeline:
        load_data = LoadDataFromDicts(name="load_data", data=task_rows, batch_size=1)

        from distilabel.models.llms import OpenAILLM
        from distilabel.steps.tasks import TextGeneration

        api_key = SecretStr(config.api_key) if config.api_key else None
        annotate = TextGeneration(
            name=f"{agent.name}_annotation",
            llm=OpenAILLM(
                model=config.model,
                base_url=config.base_url,
                api_key=api_key,
                timeout=config.timeout_seconds,
                max_retries=config.max_retries,
                generation_kwargs={
                    "temperature": config.temperature,
                    "max_new_tokens": config.max_new_tokens,
                    "response_format": {"type": "json_object"},
                    "extra_body": config.extra_body,
                },
            ),
            system_prompt=agent.system_prompt,
            input_batch_size=1,
            use_cache=True,
        )

        load_data >> annotate

    distiset = pipeline.run(use_cache=config.use_cache)
    return list(distiset["default"]["train"])


def _run_mock_annotation(
    rows: list[dict[str, Any]],
    config: DistilabelPipelineConfig,
) -> list[dict[str, Any]]:
    generations_by_instance: dict[str, dict[str, str]] = {}
    for row in rows:
        instance_id = str(row["instance_id"])
        generations_by_instance[instance_id] = {
            agent.name: json.dumps(
                _mock_specialized_result(row, agent),
                ensure_ascii=False,
            )
            for agent in ANNOTATION_AGENTS
        }
    return _merge_annotator_generations(
        rows,
        generations_by_instance=generations_by_instance,
        model_by_instance={
            str(row["instance_id"]): config.model
            for row in rows
        },
        failed_dir=None,
    )


def _mock_specialized_result(
    row: dict[str, Any],
    agent: AnnotationAgentSpec,
) -> dict[str, Any]:
    canonical_steps = row.get("canonical_steps")
    if not isinstance(canonical_steps, list):
        canonical_steps = []
    return {
        "instance_id": row["instance_id"],
        "step_reviews": [
            {"step_id": step.get("step_id"), "label": "pass"}
            for step in canonical_steps
            if isinstance(step, dict)
        ],
    }


def _merge_annotator_generations(
    rows: list[dict[str, Any]],
    *,
    generations_by_instance: dict[str, dict[str, str]],
    model_by_instance: dict[str, str],
    failed_dir: Path | None,
) -> list[dict[str, Any]]:
    merged_rows: list[dict[str, Any]] = []
    agents_by_name = {agent.name: agent for agent in ANNOTATION_AGENTS}
    for row in rows:
        instance_id = str(row["instance_id"])
        generations = generations_by_instance.get(instance_id, {})
        missing = [
            agent.name
            for agent in ANNOTATION_AGENTS
            if agent.name not in generations
        ]
        if missing:
            raise ValueError(
                f"{instance_id} is missing annotator generations: {', '.join(missing)}"
            )

        correctness = _parse_annotator_generation(
            instance_id=instance_id,
            agent=agents_by_name["task_completion_quality"],
            generation=generations["task_completion_quality"],
            failed_dir=failed_dir,
        )
        safety_privacy = _parse_annotator_generation(
            instance_id=instance_id,
            agent=agents_by_name["safety_privacy"],
            generation=generations["safety_privacy"],
            failed_dir=failed_dir,
        )
        reporting_integrity = _parse_annotator_generation(
            instance_id=instance_id,
            agent=agents_by_name["reporting_evaluation_integrity"],
            generation=generations["reporting_evaluation_integrity"],
            failed_dir=failed_dir,
        )
        execution_efficiency = _parse_annotator_generation(
            instance_id=instance_id,
            agent=agents_by_name["execution_efficiency"],
            generation=generations["execution_efficiency"],
            failed_dir=failed_dir,
        )
        annotation = merge_specialized_annotations(
            instance_id=instance_id,
            valid_step_ids=list(row["valid_step_ids"]),
            correctness=correctness,
            safety_privacy=safety_privacy,
            reporting_integrity=reporting_integrity,
            execution_efficiency=execution_efficiency,
        )
        merged_rows.append(
            {
                **row,
                "generation": json.dumps(
                    annotation_to_dict(annotation),
                    ensure_ascii=False,
                ),
                "model_name": model_by_instance.get(instance_id),
                "annotator_generations": generations,
            }
        )
    return merged_rows


def _parse_annotator_generation(
    *,
    instance_id: str,
    agent: AnnotationAgentSpec,
    generation: str,
    failed_dir: Path | None,
) -> Any:
    try:
        return parse_specialized_annotation(
            generation,
            agent.result_type,
            instance_id=instance_id,
        )
    except Exception as exc:
        if failed_dir is not None:
            failed_dir.mkdir(parents=True, exist_ok=True)
            failed_path = failed_dir / f"{instance_id}.{agent.name}.generation.txt"
            failed_path.write_text(generation, encoding="utf-8")
        raise ValueError(
            f"{agent.name} annotator returned invalid structured output "
            f"for {instance_id}: {exc}"
        ) from exc
