import json
from pathlib import Path
from typing import Any

from distilabel.steps import Step, StepInput
from distilabel.typing import StepColumns, StepOutput
from pydantic import SecretStr


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

    from distilabel.pipeline import Pipeline
    from distilabel.steps import LoadDataFromDicts

    config.cache_dir.mkdir(parents=True, exist_ok=True)

    with Pipeline(
        name="agentic-work-review-annotation",
        description="Auto-annotate SWE agent trajectories for human review.",
        cache_dir=str(config.cache_dir),
    ) as pipeline:
        load_data = LoadDataFromDicts(name="load_data", data=rows, batch_size=1)

        if config.runner == "mock":
            annotate = MockAnnotationStep(
                name="mock_annotation",
                model_name=config.model,
                input_batch_size=1,
                use_cache=True,
            )
        else:
            from distilabel.models.llms import OpenAILLM
            from distilabel.steps.tasks import TextGeneration

            api_key = SecretStr(config.api_key) if config.api_key else None
            annotate = TextGeneration(
                name="json_annotation",
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
                system_prompt=(
                    "You are a careful software engineering trajectory reviewer. "
                    "Return only valid JSON matching the requested schema."
                ),
                input_batch_size=1,
                use_cache=True,
            )

        load_data >> annotate

    distiset = pipeline.run(use_cache=config.use_cache)
    return list(distiset["default"]["train"])


class MockAnnotationStep(Step):
    model_name: str = "mock"

    @property
    def inputs(self) -> StepColumns:
        return ["instance_id", "evaluation"]

    @property
    def outputs(self) -> StepColumns:
        return ["generation", "model_name"]

    def process(self, inputs: StepInput) -> StepOutput:
        outputs: list[dict[str, Any]] = []
        for row in inputs:
            evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), dict) else {}
            accepted = evaluation.get("success")
            eval_result = evaluation.get("eval_result")
            if isinstance(eval_result, dict) and "accepted" in eval_result:
                accepted = eval_result["accepted"]
            row["generation"] = json.dumps(
                {
                    "instance_id": row["instance_id"],
                    "final_outcome": "correct" if accepted is True else "incorrect",
                    "failures": [],
                },
                ensure_ascii=False,
            )
            row["model_name"] = self.model_name
            outputs.append(row)
        yield outputs
