from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from agentic_review_annotation_distilabel.adapters import (
    DeNovoSWEAdapter,
    MiniSWEAgentAdapter,
    OpenHandsAdapter,
)
from agentic_review_annotation_distilabel.annotation.prompt_builder import (
    PROMPT_VERSION,
    PromptBuilder,
)
from agentic_review_annotation_distilabel.annotation.schema import (
    annotation_to_dict,
    annotation_json_schema,
    parse_annotation,
    validate_annotation_against_steps,
)
from agentic_review_annotation_distilabel.pipelines import (
    DistilabelPipelineConfig,
    run_annotation_pipeline,
)
from agentic_review_annotation_distilabel.steps import (
    AgentStepParser,
    DeNovoSWEStepParser,
    MiniSWEAgentStepParser,
)

DEFAULT_INPUT = Path("annotation/samples")
DEFAULT_NORMALIZED_DIR = Path("output/annotation/normalized")
DEFAULT_NORMALIZED_PREVIEW_DIR = Path("output/annotation/preview")
DEFAULT_OUTPUT_DIR = Path("output/annotation/annotation")
DEFAULT_CACHE_DIR = Path("output/annotation/cache")

ADAPTERS = {
    "denovo": DeNovoSWEAdapter,
    "mini_swe_agent": MiniSWEAgentAdapter,
    "openhands": OpenHandsAdapter,
}

STEP_PARSERS = {
    "denovo": DeNovoSWEStepParser,
    "mini_swe_agent": MiniSWEAgentStepParser,
    "openhands": AgentStepParser,
}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = config.get("review", config)
    paths = config.get("paths") if isinstance(config.get("paths"), dict) else {}
    model_config = config.get("model") if isinstance(config.get("model"), dict) else {}

    input_path = args.input or Path(config.get("input", paths.get("input", DEFAULT_INPUT)))
    normalized_dir = args.normalized_dir or Path(
        paths.get("normalized_dir", DEFAULT_NORMALIZED_DIR)
    )
    normalized_preview_dir = args.normalized_preview_dir or Path(
        paths.get("normalized_preview_dir", DEFAULT_NORMALIZED_PREVIEW_DIR)
    )
    output_dir = args.output_dir or Path(paths.get("output_dir", DEFAULT_OUTPUT_DIR))
    cache_dir = args.cache_dir or Path(paths.get("cache_dir", DEFAULT_CACHE_DIR))

    runner = args.runner or config.get("runner") or "llm"
    dataset = args.dataset or config.get("dataset") or "mini_swe_agent"

    max_retries = int(model_config.get("max_retries", 2))

    max_new_tokens = args.model_max_new_tokens or int(model_config.get("max_new_tokens", 4096))

    prompt_budget = config.get("prompt_budget", {})
    prompt_builder = PromptBuilder(
        compact_for_model=args.compact_model_input or bool(prompt_budget.get("compact_for_model", False)),
        max_task_chars=args.max_task_chars or int(prompt_budget.get("max_task_chars", 12000)),
        max_patch_chars=args.max_patch_chars or int(prompt_budget.get("max_patch_chars", 20000)),
        max_step_chars=args.max_step_chars or int(prompt_budget.get("max_step_chars", 6000)),
        max_total_step_chars=args.max_total_step_chars
        or int(prompt_budget.get("max_total_step_chars", 60000)),
    )

    rows = prepare_rows(
        input_path=input_path,
        normalized_dir=normalized_dir,
        normalized_preview_dir=normalized_preview_dir,
        output_dir=output_dir,
        dataset=dataset,
        limit=args.limit,
        start_index=args.start_index,
        overwrite=args.overwrite,
        structured_max_retries=max_retries,
        prompt_builder=prompt_builder,
    )

    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    pipeline_config = DistilabelPipelineConfig(
        runner=runner,
        model=model_config.get("model") or ("mock" if runner == "mock" else "gpt-4.1"),
        api_key=api_key,
        base_url=base_url,
        temperature=float(model_config.get("temperature", 0.0)),
        max_new_tokens=max_new_tokens,
        timeout_seconds=int(model_config.get("timeout_seconds", 120)),
        max_retries=max_retries,
        extra_body=build_extra_body(
            base_url=base_url,
            model=model_config.get("model") or "",
            model_config=model_config,
            disable_thinking=args.disable_thinking,
            enable_thinking=args.enable_thinking,
        ),
        cache_dir=cache_dir,
        output_dir=output_dir,
        use_cache=not args.no_cache and bool(config.get("use_cache", True)),
    )

    if runner == "llm" and not pipeline_config.api_key:
        raise RuntimeError("Missing API key: export LLM_API_KEY or run with --runner mock.")

    saved = run_and_save_each(rows, pipeline_config, output_dir)
    print(f"done: queued={len(rows)} saved={saved} output_dir={output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Distilabel auto-annotation for agentic work review."
    )
    parser.add_argument("--input", type=Path, default=None, help="Input JSON file or directory.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--normalized-dir", type=Path, default=None)
    parser.add_argument("--normalized-preview-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", choices=sorted(ADAPTERS), default=None)
    parser.add_argument("--runner", choices=["llm", "mock"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based input offset.")
    parser.add_argument("--model-max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Pass DeepSeek/OpenAI-compatible extra_body to disable thinking mode.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Do not auto-disable thinking mode for DeepSeek models.",
    )
    parser.add_argument(
        "--compact-model-input",
        action="store_true",
        help="Use a truncated payload for cheap pipeline debugging. Formal annotation uses full input by default.",
    )
    parser.add_argument("--max-task-chars", type=int)
    parser.add_argument("--max-patch-chars", type=int)
    parser.add_argument("--max-step-chars", type=int)
    parser.add_argument("--max-total-step-chars", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def prepare_rows(
    *,
    input_path: Path,
    normalized_dir: Path,
    normalized_preview_dir: Path,
    output_dir: Path,
    dataset: str,
    limit: int | None,
    start_index: int,
    overwrite: bool,
    structured_max_retries: int,
    prompt_builder: PromptBuilder,
) -> list[dict[str, Any]]:
    adapter = ADAPTERS[dataset]()
    step_parser = STEP_PARSERS[dataset]()
    input_paths = collect_input_paths(input_path, dataset)
    input_paths = input_paths[start_index:]
    if limit is not None:
        input_paths = input_paths[:limit]

    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized_preview_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skipped = 0
    for path in input_paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        sample = adapter.adapt(raw)
        steps = step_parser.parse(sample)
        valid_step_ids = [step.step_id for step in steps]

        normalized = prompt_builder.build_payload(sample, steps)
        normalized = {
            "dataset": dataset,
            "source_path": str(path),
            "raw_keys": list(raw.keys()),
            "instance_id": normalized["instance_id"],
            "repository": normalized.get("repository"),
            "environment": normalized.get("environment"),
            "task": normalized["task"],
            "trajectory": sample.trajectory,
            "patch": normalized["patch"],
            "evaluation": normalized["evaluation"],
            "canonical_steps": normalized["canonical_steps"],
        }
        normalized_path = normalized_dir / f"{sample.instance_id}.json"
        normalized_path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        normalized_preview_path = normalized_preview_dir / f"{sample.instance_id}.json"
        normalized_preview_path.write_text(
            json.dumps(build_normalized_preview(normalized), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

        output_path = output_dir / f"{sample.instance_id}.json"
        if output_path.exists() and not overwrite and is_valid_existing_result(
            output_path,
            sample.instance_id,
            valid_step_ids,
        ):
            print(f"skip existing: {output_path}")
            skipped += 1
            continue

        rows.append(
            {
                "instance_id": sample.instance_id,
                "instruction": prompt_builder.build_instruction(sample, steps),
                "structured_output": {
                    "format": "json",
                    "schema": annotation_json_schema(),
                    "max_retries": structured_max_retries,
                },
                "task": sample.task,
                "repository": sample.repository,
                "environment": sample.environment,
                "patch": sample.patch,
                "evaluation": sample.evaluation,
                "trajectory": sample.trajectory,
                "canonical_steps": normalized["canonical_steps"],
                "valid_step_ids": valid_step_ids,
                "prompt_version": PROMPT_VERSION,
                "source_path": str(path),
            }
        )

    print(
        f"prepared: inputs={len(input_paths)} queued={len(rows)} skipped_existing={skipped}"
    )
    return rows


def build_normalized_preview(normalized: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": normalized["dataset"],
        "source_path": normalized["source_path"],
        "instance_id": normalized["instance_id"],
        "repository": normalized.get("repository"),
        "environment_preview": preview_text(normalized.get("environment"), max_chars=2000),
        "task_preview": preview_task(normalized.get("task")),
        "patch_preview": preview_text(normalized.get("patch"), max_chars=3000),
        "evaluation": normalized.get("evaluation"),
        "canonical_steps_preview": [
            preview_step(step) for step in normalized.get("canonical_steps", [])
        ],
    }


def preview_task(task: Any) -> Any:
    if not isinstance(task, list):
        return preview_text(task, max_chars=4000)

    return [
        {
            "role": message.get("role"),
            "content": preview_text(message.get("content"), max_chars=4000),
        }
        for message in task
        if isinstance(message, dict)
    ]


def preview_step(step: dict[str, Any]) -> dict[str, Any]:
    content = step.get("content")
    if not isinstance(content, dict):
        return {
            "step_id": step.get("step_id"),
            "content_preview": preview_text(content, max_chars=1200),
        }

    if content.get("type") == "agent_turn":
        return {
            "step_id": step.get("step_id"),
            "type": "agent_turn",
            "agent": preview_message(content.get("agent_message"), max_chars=2400),
            "actions": preview_text(content.get("actions"), max_chars=1600),
            "observations": [
                preview_message(observation, max_chars=1800)
                for observation in content.get("observations", [])
                if isinstance(observation, dict)
            ],
            "context_messages": [
                preview_message(message, max_chars=1600)
                for message in content.get("context_messages", [])
                if isinstance(message, dict)
            ],
        }

    action = content.get("action") if isinstance(content.get("action"), dict) else {}
    tool_calls = []
    for tool_call in action.get("tool_calls") or []:
        function = tool_call.get("function") if isinstance(tool_call, dict) else {}
        if not isinstance(function, dict):
            continue
        tool_calls.append(
            {
                "name": function.get("name"),
                "arguments": preview_text(function.get("arguments"), max_chars=1000),
            }
        )

    return {
        "step_id": step.get("step_id"),
        "action_type": action.get("type"),
        "reasoning": preview_text(action.get("reasoning_text"), max_chars=1400),
        "tool_calls": tool_calls,
        "observations": preview_text(content.get("observations"), max_chars=1600),
    }


def preview_message(message: Any, max_chars: int) -> dict[str, Any]:
    if not isinstance(message, dict):
        return {}
    return {
        "role": message.get("role"),
        "content": preview_text(
            message.get("content", message.get("text", message.get("message"))),
            max_chars=max_chars,
        ),
        "extra": preview_text(message.get("extra"), max_chars=800)
        if "extra" in message
        else None,
    }


def preview_text(value: Any, max_chars: int) -> dict[str, Any]:
    if value is None:
        return {"text": None, "chars": 0, "truncated": False}

    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)

    normalized = " ".join(text.split())
    truncated = len(normalized) > max_chars
    if truncated:
        half = max((max_chars - 32) // 2, 0)
        normalized = normalized[:half] + " ...[truncated]... " + normalized[-half:]

    return {
        "text": normalized,
        "chars": len(text),
        "truncated": truncated,
    }


def collect_input_paths(path: Path, dataset: str | None = None) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    paths = sorted(child for child in path.iterdir() if child.suffix == ".json" and child.is_file())
    if dataset:
        paths = [
            child
            for child in paths
            if json.loads(child.read_text(encoding="utf-8")).get("harness") in (None, dataset)
        ]
    return paths


def is_valid_existing_result(path: Path, instance_id: str, valid_step_ids: list[int]) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        annotation = parse_annotation(
            {
                "instance_id": payload["instance_id"],
                "step_reviews": payload.get("step_reviews", []),
            }
        )
        validate_annotation_against_steps(annotation, instance_id, valid_step_ids)
        return True
    except Exception:
        return False


def save_annotation_outputs(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    model: str,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    failures: list[str] = []

    for row in rows:
        instance_id = str(row.get("instance_id"))
        generation = row.get("generation")
        failed_dir = output_dir / "_failed"
        if not generation:
            failures.append(
                f"{instance_id}: model generation is empty. Check Distilabel pipeline.log in the cache dir."
            )
            continue

        try:
            annotation = parse_annotation(generation)
            validate_annotation_against_steps(
                annotation=annotation,
                instance_id=instance_id,
                valid_step_ids=list(row["valid_step_ids"]),
            )
        except Exception as exc:
            failed_dir.mkdir(parents=True, exist_ok=True)
            failed_path = failed_dir / f"{instance_id}.generation.txt"
            failed_path.write_text(str(generation), encoding="utf-8")
            failures.append(f"{instance_id}: invalid annotation output: {exc}")
            continue

        result = {
            **annotation_to_dict(annotation),
            "metadata": {
                "model": row.get("model_name") or model,
                "prompt_version": row.get("prompt_version") or PROMPT_VERSION,
                "source_path": row.get("source_path"),
            },
        }
        output_path = output_dir / f"{annotation.instance_id}.json"
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        saved += 1

    if failures:
        joined = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(f"Some annotations failed validation or generation:\n{joined}")

    return saved


def build_extra_body(
    *,
    base_url: str | None,
    model: str,
    model_config: dict[str, Any],
    disable_thinking: bool,
    enable_thinking: bool,
) -> dict[str, Any] | None:
    configured = model_config.get("extra_body")
    if isinstance(configured, dict):
        return configured

    if enable_thinking:
        return None

    is_deepseek = "deepseek" in (base_url or "").lower() or model.startswith("deepseek-")
    if disable_thinking or is_deepseek:
        return {"thinking": {"type": "disabled"}}

    return None


def run_and_save_each(
    rows: list[dict[str, Any]],
    pipeline_config: DistilabelPipelineConfig,
    output_dir: Path,
) -> int:
    saved = 0
    failures: list[str] = []

    for index, row in enumerate(rows, start=1):
        instance_id = row["instance_id"]
        print(f"annotating {index}/{len(rows)}: {instance_id}")
        try:
            annotated_rows = run_annotation_pipeline([row], pipeline_config)
            saved += save_annotation_outputs(
                annotated_rows,
                output_dir=output_dir,
                model=pipeline_config.model,
            )
        except Exception as exc:
            failures.append(f"{instance_id}: {exc}")
            print(f"failed: {instance_id}: {exc}")

    if failures:
        joined = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(f"Some samples failed:\n{joined}")

    return saved


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


if __name__ == "__main__":
    main()
