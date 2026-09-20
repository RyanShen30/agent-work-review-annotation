#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

CONFIG=config/example.yaml
N_WORKERS=4

# Trajectory generation
HARNESS=mini_swe_agent
GENERATION_DATASET=auto
GENERATION_MODEL=deepseek/deepseek-flash
RUNTIME=docker
PLATFORM=mac
BENCHMARK_PATH=data/SWE-bench_Verified
INSTANCE=100
WORKSPACE=.
STEP_LIMIT=50
COST_LIMIT=3.0
COMMAND_TIMEOUT=120
DROP_PARAMS=true
KEEP_IMAGE=true
SAVE_FINAL_SNAPSHOT=true
PULL_TIMEOUT=900

# Official SWE-bench evaluation
RUN_OFFICIAL_EVALUATION=true
EVALUATION_TIMEOUT=1800
EVALUATION_OPEN_FILE_LIMIT=4096

# Review
DATASET=mini_swe_agent
RUNNER=llm
REVIEW_RUNTIME=docker
USE_CACHE=true
REVIEW_MODEL=deepseek-v4-pro
TEMPERATURE=0.0
MAX_NEW_TOKENS=4096
TIMEOUT_SECONDS=120
MAX_RETRIES=2
COMPACT_FOR_MODEL=false
MAX_TASK_CHARS=12000
MAX_PATCH_CHARS=20000
MAX_STEP_CHARS=6000
MAX_TOTAL_STEP_CHARS=60000
CLEANUP_POLICY=on_success

exec .venv/bin/python main.py --config "$CONFIG" --mode full --n-workers "$N_WORKERS" \
  --set generation.harness="$HARNESS" \
  --set generation.dataset="$GENERATION_DATASET" \
  --set generation.model="$GENERATION_MODEL" \
  --set generation.runtime="$RUNTIME" \
  --set generation.platform="$PLATFORM" \
  --set generation.benchmark_path="$BENCHMARK_PATH" \
  --set generation.instance="$INSTANCE" \
  --set generation.workspace="$WORKSPACE" \
  --set generation.step_limit="$STEP_LIMIT" \
  --set generation.cost_limit="$COST_LIMIT" \
  --set generation.command_timeout="$COMMAND_TIMEOUT" \
  --set generation.model_kwargs.drop_params="$DROP_PARAMS" \
  --set generation.environment_kwargs.keep_image="$KEEP_IMAGE" \
  --set generation.environment_kwargs.save_final_snapshot="$SAVE_FINAL_SNAPSHOT" \
  --set generation.environment_kwargs.pull_timeout="$PULL_TIMEOUT" \
  --set evaluation.enabled="$RUN_OFFICIAL_EVALUATION" \
  --set evaluation.runner=official_swebench \
  --set evaluation.timeout="$EVALUATION_TIMEOUT" \
  --set evaluation.open_file_limit="$EVALUATION_OPEN_FILE_LIMIT" \
  --set review.dataset="$DATASET" \
  --set review.runner="$RUNNER" \
  --set review.runtime="$REVIEW_RUNTIME" \
  --set review.use_cache="$USE_CACHE" \
  --set review.model.model="$REVIEW_MODEL" \
  --set review.model.temperature="$TEMPERATURE" \
  --set review.model.max_new_tokens="$MAX_NEW_TOKENS" \
  --set review.model.timeout_seconds="$TIMEOUT_SECONDS" \
  --set review.model.max_retries="$MAX_RETRIES" \
  --set review.prompt_budget.compact_for_model="$COMPACT_FOR_MODEL" \
  --set review.prompt_budget.max_task_chars="$MAX_TASK_CHARS" \
  --set review.prompt_budget.max_patch_chars="$MAX_PATCH_CHARS" \
  --set review.prompt_budget.max_step_chars="$MAX_STEP_CHARS" \
  --set review.prompt_budget.max_total_step_chars="$MAX_TOTAL_STEP_CHARS" \
  --set cleanup_policy="$CLEANUP_POLICY"
