#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

CONFIG=config/example.yaml
N_WORKERS=4

# Review every trajectory JSON directly under INPUT
DATASET=mini_swe_agent
INPUT=output/pipeline/traj
RUNNER=llm
REVIEW_RUNTIME=docker
USE_CACHE=true
MODEL=deepseek-flash
TEMPERATURE=0.0
MAX_NEW_TOKENS=4096
TIMEOUT_SECONDS=120
MAX_RETRIES=2
COMPACT_FOR_MODEL=false
MAX_TASK_CHARS=12000
MAX_PATCH_CHARS=20000
MAX_STEP_CHARS=6000
MAX_TOTAL_STEP_CHARS=60000

exec .venv/bin/python main.py --config "$CONFIG" --mode review-only --n-workers "$N_WORKERS" \
  --input "$INPUT" \
  --set review.dataset="$DATASET" \
  --set review.runner="$RUNNER" \
  --set review.runtime="$REVIEW_RUNTIME" \
  --set review.use_cache="$USE_CACHE" \
  --set review.model.model="$MODEL" \
  --set review.model.temperature="$TEMPERATURE" \
  --set review.model.max_new_tokens="$MAX_NEW_TOKENS" \
  --set review.model.timeout_seconds="$TIMEOUT_SECONDS" \
  --set review.model.max_retries="$MAX_RETRIES" \
  --set review.prompt_budget.compact_for_model="$COMPACT_FOR_MODEL" \
  --set review.prompt_budget.max_task_chars="$MAX_TASK_CHARS" \
  --set review.prompt_budget.max_patch_chars="$MAX_PATCH_CHARS" \
  --set review.prompt_budget.max_step_chars="$MAX_STEP_CHARS" \
  --set review.prompt_budget.max_total_step_chars="$MAX_TOTAL_STEP_CHARS"
