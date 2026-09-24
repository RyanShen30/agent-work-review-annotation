#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

CONFIG=config/example.yaml

# Trajectory generation
HARNESS=mini_swe_agent
MODEL=deepseek/deepseek-flash
RUNTIME=docker
BENCHMARK_PATH=data/SWE-bench_Verified
INSTANCE=1
WORKSPACE=.
STEP_LIMIT=50
COST_LIMIT=3.0
COMMAND_TIMEOUT=120
DROP_PARAMS=true
KEEP_IMAGE=true
SAVE_FINAL_SNAPSHOT=true
CWD=/testbed
PULL_TIMEOUT=900

exec .venv/bin/python main.py --config "$CONFIG" --mode traj-only \
  --set generation.harness="$HARNESS" \
  --set generation.model="$MODEL" \
  --set generation.runtime="$RUNTIME" \
  --set generation.benchmark_path="$BENCHMARK_PATH" \
  --set generation.instance="$INSTANCE" \
  --set generation.workspace="$WORKSPACE" \
  --set generation.step_limit="$STEP_LIMIT" \
  --set generation.cost_limit="$COST_LIMIT" \
  --set generation.command_timeout="$COMMAND_TIMEOUT" \
  --set generation.model_kwargs.drop_params="$DROP_PARAMS" \
  --set generation.environment_kwargs.keep_image="$KEEP_IMAGE" \
  --set generation.environment_kwargs.save_final_snapshot="$SAVE_FINAL_SNAPSHOT" \
  --set generation.environment_kwargs.cwd="$CWD" \
  --set generation.environment_kwargs.pull_timeout="$PULL_TIMEOUT"
