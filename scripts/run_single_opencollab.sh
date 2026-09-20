#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

CONFIG=config/example.yaml

# Trajectory generation
HARNESS=opencollab
DATASET=auto
MODEL=openai/gpt-5.4
PROVIDER=openai
MODE=team
TOKEN_BUDGET=1000000
TEAM_CONFIG=config/opencollab_team.yaml
RUNTIME=docker
BENCHMARK_PATH=data/SWE-bench_Verified
INSTANCE=10
WORKSPACE=.
STEP_LIMIT=50
COMMAND_TIMEOUT=120
USE_WORKTREES=true
PREBUILD_TEAM=false
SERIALIZE_TURNS=false

exec .venv/bin/python main.py --config "$CONFIG" --mode traj-only \
  --set generation.harness="$HARNESS" \
  --set generation.dataset="$DATASET" \
  --set generation.model="$MODEL" \
  --set generation.runtime="$RUNTIME" \
  --set generation.benchmark_path="$BENCHMARK_PATH" \
  --set generation.instance="$INSTANCE" \
  --set generation.workspace="$WORKSPACE" \
  --set generation.step_limit="$STEP_LIMIT" \
  --set generation.command_timeout="$COMMAND_TIMEOUT" \
  --set generation.harness_kwargs.mode="$MODE" \
  --set generation.harness_kwargs.provider="$PROVIDER" \
  --set generation.harness_kwargs.budget="$TOKEN_BUDGET" \
  --set generation.harness_kwargs.team_config="$TEAM_CONFIG" \
  --set generation.harness_kwargs.use_worktrees="$USE_WORKTREES" \
  --set generation.harness_kwargs.prebuild_team="$PREBUILD_TEAM" \
  --set generation.harness_kwargs.serialize_turns="$SERIALIZE_TURNS"
