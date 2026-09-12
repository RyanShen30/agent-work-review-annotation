#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

exec env MSWEA_SILENT_STARTUP=1 \
  agentic_review_annotation_distilabel/thirdparty/mini-swe-agent/.venv/bin/python \
  -m agentic_review_annotation_distilabel.agents.run \
  --config agentic_review_annotation_distilabel/config/config.mini_swe_agent.yaml
