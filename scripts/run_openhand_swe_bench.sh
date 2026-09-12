#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

exec env OPENHANDS_SUPPRESS_BANNER=1 DOCKER_DEFAULT_PLATFORM=linux/amd64 \
  agentic_review_annotation_distilabel/thirdparty/openhands/.venv/bin/python \
  -m agentic_review_annotation_distilabel.agents.run \
  --config agentic_review_annotation_distilabel/config/config.openhand.yaml
