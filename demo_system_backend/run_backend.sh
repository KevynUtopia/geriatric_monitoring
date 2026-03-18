#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_VIDEO="$1"

if [[ -z "${INPUT_VIDEO}" ]]; then
  echo "Usage: bash demo_system_backend/run_backend.sh <input_video.mp4> [output_video.mp4]"
  exit 1
fi

OUTPUT_VIDEO="${2:-output.mp4}"

cd "${PROJECT_ROOT}"

python -m demo_system_backend.main \
  --input "${INPUT_VIDEO}" \
  --output "${OUTPUT_VIDEO}"

