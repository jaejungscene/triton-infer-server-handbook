#!/usr/bin/env bash
# Reload a model in EXPLICIT control mode after artifacts have been published.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="${1:?Usage: $0 <model_name> [base_url] [timeout_seconds]}"
BASE_URL="${2:-http://localhost:8000}"
TIMEOUT_SECONDS="${3:-120}"

echo "[reload] Reloading ${MODEL_NAME} in EXPLICIT mode"
echo "[reload] This unload/load sequence creates an availability gap for a single replica."
echo "[reload] Publish and validate immutable artifacts before running this command."

"${SCRIPT_DIR}/unload.sh" "${MODEL_NAME}" "${BASE_URL}" "${TIMEOUT_SECONDS}"
"${SCRIPT_DIR}/load.sh" "${MODEL_NAME}" "${BASE_URL}" "${TIMEOUT_SECONDS}"

echo "[reload] OK: ${MODEL_NAME} reloaded and ready"
