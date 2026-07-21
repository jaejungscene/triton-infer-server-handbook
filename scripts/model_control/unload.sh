#!/usr/bin/env bash
# =============================================================================
# unload.sh — Triton 런타임 모델 언로드
# =============================================================================
# 사용법:
#   ./scripts/model_control/unload.sh yolox
# =============================================================================

set -euo pipefail

MODEL_NAME="${1:?Usage: $0 <model_name> [base_url]}"
BASE_URL="${2:-http://localhost:8000}"

echo "[unload] Unloading model: ${MODEL_NAME}"
response=$(curl -sS -w "\n%{http_code}" -X POST "${BASE_URL}/v2/repository/models/${MODEL_NAME}/unload")
http_code=$(echo "${response}" | tail -1)
body=$(printf '%s\n' "${response}" | sed '$d')

if [[ "${http_code}" == "200" ]]; then
    echo "[unload] OK: ${MODEL_NAME} unloaded successfully"
else
    echo "[unload] FAIL (HTTP ${http_code}): ${body}" >&2
    exit 1
fi
