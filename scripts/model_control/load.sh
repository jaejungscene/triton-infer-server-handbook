#!/usr/bin/env bash
# =============================================================================
# load.sh — Triton 런타임 모델 로드 (Dynamic Loading)
# =============================================================================
# 사용법:
#   ./scripts/model_control/load.sh yolox
#   ./scripts/model_control/load.sh yolox http://triton:8000
#
# 사전조건: --model-control-mode=explicit 로 서버 시작
# =============================================================================

set -euo pipefail

MODEL_NAME="${1:?Usage: $0 <model_name> [base_url]}"
BASE_URL="${2:-http://localhost:8000}"

echo "[load] Loading model: ${MODEL_NAME}"
response=$(curl -sS -w "\n%{http_code}" -X POST "${BASE_URL}/v2/repository/models/${MODEL_NAME}/load")
http_code=$(echo "${response}" | tail -1)
body=$(printf '%s\n' "${response}" | sed '$d')

if [[ "${http_code}" == "200" ]]; then
    echo "[load] OK: ${MODEL_NAME} loaded successfully"
else
    echo "[load] FAIL (HTTP ${http_code}): ${body}" >&2
    exit 1
fi
