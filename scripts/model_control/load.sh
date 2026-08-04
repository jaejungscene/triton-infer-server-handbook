#!/usr/bin/env bash
# Load a model in EXPLICIT control mode and verify that it becomes ready.

set -euo pipefail

MODEL_NAME="${1:?Usage: $0 <model_name> [base_url] [timeout_seconds]}"
BASE_URL="${2:-http://localhost:8000}"
TIMEOUT_SECONDS="${3:-120}"
BASE_URL="${BASE_URL%/}"

if [[ ! "${MODEL_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "[load] Invalid model name: ${MODEL_NAME}" >&2
    exit 2
fi
if [[ ! "${TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[load] timeout_seconds must be a positive integer" >&2
    exit 2
fi

body_file=$(mktemp)
trap 'rm -f "${body_file}"' EXIT

echo "[load] Requesting model load: ${MODEL_NAME}"
if ! http_code=$(curl -sS \
    --connect-timeout 3 \
    --max-time "${TIMEOUT_SECONDS}" \
    -o "${body_file}" \
    -w "%{http_code}" \
    -X POST \
    "${BASE_URL}/v2/repository/models/${MODEL_NAME}/load"); then
    echo "[load] Model-control request failed" >&2
    exit 1
fi

if [[ "${http_code}" != "200" ]]; then
    echo "[load] FAIL (HTTP ${http_code}): $(<"${body_file}")" >&2
    exit 1
fi

deadline=$((SECONDS + TIMEOUT_SECONDS))
until curl -fsS --connect-timeout 3 --max-time 5 \
    "${BASE_URL}/v2/models/${MODEL_NAME}/ready" > /dev/null; do
    if (( SECONDS >= deadline )); then
        echo "[load] Timed out waiting for ${MODEL_NAME} to become ready" >&2
        exit 1
    fi
    sleep 1
done

echo "[load] OK: ${MODEL_NAME} is ready"
