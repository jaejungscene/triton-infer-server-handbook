#!/usr/bin/env bash
# =============================================================================
# health_check.sh — Triton 서버 및 모델 상태 확인
# =============================================================================
# 사용법:
#   ./scripts/health_check.sh                      # localhost:8000
#   ./scripts/health_check.sh http://triton:8000   # 커스텀 URL
# =============================================================================

set -euo pipefail

BASE_URL="${1:-http://localhost:8000}"
BASE_URL="${BASE_URL%/}"
CURL_ARGS=(-fsS --connect-timeout 3 --max-time 10)

echo "[health] Checking Triton server at ${BASE_URL}"
echo "=========================================="

# Server liveness
echo -n "Server Live:  "
if curl "${CURL_ARGS[@]}" "${BASE_URL}/v2/health/live" > /dev/null 2>&1; then
    echo "OK"
else
    echo "FAIL"
    echo "[health] Server is not running"
    exit 1
fi

# Server readiness
echo -n "Server Ready: "
if curl "${CURL_ARGS[@]}" "${BASE_URL}/v2/health/ready" > /dev/null 2>&1; then
    echo "OK"
else
    echo "FAIL (server is live but not ready)"
    exit 1
fi

echo "=========================================="

# Model status
echo "[health] Loaded Models:"
if ! command -v python3 &>/dev/null; then
    echo "[health] python3 is required to validate the Repository Index response" >&2
    exit 1
fi

if ! models_response=$(curl "${CURL_ARGS[@]}" \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{"ready":true}' \
    "${BASE_URL}/v2/repository/index"); then
    echo "[health] Repository Index API request failed" >&2
    exit 1
fi

if ! printf '%s' "${models_response}" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if not isinstance(data, list):
        raise ValueError('expected a JSON array')
    models = [model for model in data if model.get('name')]
    if not models:
        print('  FAIL: no ready models', file=sys.stderr)
        sys.exit(1)
    for m in models:
        name = m.get('name', 'unknown')
        version = m.get('version', '-')
        state = m.get('state', 'READY') or 'READY'
        print(f'  {name} (v{version}): {state}')
except Exception as e:
    print(f'  FAIL: invalid Repository Index response: {e}', file=sys.stderr)
    sys.exit(1)
"; then
    exit 1
fi

echo "=========================================="
echo "[health] Check complete"
