#!/usr/bin/env bash
# =============================================================================
# health_check.sh — Triton 서버 및 모델 상태 확인
# =============================================================================
# 사용법:
#   ./scripts/health_check.sh                      # localhost:8000
#   ./scripts/health_check.sh http://triton:8000   # 커스텀 URL
# =============================================================================

set -euo pipefail

if ! command -v python3 &>/dev/null; then
    echo "[health] python3 is required to validate the endpoint and response" >&2
    exit 1
fi

if ! BASE_URL=$(python3 - "${1:-http://localhost:8000}" <<'PYTHON'
import sys
from urllib.parse import urlsplit, urlunsplit

url = sys.argv[1]
try:
    parsed = urlsplit(url)
    port = parsed.port
except ValueError as error:
    raise SystemExit(f"[health] invalid base URL: {error}") from error

if (
    parsed.scheme not in {"http", "https"}
    or not parsed.hostname
    or parsed.username is not None
    or parsed.password is not None
    or parsed.path not in {"", "/"}
    or parsed.query
    or parsed.fragment
    or (port is not None and not 1 <= port <= 65535)
):
    raise SystemExit(
        "[health] base URL must be an HTTP(S) origin without credentials, path, query, or fragment"
    )

host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
netloc = f"{host}:{port}" if port is not None else host
print(urlunsplit((parsed.scheme, netloc, "", "", "")))
PYTHON
); then
    exit 2
fi

AUTH_HEADER_FILE=""
cleanup() {
    if [[ -n "${AUTH_HEADER_FILE}" ]]; then
        rm -f "${AUTH_HEADER_FILE}"
    fi
}
trap cleanup EXIT

if [[ "${TRITON_AUTH_TOKEN:-}" == *$'\r'* || "${TRITON_AUTH_TOKEN:-}" == *$'\n'* ]]; then
    echo "[health] TRITON_AUTH_TOKEN must not contain line breaks" >&2
    exit 2
fi
if [[ -n "${TRITON_AUTH_TOKEN:-}" ]]; then
    AUTH_HEADER_FILE="$(mktemp)"
    chmod 600 "${AUTH_HEADER_FILE}"
    printf 'Authorization: Bearer %s\n' "${TRITON_AUTH_TOKEN}" > "${AUTH_HEADER_FILE}"
fi

health_curl() {
    if [[ -n "${AUTH_HEADER_FILE}" ]]; then
        command curl --header "@${AUTH_HEADER_FILE}" "$@"
    else
        command curl "$@"
    fi
}

CURL_ARGS=(-fsS --connect-timeout 3 --max-time 10)

echo "[health] Checking Triton server at ${BASE_URL}"
echo "=========================================="

# Server liveness
echo -n "Server Live:  "
if health_curl --proto "=http,https" "${CURL_ARGS[@]}" \
    "${BASE_URL}/v2/health/live" > /dev/null 2>&1; then
    echo "OK"
else
    echo "FAIL"
    echo "[health] Server is not running"
    exit 1
fi

# Server readiness
echo -n "Server Ready: "
if health_curl --proto "=http,https" "${CURL_ARGS[@]}" \
    "${BASE_URL}/v2/health/ready" > /dev/null 2>&1; then
    echo "OK"
else
    echo "FAIL (server is live but not ready)"
    exit 1
fi

echo "=========================================="

# Model status
echo "[health] Loaded Models:"
if ! models_response=$(health_curl --proto "=http,https" "${CURL_ARGS[@]}" \
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
