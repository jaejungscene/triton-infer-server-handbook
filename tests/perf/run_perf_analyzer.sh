#!/usr/bin/env bash
# =============================================================================
# run_perf_analyzer.sh — Triton perf_analyzer 래핑 스크립트
# =============================================================================
# 사용법:
#   ./tests/perf/run_perf_analyzer.sh                    # 전체 모델
#   ./tests/perf/run_perf_analyzer.sh --model resnet50   # 특정 모델
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRITON_URL="${TRITON_URL:-localhost:8000}"
MODEL=""
CONCURRENCY="1:8"
CA_CERT="${TRITON_CA_CERT:-}"
RESULTS_DIR="${PERF_RESULTS_DIR:-${SCRIPT_DIR}/results}"
BASELINE="${SCRIPT_DIR}/baseline.json"
PROFILES="${SCRIPT_DIR}/profiles.json"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model|--concurrency|--url|--ca-cert)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: $1 requires a value" >&2
                exit 2
            fi
            if [[ -z "$2" ]]; then
                echo "ERROR: $1 requires a non-empty value" >&2
                exit 2
            fi
            case $1 in
                --model) MODEL="$2" ;;
                --concurrency) CONCURRENCY="$2" ;;
                --url) TRITON_URL="$2" ;;
                --ca-cert) CA_CERT="$2" ;;
            esac
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--model <name>] [--concurrency 1:8] [--url localhost:8000] [--ca-cert ca.pem]"
            exit 0
            ;;
        *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
    esac
done

runner_values=()
while IFS= read -r value; do
    runner_values+=("${value}")
done < <(python3 "${SCRIPT_DIR}/validate_runner_args.py" \
    --model "${MODEL}" --concurrency "${CONCURRENCY}" --url "${TRITON_URL}")
if [[ ${#runner_values[@]} -ne 3 ]]; then
    echo "ERROR: failed to normalize performance runner arguments" >&2
    exit 2
fi
HTTP_URL="${runner_values[0]}"
PERF_URL="${runner_values[1]}"
USE_TLS="${runner_values[2]}"

if [[ -n "${TRITON_AUTH_TOKEN:-}" && "${TRITON_AUTH_TOKEN}" == *$'\n'* ]] || \
   [[ -n "${TRITON_AUTH_TOKEN:-}" && "${TRITON_AUTH_TOKEN}" == *$'\r'* ]]; then
    echo "ERROR: TRITON_AUTH_TOKEN must not contain line breaks" >&2
    exit 2
fi
if [[ -n "${CA_CERT}" && ( ! -f "${CA_CERT}" || -L "${CA_CERT}" ) ]]; then
    echo "ERROR: --ca-cert must reference a regular CA file" >&2
    exit 2
fi
if [[ -n "${CA_CERT}" && "${USE_TLS}" != "true" ]]; then
    echo "ERROR: --ca-cert requires an https URL" >&2
    exit 2
fi

if ! command -v perf_analyzer &>/dev/null; then
    echo "ERROR: perf_analyzer not found. Install from Triton client SDK."
    echo "  Run this script inside nvcr.io/nvidia/tritonserver:<version>-py3-sdk."
    exit 1
fi

mkdir -p "${RESULTS_DIR}"
rm -f "${RESULTS_DIR}"/*_perf.csv "${RESULTS_DIR}"/*_perf.log

auth_header_file=""
curl_args=(-sf -X POST -H "Content-Type: application/json")
perf_connection_args=()
if [[ -n "${TRITON_AUTH_TOKEN:-}" ]]; then
    auth_header_file="$(mktemp "${TMPDIR:-/tmp}/triton-perf-auth.XXXXXX")"
    chmod 600 "${auth_header_file}"
    printf 'Authorization: Bearer %s\n' "${TRITON_AUTH_TOKEN}" > "${auth_header_file}"
    curl_args+=(-H "@${auth_header_file}")
    perf_connection_args+=(-H "Authorization: Bearer ${TRITON_AUTH_TOKEN}")
fi
if [[ "${USE_TLS}" == "true" ]]; then
    perf_connection_args+=(--ssl-https-verify-peer=1 --ssl-https-verify-host=2)
fi
if [[ -n "${CA_CERT}" ]]; then
    curl_args+=(--cacert "${CA_CERT}")
    perf_connection_args+=(--ssl-https-ca-certificates-file="${CA_CERT}")
fi
cleanup() {
    if [[ -n "${auth_header_file}" ]]; then
        rm -f "${auth_header_file}"
    fi
}
trap cleanup EXIT

run_perf() {
    local model_name="$1"
    local profile_output
    local result_file
    local -a perf_command=()
    local -a profile_args=()

    if [[ ! "${model_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "ERROR: unsafe model name returned by Triton: ${model_name}" >&2
        return 1
    fi

    if ! profile_output=$(python3 "${SCRIPT_DIR}/profile_args.py" \
        --profiles "${PROFILES}" \
        --model "${model_name}" \
        --perf-dir "${SCRIPT_DIR}"); then
        return 1
    fi
    while IFS= read -r profile_arg; do
        profile_args+=("${profile_arg}")
    done <<< "${profile_output}"

    echo "=========================================="
    echo "[perf] Testing model: ${model_name}"
    echo "=========================================="

    result_file="${RESULTS_DIR}/${model_name}_perf.csv"

    perf_command=(perf_analyzer \
        -m "${model_name}" \
        -u "${PERF_URL}")
    if [[ ${#perf_connection_args[@]} -gt 0 ]]; then
        perf_command+=("${perf_connection_args[@]}")
    fi
    perf_command+=( \
        --percentile=95 \
        --concurrency-range="${CONCURRENCY}" \
        --measurement-interval=10000 \
        "${profile_args[@]}" \
        -f "${result_file}")

    "${perf_command[@]}" \
        2>&1 | tee "${RESULTS_DIR}/${model_name}_perf.log"

    if [[ ! -s "${result_file}" || -L "${result_file}" ]]; then
        echo "ERROR: perf_analyzer did not create a regular non-empty CSV for ${model_name}" >&2
        return 1
    fi

    echo ""
}

if [[ -n "${MODEL}" ]]; then
    run_perf "${MODEL}"
else
    # Repository Index API에서 ready 모델만 조회
    if ! models_json=$(curl "${curl_args[@]}" \
        -d '{"ready":true}' \
        "${HTTP_URL}/v2/repository/index"); then
        echo "ERROR: Failed to query Triton Repository Index API at ${HTTP_URL}"
        exit 1
    fi

    models=$(printf '%s' "${models_json}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if not isinstance(data, list):
    raise SystemExit('Repository Index response must be a JSON array')
names = {m.get('name') for m in data if isinstance(m, dict) and m.get('name')}
for name in sorted(names):
    if not isinstance(name, str) or not __import__('re').fullmatch(r'[A-Za-z0-9._-]+', name):
        raise SystemExit(f'Unsafe model name in Repository Index: {name!r}')
    print(name)
")

    if [[ -z "${models}" ]]; then
        echo "ERROR: Repository Index API returned no ready models"
        exit 1
    fi

    while IFS= read -r model; do
        run_perf "${model}"
    done <<< "${models}"
fi

echo "=========================================="
echo "[perf] Results saved to: ${RESULTS_DIR}/"
echo "=========================================="

# Baseline 비교
if [[ ! -f "${BASELINE}" ]]; then
    echo "ERROR: Performance baseline not found: ${BASELINE}"
    exit 1
fi

compare_args=(
    --baseline "${BASELINE}"
    --results-dir "${RESULTS_DIR}"
)
if [[ -n "${MODEL}" ]]; then
    compare_args+=(--model "${MODEL}")
fi
python3 "${SCRIPT_DIR}/compare_baseline.py" "${compare_args[@]}"
