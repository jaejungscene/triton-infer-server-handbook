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
RESULTS_DIR="${SCRIPT_DIR}/results"
BASELINE="${SCRIPT_DIR}/baseline.json"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)       MODEL="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --url)         TRITON_URL="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--model <name>] [--concurrency 1:8] [--url localhost:8000]"
            exit 0
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

mkdir -p "${RESULTS_DIR}"
rm -f "${RESULTS_DIR}"/*_perf.csv "${RESULTS_DIR}"/*_perf.log

if ! command -v perf_analyzer &>/dev/null; then
    echo "ERROR: perf_analyzer not found. Install from Triton client SDK."
    echo "  Run this script inside nvcr.io/nvidia/tritonserver:<version>-py3-sdk."
    exit 1
fi

HTTP_URL="${TRITON_URL}"
if [[ "${HTTP_URL}" != http://* && "${HTTP_URL}" != https://* ]]; then
    HTTP_URL="http://${HTTP_URL}"
fi
PERF_URL="${TRITON_URL#http://}"
PERF_URL="${PERF_URL#https://}"

run_perf() {
    local model_name="$1"
    echo "=========================================="
    echo "[perf] Testing model: ${model_name}"
    echo "=========================================="

    perf_analyzer \
        -m "${model_name}" \
        -u "${PERF_URL}" \
        --percentile=95 \
        --concurrency-range="${CONCURRENCY}" \
        --measurement-interval=10000 \
        -f "${RESULTS_DIR}/${model_name}_perf.csv" \
        2>&1 | tee "${RESULTS_DIR}/${model_name}_perf.log"

    echo ""
}

if [[ -n "${MODEL}" ]]; then
    run_perf "${MODEL}"
else
    # Repository Index API에서 ready 모델만 조회
    if ! models_json=$(curl -sf \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"ready":true}' \
        "${HTTP_URL}/v2/repository/index"); then
        echo "ERROR: Failed to query Triton Repository Index API at ${HTTP_URL}"
        exit 1
    fi

    models=$(printf '%s' "${models_json}" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for name in sorted({m['name'] for m in data if m.get('name')}):
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
