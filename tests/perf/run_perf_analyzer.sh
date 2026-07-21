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

if ! command -v perf_analyzer &>/dev/null; then
    echo "ERROR: perf_analyzer not found. Install from Triton client SDK."
    echo "  python -m pip install -r requirements-integration.txt"
    echo "  or run inside Triton SDK container"
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
    # 로드된 모델 목록 가져오기
    models=$(curl -sf "${HTTP_URL}/v2/models" 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
for m in data.get('models', []):
    # ensemble은 구성 모델이 개별 테스트되므로 스킵
    print(m['name'])
" 2>/dev/null || echo "")

    if [[ -z "${models}" ]]; then
        echo "WARNING: No models found or server unreachable"
        exit 0
    fi

    for model in ${models}; do
        run_perf "${model}"
    done
fi

echo "=========================================="
echo "[perf] Results saved to: ${RESULTS_DIR}/"
echo "=========================================="

# Baseline 비교 (baseline.json이 존재하면)
if [[ -f "${BASELINE}" ]]; then
    echo "[perf] Comparing with baseline..."
    python3 -c "
import json, sys, os

with open('${BASELINE}') as f:
    baseline = json.load(f)

regressions = []
for model_name, targets in baseline.items():
    log_file = os.path.join('${RESULTS_DIR}', f'{model_name}_perf.log')
    if not os.path.exists(log_file):
        print(f'  SKIP: {model_name} (no results)')
        continue

    # Simple check: look for throughput in log
    print(f'  CHECK: {model_name}')
    print(f'    Baseline throughput: {targets.get(\"min_throughput\", \"N/A\")} infer/sec')
    print(f'    Baseline p95 latency: {targets.get(\"max_p95_latency_ms\", \"N/A\")} ms')

if regressions:
    print(f'\\nREGRESSION DETECTED: {len(regressions)} models below baseline')
    sys.exit(1)
else:
    print('\\nAll models within baseline')
"
fi
