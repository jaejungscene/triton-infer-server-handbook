#!/usr/bin/env bash
# Convert a validated ONNX artifact into an atomically published TensorRT engine.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

INPUT=""
OUTPUT=""
PRECISION="fp16"
WORKSPACE_MIB="4096"
MIN_SHAPES="input:1x3x224x224"
OPT_SHAPES="input:8x3x224x224"
MAX_SHAPES="input:8x3x224x224"
EXPECTED_SHA256=""
CALIBRATION_CACHE=""

require_value() {
    if [[ $# -lt 2 ]]; then
        echo "ERROR: $1 requires a value" >&2
        exit 2
    fi
    if [[ -z "$2" ]]; then
        echo "ERROR: $1 requires a non-empty value" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --input|--output|--precision|--workspace-mib|--min-shapes|--opt-shapes|--max-shapes|--expected-sha256|--calibration-cache)
            require_value "$@"
            case $1 in
                --input) INPUT="$2" ;;
                --output) OUTPUT="$2" ;;
                --precision) PRECISION="$2" ;;
                --workspace-mib) WORKSPACE_MIB="$2" ;;
                --min-shapes) MIN_SHAPES="$2" ;;
                --opt-shapes) OPT_SHAPES="$2" ;;
                --max-shapes) MAX_SHAPES="$2" ;;
                --expected-sha256) EXPECTED_SHA256="$2" ;;
                --calibration-cache) CALIBRATION_CACHE="$2" ;;
            esac
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --input model.onnx --output model.plan [options]"
            echo "  --precision fp16|fp32|int8"
            echo "  --min-shapes input:1x3x224x224 [--opt-shapes ... --max-shapes ...]"
            echo "  --workspace-mib 4096 [--expected-sha256 <digest>]"
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            exit 2
            ;;
    esac
done

if [[ -z "${INPUT}" || -z "${OUTPUT}" ]]; then
    echo "ERROR: --input and --output are required" >&2
    exit 2
fi
if [[ ! "${WORKSPACE_MIB}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --workspace-mib must be a positive integer" >&2
    exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python 3 is required for profile validation" >&2
    exit 1
fi
if ! command -v trtexec >/dev/null 2>&1; then
    echo "ERROR: trtexec not found. Run inside the pinned converter image." >&2
    exit 1
fi

validation_args=(
    --input "${INPUT}"
    --output "${OUTPUT}"
    --min-shapes "${MIN_SHAPES}"
    --opt-shapes "${OPT_SHAPES}"
    --max-shapes "${MAX_SHAPES}"
)
if [[ -n "${EXPECTED_SHA256}" ]]; then
    validation_args+=(--expected-sha256 "${EXPECTED_SHA256}")
fi
if [[ -n "${CALIBRATION_CACHE}" ]]; then
    validation_args+=(--calibration-cache "${CALIBRATION_CACHE}")
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_tensorrt_profile.py" "${validation_args[@]}"

precision_args=()
case "${PRECISION}" in
    fp16) precision_args+=(--fp16) ;;
    int8) precision_args+=(--int8) ;;
    fp32) ;;
    *)
        echo "ERROR: --precision must be fp16, fp32, or int8" >&2
        exit 2
        ;;
esac
if [[ -n "${CALIBRATION_CACHE}" ]]; then
    if [[ "${PRECISION}" != "int8" ]]; then
        echo "ERROR: --calibration-cache requires --precision int8" >&2
        exit 2
    fi
    precision_args+=(--calib="${CALIBRATION_CACHE}")
fi

output_dir="$(dirname "${OUTPUT}")"
output_name="$(basename "${OUTPUT}")"
temporary_output="$(mktemp "${output_dir}/.${output_name}.XXXXXX")"
cleanup() {
    rm -f "${temporary_output}"
}
trap cleanup EXIT

echo "[convert] ONNX -> TensorRT"
echo "  Input:     ${INPUT}"
echo "  Output:    ${OUTPUT}"
echo "  Precision: ${PRECISION}"
echo "  Profiles:  ${MIN_SHAPES} / ${OPT_SHAPES} / ${MAX_SHAPES}"

trtexec \
    --onnx="${INPUT}" \
    --saveEngine="${temporary_output}" \
    "${precision_args[@]}" \
    --memPoolSize="workspace:${WORKSPACE_MIB}" \
    --minShapes="${MIN_SHAPES}" \
    --optShapes="${OPT_SHAPES}" \
    --maxShapes="${MAX_SHAPES}" \
    --buildOnly

if [[ ! -s "${temporary_output}" ]]; then
    echo "ERROR: trtexec produced an empty engine" >&2
    exit 1
fi
mv -f "${temporary_output}" "${OUTPUT}"
trap - EXIT

echo "[convert] Done: ${OUTPUT}"
