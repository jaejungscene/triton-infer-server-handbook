#!/usr/bin/env bash
# Shared validation and curl authentication for explicit model control scripts.

MODEL_CONTROL_TMP_FILES=()
MODEL_CONTROL_AUTH_HEADER_FILE=""

model_control_temp_file() {
    local output_variable="$1"
    local temporary_file
    temporary_file="$(mktemp)"
    chmod 600 "${temporary_file}"
    MODEL_CONTROL_TMP_FILES+=("${temporary_file}")
    printf -v "${output_variable}" '%s' "${temporary_file}"
}

model_control_cleanup() {
    if (( ${#MODEL_CONTROL_TMP_FILES[@]} > 0 )); then
        rm -f "${MODEL_CONTROL_TMP_FILES[@]}"
    fi
}

model_control_curl() {
    if [[ -n "${MODEL_CONTROL_AUTH_HEADER_FILE}" ]]; then
        command curl --header "@${MODEL_CONTROL_AUTH_HEADER_FILE}" "$@"
    else
        command curl "$@"
    fi
}

model_control_init() {
    local operation="$1"
    local model_name="${2:-}"
    local base_url="${3:-http://localhost:8000}"
    local timeout_seconds="${4:-120}"

    if [[ -z "${model_name}" ]]; then
        echo "Usage: ${0} <model_name> [base_url] [timeout_seconds]" >&2
        exit 2
    fi
    if [[ ! "${model_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "[${operation}] Invalid model name: ${model_name}" >&2
        exit 2
    fi
    if [[ ! "${base_url}" =~ ^https?://([A-Za-z0-9.-]+|\[[0-9A-Fa-f:]+\])(:[0-9]{1,5})?/?$ ]]; then
        echo "[${operation}] base_url must be an HTTP(S) origin without credentials or a path" >&2
        exit 2
    fi
    if [[ ! "${timeout_seconds}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[${operation}] timeout_seconds must be a positive integer" >&2
        exit 2
    fi
    if [[ "${TRITON_AUTH_TOKEN:-}" == *$'\r'* || "${TRITON_AUTH_TOKEN:-}" == *$'\n'* ]]; then
        echo "[${operation}] TRITON_AUTH_TOKEN must not contain line breaks" >&2
        exit 2
    fi

    MODEL_NAME="${model_name}"
    BASE_URL="${base_url%/}"
    TIMEOUT_SECONDS="${timeout_seconds}"
    if [[ -n "${TRITON_AUTH_TOKEN:-}" ]]; then
        model_control_temp_file MODEL_CONTROL_AUTH_HEADER_FILE
        printf 'Authorization: Bearer %s\n' "${TRITON_AUTH_TOKEN}" \
            > "${MODEL_CONTROL_AUTH_HEADER_FILE}"
    fi

    trap model_control_cleanup EXIT
}
