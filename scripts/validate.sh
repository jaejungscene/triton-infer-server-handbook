#!/usr/bin/env bash
# =============================================================================
# validate.sh — config.pbtxt 유효성 검사
# =============================================================================
# 사용법:
#   ./scripts/validate.sh                           # 전체 검사
#   ./scripts/validate.sh models/serving/vision/     # 특정 디렉토리만
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET="${1:-${PROJECT_ROOT}/models}"

echo "[validate] Scanning: ${TARGET}"

errors=0
checked=0

# Find all config.pbtxt files
while IFS= read -r config_file; do
    checked=$((checked + 1))
    relative=$(python3 - "${PROJECT_ROOT}" "${config_file}" <<'PY'
import os
import sys

print(os.path.relpath(sys.argv[2], sys.argv[1]))
PY
)

    # Basic syntax checks
    # 1. Check for required fields (name or backend/platform)
    if ! grep -qE '^\s*(backend|platform)\s*:' "${config_file}"; then
        echo "  ERROR [${relative}]: Missing 'backend' or 'platform' field"
        errors=$((errors + 1))
        continue
    fi

    # 2. Check for input/output definitions (except ensemble)
    if ! grep -qE '^\s*platform\s*:\s*"ensemble"' "${config_file}"; then
        if ! grep -qE '^\s*input\s*\[' "${config_file}"; then
            echo "  WARN  [${relative}]: No 'input' definition found"
        fi
        if ! grep -qE '^\s*output\s*\[' "${config_file}"; then
            echo "  WARN  [${relative}]: No 'output' definition found"
        fi
    fi

    # 3. Check for balanced brackets
    open_brackets=$(grep -o '\[' "${config_file}" 2>/dev/null | wc -l || true)
    close_brackets=$(grep -o '\]' "${config_file}" 2>/dev/null | wc -l || true)
    if [[ "${open_brackets}" -ne "${close_brackets}" ]]; then
        echo "  ERROR [${relative}]: Unbalanced brackets ([ ${open_brackets} vs ] ${close_brackets})"
        errors=$((errors + 1))
        continue
    fi

    open_braces=$(grep -o '{' "${config_file}" 2>/dev/null | wc -l || true)
    close_braces=$(grep -o '}' "${config_file}" 2>/dev/null | wc -l || true)
    if [[ "${open_braces}" -ne "${close_braces}" ]]; then
        echo "  ERROR [${relative}]: Unbalanced braces ({ ${open_braces} vs } ${close_braces})"
        errors=$((errors + 1))
        continue
    fi

    echo "  OK    [${relative}]"
done < <(find "${TARGET}" -name "config.pbtxt" -type f | sort)

echo ""
echo "[validate] Result: ${checked} files checked, ${errors} errors"

if [[ ${errors} -gt 0 ]]; then
    echo "[validate] FAILED"
    exit 1
fi

echo "[validate] PASSED"
