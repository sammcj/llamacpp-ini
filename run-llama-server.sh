#!/usr/bin/env bash
#
# Launch llama-server in router mode with this host's defaults.
#
# Usage:
#   ./run-llama-server.sh [extra llama-server args...]
#
# Env overrides:
#   LLAMA_SERVER_BIN   path to the llama-server binary (default: first on PATH)
#   LLAMA_BASE_INI     base preset to extend (default: <hostname>.ini, then samm-mbp.ini)
#   MODELS_SRC         LM Studio model tree to scan (default: ~/.lmstudio/models)
#   MODELS_DIR         flat staging dir passed to --models-dir (default: <here>/models)
#   SKIP_SYNC=1        skip the symlink refresh before launch
#
set -euo pipefail

# Directory this script lives in, so the preset path is stable regardless of CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Per-host tuning. The base preset and sync thresholds differ per machine (e.g.
# a 128 GB M5 Max vs a 16 GB M1 Air), so both are selected by short hostname:
#   <hostname>.ini   base server preset (falls back to samm-mbp.ini)
#   <hostname>.env   optional; sourced here to set SIZE_* sync thresholds etc.
# Override the preset explicitly with LLAMA_BASE_INI.
HOST="$(hostname -s)"
HOST_ENV="${SCRIPT_DIR}/${HOST}.env"
if [[ -f "${HOST_ENV}" ]]; then
    # shellcheck source=/dev/null
    source "${HOST_ENV}"
fi
BASE_INI="${LLAMA_BASE_INI:-${SCRIPT_DIR}/${HOST}.ini}"
if [[ ! -f "${BASE_INI}" ]]; then
    BASE_INI="${SCRIPT_DIR}/samm-mbp.ini"
fi
# Export so sync-models.py resolves the same base preset instead of re-guessing.
export LLAMA_BASE_INI="${BASE_INI}"

# llama-server binary: honour $LLAMA_SERVER_BIN, otherwise find it on PATH.
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"
if ! command -v "${LLAMA_SERVER_BIN}" >/dev/null 2>&1; then
    echo "error: '${LLAMA_SERVER_BIN}' not found on PATH; set LLAMA_SERVER_BIN to its full path." >&2
    exit 1
fi

# llama-server --models-dir only scans the top level (no recursion), so LM
# Studio's nested publisher/repo/file.gguf layout and the bytkim folder are
# invisible to it. sync-models.py flattens the tree into a symlink farm that
# the scanner understands; we refresh it on every launch (set SKIP_SYNC=1 to
# skip). --models-dir then points at that flat staging dir.
MODELS_DIR="${MODELS_DIR:-${SCRIPT_DIR}/models}"
if [[ "${SKIP_SYNC:-0}" != "1" ]]; then
    MODELS_DIR="${MODELS_DIR}" "${SCRIPT_DIR}/sync-models.py"
fi
if [[ ! -d "${MODELS_DIR}" ]]; then
    echo "warning: models directory '${MODELS_DIR}' does not exist." >&2
fi

# sync-models.py writes the base preset plus per-model MTP wiring here; fall
# back to the base preset if a sync was skipped and nothing was generated yet.
PRESET="${SCRIPT_DIR}/.generated/router.ini"
if [[ ! -f "${PRESET}" ]]; then
    PRESET="${BASE_INI}"
fi

# Router mode = launch with no -m/-hf model. The router loads models on demand
# and unloads them after the idle timeout below.
exec "${LLAMA_SERVER_BIN}" \
    --host 127.0.0.1 \
    --models-dir "${MODELS_DIR}" \
    --models-preset "${PRESET}" \
    --sleep-idle-seconds 1200 \
    "$@"
