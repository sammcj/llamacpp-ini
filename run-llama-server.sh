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
#
# The idle sleep used to throw away the prompt cache: handle_sleeping_state() calls
# destroy(), and load_model() then rebuilds prompt_cache with make_unique, so waking
# left the next agent prompt fully cold - a 28903-token prompt back at ~38 s.
#
# --sleep-preserve-cache (PR #28022) writes the idle slot states into the prompt cache
# before sleeping and restores them on wake, taking that to 0.1 s of prefill and 3.2 s
# of wall time, the wall being the weight reload. Off by default upstream, so it has to
# be passed explicitly. Figures and method in QWEN_NEXT.md.
#
# With the cache surviving, sleeping costs only that reload, so the idle window no
# longer has to span a working day. An hour frees the ~94 GB over lunch and overnight
# while keeping the reload rare. LLAMA_SLEEP_IDLE=-1 never sleeps.
#
# Note the interaction with the per-model --cache-disk that sync-models.py sets: each
# sleep runs every idle slot through prompt_save, so a shorter idle window means more
# writes - up to ~900 MiB per slot per sleep, several times a day.
SLEEP_IDLE="${LLAMA_SLEEP_IDLE:-3600}"

# A restart loses the cache outright, since it only ever lived in RAM. --cache-disk
# (PR #28092) fixes that, but it is set per model by sync-models.py rather than here:
# the router builds its child presets from its own argv, so a path given here would be
# handed to every child, and a child deletes every entry in its cache directory whose
# key is not its own. One shared path means loading a second model wipes the first
# model's cache. See cache_disk_overrides() in sync-models.py.
#
# #28022 is not upstream yet and an unknown flag is fatal, so only pass it to a binary
# that advertises it. Drop this probe once it lands in master.
SLEEP_ARGS=(--sleep-idle-seconds "${SLEEP_IDLE}")
if "${LLAMA_SERVER_BIN}" --help 2>&1 | grep -q -- '--sleep-preserve-cache'; then
    SLEEP_ARGS+=(--sleep-preserve-cache)
else
    echo "note: ${LLAMA_SERVER_BIN} has no --sleep-preserve-cache; waking will be cold." >&2
fi

exec "${LLAMA_SERVER_BIN}" \
    --host 127.0.0.1 \
    --models-dir "${MODELS_DIR}" \
    --models-preset "${PRESET}" \
    "${SLEEP_ARGS[@]}" \
    "$@"
