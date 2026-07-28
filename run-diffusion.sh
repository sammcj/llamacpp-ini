#!/usr/bin/env bash
#
# Run a diffusion-arch model (diffusion-gemma, dream, llada, ...) with the M5 Max
# defaults. These models cannot load in llama-server / the router - they only run
# one-shot under llama-diffusion-cli - so this is a separate launcher.
#
# Usage:
#   ./run-diffusion.sh <model> -p "your prompt" [extra llama-diffusion-cli args]
#   ./run-diffusion.sh <model> -f prompt.txt
#   ./run-diffusion.sh --list                 list known diffusion models
#
# <model> is either a path to a .gguf, or a name fragment matched against the
# diffusion models sync-models.py discovered (.generated/diffusion-models.tsv).
#
# Arch support is whatever your llama.cpp build has. Mainline currently covers
# dream / llada / llada-moe. The diffusion-gemma arch (unsloth diffusiongemma)
# is NOT in mainline - it lives in PR #24427; build that branch to run those, or
# the CLI aborts with "unknown model architecture: 'diffusion-gemma'".
#
# Env overrides:
#   LLAMA_DIFFUSION_BIN   diffusion binary  (default: llama-diffusion-cli on PATH)
#   MODELS_SRC            source tree to scan if the tsv is missing
#                         (default: ~/.lmstudio/models)
#
# Only the M5 Max defaults that transfer to a one-shot CLI are forced here:
# threads=16 and all layers on the GPU. 128K ctx, q8_0 KV and MTP are router
# concerns that do not apply to diffusion decoding; pass your own flags to tune
# --diffusion-steps, --ctx-size, sampling, etc. (your flags win - they come last).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIFFUSION_TSV="${SCRIPT_DIR}/.generated/diffusion-models.tsv"
SRC="${MODELS_SRC:-${HOME}/.lmstudio/models}"

BIN="${LLAMA_DIFFUSION_BIN:-llama-diffusion-cli}"
if ! command -v "${BIN}" >/dev/null 2>&1; then
    echo "error: '${BIN}' not found on PATH; set LLAMA_DIFFUSION_BIN to its full path." >&2
    exit 1
fi

# list_models -> print "name<TAB>path" for every known diffusion model. Uses the
# tsv sync wrote; falls back to scanning the source tree by arch via llama-gguf.
list_models() {
    if [[ -s "${DIFFUSION_TSV}" ]]; then
        cat "${DIFFUSION_TSV}"
        return
    fi
    command -v llama-gguf >/dev/null 2>&1 || return 0
    while IFS= read -r -d '' g; do
        local arch
        arch="$(llama-gguf "${g}" r n 2>/dev/null | grep -oE '[a-z0-9_-]+\.block_count' | head -1 || true)"
        arch="${arch%.block_count}"
        [[ "${arch}" =~ ^(diffusion|dream|llada) ]] && printf '%s\t%s\n' "$(basename "${g}" .gguf)" "${g}"
    done < <(find "${SRC}" -type f -name '*.gguf' -print0)
}

if [[ "${1:-}" == "--list" || "${1:-}" == "-l" ]]; then
    models="$(list_models)"
    if [[ -z "${models}" ]]; then
        echo "No diffusion models found. Run ./sync-models.py (or check ${SRC})." >&2
        exit 1
    fi
    echo "Diffusion models (name -> path):"
    printf '%s\n' "${models}" | awk -F'\t' '{printf "  %-45s %s\n", $1, $2}'
    exit 0
fi

if [[ $# -lt 1 ]]; then
    echo "usage: $(basename "$0") <model> -p \"prompt\" [args...]   (or --list)" >&2
    exit 1
fi

MODEL_ARG="$1"; shift

# Resolve the model: an existing .gguf path is used as-is; otherwise treat it as
# a name fragment and match it against the known diffusion models.
if [[ -f "${MODEL_ARG}" ]]; then
    MODEL="${MODEL_ARG}"
else
    matches="$(list_models | awk -F'\t' -v q="${MODEL_ARG}" 'tolower($1) ~ tolower(q) || tolower($2) ~ tolower(q)')"
    count="$(printf '%s' "${matches}" | grep -c . || true)"
    if (( count == 0 )); then
        echo "error: no diffusion model matches '${MODEL_ARG}'. Try: $(basename "$0") --list" >&2
        exit 1
    elif (( count > 1 )); then
        echo "error: '${MODEL_ARG}' is ambiguous - matches ${count} models:" >&2
        printf '%s\n' "${matches}" | awk -F'\t' '{printf "  %s\n", $1}' >&2
        exit 1
    fi
    MODEL="$(printf '%s' "${matches}" | cut -f2)"
fi

# Require a prompt - the CLI is one-shot and has nothing to do without one.
has_prompt=0
for a in "$@"; do
    case "${a}" in -p|--prompt|-f|--file|-bf|--binary-file) has_prompt=1; break ;; esac
done
if (( ! has_prompt )); then
    echo "error: no prompt given. Add -p \"your prompt\" (or -f prompt.txt)." >&2
    exit 1
fi

echo "diffusion model: ${MODEL}" >&2

# Defaults first, user args last so they override.
exec "${BIN}" \
    -m "${MODEL}" \
    --threads 16 \
    --n-gpu-layers 999 \
    "$@"
