#!/usr/bin/env bash
set -euo pipefail

# Cold-prefill benchmark. Measures the number that dominates wall-clock when an agent
# opens a fresh session against a large repo context: prompt eval time for a ~30k-token
# prompt that the server has never seen.
#
# Three distinct prompts are sent to one server instance. Each is unrelated to the last,
# so every one is a genuine cold prefill and no slot restart is needed between them.
#
# Server args mirror the production ini (131072 ctx, q8_0 KV, ubatch 2048) so the numbers
# can be read against the live server log rather than only against each other. Do not add
# an explicit --parallel: it forces kv_unified = false, which divides ctx-size by the slot
# count and rejects a 35k-token prompt at 32768. The ini leaves it at the -1 auto default,
# which is 4 slots sharing one unified KV, each seeing the full context.
#
# The build under test is whatever is in build/bin. Rebuild IN PLACE between arms - the
# binary carries an absolute LC_RPATH, so a copied llama-server still loads the live
# dylibs and both arms measure the same code. See QWEN_NEXT.md.
#
# Usage: ./bench-prefill.sh <label>

LABEL="${1:?usage: bench-prefill.sh <label>}"
BIN="${SERVER_BIN:-/Users/samm/git/llama.cpp-pr27836/build/bin/llama-server}"
MODEL="${MTP_MODEL:-${HOME}/git/sammcj/llamacpp-ini/models/Qwen3.8-Flash-Next-MTP-Merged-GGUF/Qwen3.8-Flash-Next-MTP-UD-IQ4_XS-00001-of-00004.gguf}"
TEMPLATE="${HOME}/git/Qwen-Fixed-Chat-Templates/chat_template.jinja"
PROMPT_DIR="${PROMPT_DIR:-/tmp/claude/pf}"
OUT="/tmp/claude/prefill-${LABEL}.txt"
LOG="/tmp/claude/prefill-${LABEL}.log"
PORT="${PORT:-8971}"

SRV=""
# shellcheck disable=SC2329  # invoked by the EXIT trap below
cleanup() { [[ -n "${SRV}" ]] && kill "${SRV}" 2>/dev/null || true; }
trap cleanup EXIT

# EXTRA_ARGS below is intentionally unquoted: callers pass several server flags in one
# string, e.g. EXTRA_ARGS="--override-kv qwen4exp.expert_used_count=int:1".
# shellcheck disable=SC2086

"${BIN}" -m "${MODEL}" --host 127.0.0.1 --port "${PORT}" \
  -ctk q8_0 -ctv q8_0 -ngl 999 --ctx-size 131072 --cache-ram 16384 \
  -ub "${UB:-2048}" -b "${UB:-2048}" \
  --spec-type draft-mtp --spec-draft-n-max 6 --draft-p-min 0.7 \
  --spec-draft-backend-sampling \
  --temp 1.0 --top-k 20 --min-p 0.0 \
  --reasoning-format deepseek --reasoning-preserve \
  ${EXTRA_ARGS:-} \
  --chat-template-file "${TEMPLATE}" > "${LOG}" 2>&1 &
SRV=$!

waited=0
until grep -q 'listening on' "${LOG}" 2>/dev/null; do
  sleep 3; waited=$((waited + 3))
  if ! kill -0 "${SRV}" 2>/dev/null || [[ ${waited} -gt 2400 ]]; then
    echo "server failed to start"; tail -20 "${LOG}"; exit 1
  fi
done

{
  echo "cold prefill  ${LABEL}  $(date '+%Y-%m-%d %H:%M')  bin=${BIN}"
  echo "commit: $(git -C "$(dirname "$(dirname "$(dirname "${BIN}")")")" log --oneline -1)"
  printf '%-6s %10s %12s %12s\n' prompt tokens ms tok_per_s
} | tee "${OUT}"

for p in "${PROMPT_DIR}"/*.txt; do
  # max_tokens 1: generation is not what is being measured, and a long answer would
  # only add noise from the draft/verify path on top of the prefill number.
  r="$(curl -sS -m 1800 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(jq -n --rawfile p "${p}" \
              '{messages:[{role:"user",content:$p}],max_tokens:1,seed:1234,cache_prompt:true}')")"
  printf '%-6s %10s %12s %12s\n' "$(basename "${p}" .txt)" \
    "$(jq -r '.timings.prompt_n // "?"' <<< "${r}")" \
    "$(jq -r '.timings.prompt_ms // "?" | if type=="number" then (.*10|round/10) else . end' <<< "${r}")" \
    "$(jq -r '.timings.prompt_per_second // "?" | if type=="number" then (.*10|round/10) else . end' <<< "${r}")" \
    | tee -a "${OUT}"
done

kill "${SRV}" 2>/dev/null || true
wait "${SRV}" 2>/dev/null || true
SRV=""

awk 'NR>3 && $2 ~ /^[0-9]+$/ {tok+=$2; ms+=$3} END {if (ms) printf "total: %d tokens / %.0f ms = %.1f tok/s\n", tok, ms, tok*1000/ms}' \
  "${OUT}" | tee -a "${OUT}"

echo "done -> ${OUT}"
