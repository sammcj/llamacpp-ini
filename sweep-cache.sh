#!/usr/bin/env bash
set -euo pipefail

# Does prompt-cache continuation actually work on this arch?
#
# Every benchmark in QWEN_NEXT.md so far ran cache_prompt=false, which is the worst case
# and not what agent traffic looks like. Agents send a growing conversation: turn N+1 is
# turn N's prompt plus a suffix, with no divergence in the middle. That needs no KV shift,
# so the documented "cache_reuse is not supported by this context" note does not settle it -
# that covers mid-prompt divergence. Continuation is the ctx-checkpoints path.
#
# It matters more than anything on the decode side: prefill is roughly half of a fresh
# request here. If continuation works, an agent turn drops that half to near zero. If it
# does not, every turn re-prefills the whole conversation and the ini should say so.
#
# Reads timings.prompt_n from the response - the count of tokens actually prefilled.
# A cache hit shows prompt_n collapsing to about the suffix length.
#
# WARNING: the "extend" rows here are NOT representative of agent traffic, and reading
# them as such produced a wrong conclusion once already (see QWEN_NEXT.md). They extend a
# conversation whose previous request was an exact repeat, and a repeat prefills only 4
# tokens, so it leaves no checkpoint near the end and the extend falls back to a far older
# one - reporting an n_ubatch-sized replay that real turns never pay. Use sweep-chat.sh
# for the steady-state per-turn cost. This script is still the right one for isolating the
# checkpoint-fallback path itself.
#
# Usage: ./sweep-cache.sh [outfile]

BIN="${LLAMA_SERVER_BIN:-${HOME}/git/llama.cpp-pr27836/build/bin/llama-server}"
MODEL="${MTP_MODEL:-${HOME}/git/sammcj/llamacpp-ini/models/Qwen3.8-Flash-Next-MTP-Merged-GGUF/Qwen3.8-Flash-Next-MTP-UD-IQ4_XS-00001-of-00004.gguf}"
TEMPLATE="${HOME}/git/Qwen-Fixed-Chat-Templates/chat_template.jinja"
OUT="${1:-/private/tmp/claude-501/cache.txt}"
LOG="${OUT%.txt}.log"
PORT="${PORT:-8924}"
NGEN="${NGEN:-32}"

BASE="$(cat "${PROMPT_FILE:-/private/tmp/claude-501/pp-prompt.txt}")"

SRV=""
# shellcheck disable=SC2329  # invoked by the EXIT trap below
cleanup() { [[ -n "${SRV}" ]] && kill "${SRV}" 2>/dev/null || true; }
trap cleanup EXIT

"${BIN}" -m "${MODEL}" --host 127.0.0.1 --port "${PORT}" \
  -ctk q8_0 -ctv q8_0 -ngl 999 --ctx-size 65536 -ub "${UB:-2048}" -b 2048 \
  --ctx-checkpoints "${CKPT:-32}" --checkpoint-min-step "${MINSTEP:-8192}" \
  --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.7 \
  --spec-draft-backend-sampling \
  --temp 1.0 --top-k 20 --min-p 0.0 \
  --chat-template-file "${TEMPLATE}" > "${LOG}" 2>&1 &
SRV=$!

waited=0
until grep -q 'listening on' "${LOG}" 2>/dev/null; do
  sleep 3; waited=$((waited + 3))
  if ! kill -0 "${SRV}" 2>/dev/null || [[ ${waited} -gt 2400 ]]; then
    echo "server failed to start"; tail -5 "${LOG}"; exit 1
  fi
done

{
  echo "prompt-cache continuation $(date '+%Y-%m-%d %H:%M')  ctx-checkpoints=${CKPT:-32} checkpoint-min-step=${MINSTEP:-8192}"
  "${BIN}" --version 2>&1 | head -1
  grep -iE 'cache_reuse|checkpoint' "${LOG}" | head -5 || true
  echo
  printf '%-34s %10s %12s %10s\n' request prompt_n prompt_ms predicted_n
} | tee "${OUT}"

# turn 1 primes the cache; turns 2 and 3 extend it by a short suffix each, which is
# exactly the shape of an agent appending a tool result and asking again.
ask() {
  local label="$1" content="$2" cache="$3"
  local r
  r="$(curl -sS -m 1800 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(jq -n --arg p "${content}" --argjson n "${NGEN}" --argjson c "${cache}" \
              '{messages:[{role:"user",content:$p}],max_tokens:$n,seed:1234,cache_prompt:$c,timings_per_token:false}')")"
  printf '%-34s %10s %12s %10s\n' "${label}" \
    "$(jq -r '.timings.prompt_n // "?"' <<< "${r}")" \
    "$(jq -r '.timings.prompt_ms // "?" | if type=="number" then (.*10|round/10) else . end' <<< "${r}")" \
    "$(jq -r '.timings.predicted_n // "?"' <<< "${r}")" | tee -a "${OUT}"
}

ask "1 base           (cold)"        "${BASE}"                         true
ask "2 base           (repeat)"      "${BASE}"                         true
ask "3 base + suffix  (extend)"      "${BASE} Also, summarise that."   true
ask "4 base + suffix2 (extend)"      "${BASE} Also, summarise that. And list three risks." true
ask "5 base           (cache off)"   "${BASE}"                         false

kill "${SRV}" 2>/dev/null || true
wait "${SRV}" 2>/dev/null || true
SRV=""

{
  echo
  echo "interpretation: rows 2-4 with prompt_n collapsing to roughly the suffix length"
  echo "means continuation works. prompt_n staying at the row-5 value means every agent"
  echo "turn re-prefills the whole conversation."
} | tee -a "${OUT}"

echo "done -> ${OUT}  (server log at ${LOG})"
