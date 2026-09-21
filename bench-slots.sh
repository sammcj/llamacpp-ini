#!/usr/bin/env bash
set -euo pipefail

# Multi-slot checks for two carried PRs. Both only show up with more than one slot live
# on a unified KV cache, which is how samm-mbp.ini serves this model (parallel auto = 4
# slots, kv_unified) and which no other bench here exercises.
#
# A. #29166 (QSA per-block bias with several sequences). Two requests run concurrently
#    on different slots, each with its own ~7k-token prefix and its own codeword in the
#    last user turn. Before the fix a block could read another sequence's bias and the
#    model lost its latest messages; the reply must contain that request's codeword.
#
# B. #28992 (prompt-cache lookup skipped). Three requests pinned to one slot: a long
#    prompt P, then a short prefix of P, then P again. After the second request the slot
#    holds the short prefix (f_keep = 1.0 against P) and the cache holds all of P.
#    Before the fix the third request never consulted the cache and re-prefilled all of
#    P; with it, the server logs "found better prompt" and prompt_n is a few tokens.
#
# Usage: ./bench-slots.sh <label>

LABEL="${1:?usage: bench-slots.sh <label>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Same per-host env the launcher sources, so the server here runs with the served
# kill switches (LLAMA_QSA_NO_POOLED_CACHE for #28699). Without it the pooled cache
# aborts at the concurrent step, which is the bug the variable exists for.
HOST_ENV="${SCRIPT_DIR}/$(hostname -s).env"
if [[ -f "${HOST_ENV}" ]]; then
  # shellcheck source=/dev/null
  source "${HOST_ENV}"
fi

BIN="${SERVER_BIN:-${LLAMA_SERVER_BIN:-/Users/samm/git/llama.cpp-pr27836/build/bin/llama-server}}"
MODEL="${MTP_MODEL:-${HOME}/git/sammcj/llamacpp-ini/models/Qwen3.8-Flash-Next-MTP-Merged-GGUF/Qwen3.8-Flash-Next-MTP-UD-IQ4_XS-00001-of-00004.gguf}"
TEMPLATE="${HOME}/git/Qwen-Fixed-Chat-Templates/chat_template.jinja"
OUT="/tmp/claude/slots-${LABEL}.txt"
LOG="/tmp/claude/slots-${LABEL}.log"
PORT="${PORT:-8974}"
KV="${KV:-f16}"
NMAX="${NMAX:-6}"
# entries of synthetic log per prefix; ~24 tokens each, so 300 is ~7k tokens, well past
# the 2048-token QSA budget the per-block bias governs
ENTRIES="${ENTRIES:-300}"

# UNIFIED=0 gives each slot its own KV stream (ctx-size is then per slot), to separate a
# unified-cache bug from a plain multi-slot one. SPEC_TYPE=none takes draft-mtp out of the
# picture (and lets a master build serve the base model). EXTRA_SERVER_ARGS appends to the
# server command line.
UNIFIED="${UNIFIED:-1}"
declare -a KVU=(--kv-unified)
[[ "${UNIFIED}" == "1" ]] || KVU=()
# SPEC_TYPE=none drops speculation entirely (a repeated --spec-type does not override)
SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"
declare -a SPEC=(--spec-type "${SPEC_TYPE}")
[[ "${SPEC_TYPE}" == "none" ]] || SPEC+=(--spec-draft-n-max "${NMAX}" --draft-p-min 0.7 --spec-draft-backend-sampling)
read -r -a EXTRA <<< "${EXTRA_SERVER_ARGS:-}"

command -v jq >/dev/null 2>&1 || { echo "jq required" >&2; exit 1; }

SRV=""
# shellcheck disable=SC2329  # invoked by the EXIT trap below
cleanup() { [[ -n "${SRV}" ]] && kill "${SRV}" 2>/dev/null || true; }
trap cleanup EXIT

"${BIN}" -m "${MODEL}" --host 127.0.0.1 --port "${PORT}" \
  -ctk "${KV}" -ctv "${KV}" -ngl 999 --ctx-size 65536 -ub 2048 -b 2048 \
  --parallel 4 "${KVU[@]}" --cache-ram 16384 "${SPEC[@]}" \
  --reasoning-format deepseek --reasoning-preserve \
  --chat-template-file "${TEMPLATE}" "${EXTRA[@]}" > "${LOG}" 2>&1 &
SRV=$!

waited=0
until grep -q 'listening on' "${LOG}" 2>/dev/null; do
  sleep 3; waited=$((waited + 3))
  if ! kill -0 "${SRV}" 2>/dev/null || [[ ${waited} -gt 2400 ]]; then
    echo "server failed to start"; tail -20 "${LOG}"; exit 1
  fi
done

# Deterministic filler that differs per tag, so two prefixes share no long common prefix.
filler() {
  awk -v tag="${1}" -v n="${ENTRIES}" 'BEGIN {
    for (i = 1; i <= n; i++)
      printf "Entry %d of log %s: sensor %d read %d units at step %d, valve %s, operator note %d.\n",
        i, tag, (i * 13) % 17, (i * 37) % 997, i * 3, (i % 2 ? "open" : "closed"), (i * 53) % 811
  }'
}

# Greedy, thinking off, one turn of context then a question about the last message.
ask() {
  local slot="${1}" body="${2}" file="${3}"
  curl -sS -m 1800 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --argjson m "${body}" --argjson s "${slot}" \
          '{messages:$m,max_tokens:24,temperature:0,seed:1234,id_slot:$s,
            chat_template_kwargs:{enable_thinking:false}}')" > "${file}"
}

{
  echo "slots  ${LABEL}  $(date '+%Y-%m-%d %H:%M')  bin=${BIN##*/}  pooled_cache=$([[ -n "${LLAMA_QSA_NO_POOLED_CACHE:-}" ]] && echo off || echo on)"
  echo
  echo "A. concurrent slots, #29166"
} | tee "${OUT}"

# The codeword sits at the very end of the prompt, so a model that has lost its latest
# tokens cannot answer. The solo run first is the control: if it fails, the probe is
# wrong, not the build.
for tag in A B; do
  code="PELICAN"; [[ "${tag}" == "B" ]] && code="OSPREY"
  jq -n --arg f "$(filler "${tag}")" --arg c "${code}" \
    '[{role:"user",content:($f + "\n\nThe codeword is " + $c + ". Reply with only the codeword.")}]' \
    > "/tmp/claude/slots-${LABEL}-${tag}.json"
done

check() {
  local tag="${1}" file="${2}" label="${3}" reply n code verdict
  code="PELICAN"; [[ "${tag}" == "B" ]] && code="OSPREY"
  reply="$(jq -r '.choices[0].message.content // ""' "${file}")"
  n="$(jq -r '.timings.prompt_n // "-"' "${file}")"
  verdict=FAIL; [[ "${reply}" == *"${code}"* ]] && verdict=ok
  printf '  %-12s prompt_n %6s  want %-8s got %-24s %s\n' "${label}" "${n}" "${code}" "${reply//$'\n'/ }" "${verdict}" | tee -a "${OUT}"
  [[ "${verdict}" == ok ]]
}

pass=0
ask 1 "$(cat "/tmp/claude/slots-${LABEL}-A.json")" "/tmp/claude/slots-${LABEL}-A0.out"
check A "/tmp/claude/slots-${LABEL}-A0.out" "A solo" && pass=$((pass + 1))

ask 1 "$(cat "/tmp/claude/slots-${LABEL}-A.json")" "/tmp/claude/slots-${LABEL}-A.out" &
req_a=$!
ask 2 "$(cat "/tmp/claude/slots-${LABEL}-B.json")" "/tmp/claude/slots-${LABEL}-B.out" &
req_b=$!
# a bare wait would also block on the server started above
wait "${req_a}" "${req_b}"
check A "/tmp/claude/slots-${LABEL}-A.out" "A concurrent" && pass=$((pass + 1))
check B "/tmp/claude/slots-${LABEL}-B.out" "B concurrent" && pass=$((pass + 1))

{
  echo
  echo "B. cache lookup on a slot holding a shorter prefix, #28992 (slot 3)"
} | tee -a "${OUT}"

full="$(filler C)"
short="$(printf '%s\n' "${full}" | head -n 40)"
# jq programs; $f is a jq variable, not a shell one
# shellcheck disable=SC2016
q1='[{role:"user",content:($f + "\n\nHow many entries are in this log? Reply with only the number.")}]'
# shellcheck disable=SC2016
q2='[{role:"user",content:($f + "\n\nWhich valve state appears first? Reply with only the word.")}]'
i=0
for step in "full:${q1}" "short:${q2}" "full:${q1}"; do
  i=$((i + 1))
  which="${step%%:*}"; tmpl="${step#*:}"
  text="${full}"; [[ "${which}" == "short" ]] && text="${short}"
  ask 3 "$(jq -n --arg f "${text}" "${tmpl}")" "/tmp/claude/slots-${LABEL}-r${i}.out"
  n="$(jq -r '.timings.prompt_n // "-"' "/tmp/claude/slots-${LABEL}-r${i}.out")"
  ms="$(jq -r '.timings.prompt_ms // "-" | tostring | .[0:7]' "/tmp/claude/slots-${LABEL}-r${i}.out")"
  printf '  r%d %-6s prompt_n %6s  prompt_ms %8s\n' "${i}" "${which}" "${n}" "${ms}" | tee -a "${OUT}"
done

kill "${SRV}" 2>/dev/null || true
wait "${SRV}" 2>/dev/null || true
SRV=""

{
  echo
  echo "A: ${pass}/3 codewords correct. B: r3 prompt_n should be tens of tokens, not r1's."
} | tee -a "${OUT}"
echo "done -> ${OUT}"
