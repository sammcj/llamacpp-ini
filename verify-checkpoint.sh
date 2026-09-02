#!/usr/bin/env bash
set -euo pipefail

# Does the extra context checkpoint change what the model produces?
#
# The checkpoint offset patch makes the server restore from a nearer snapshot on a
# cached follow-up turn. Restoring recurrent state is supposed to be exact, so a
# greedy continuation reached through the cache must match one computed from scratch.
# If it does not, the snapshot is lossy and the speedup is not free.
#
# Greedy (temp 0), fixed seed. Arm A primes the cache then extends it; arm B asks the
# same extended prompt against a cold slot with cache_prompt=false.
#
# Usage: SERVER_BIN=<path> ./verify-checkpoint.sh

BIN="${SERVER_BIN:?set SERVER_BIN}"
MODEL="${MTP_MODEL:-${HOME}/git/sammcj/llamacpp-ini/models/Qwen3.8-Flash-Next-MTP-Merged-GGUF/Qwen3.8-Flash-Next-MTP-UD-IQ4_XS-00001-of-00004.gguf}"
TEMPLATE="${HOME}/git/Qwen-Fixed-Chat-Templates/chat_template.jinja"
PORT="${PORT:-8953}"
NGEN="${NGEN:-192}"
LOG=/private/tmp/claude-501/verify-ckpt.log

BASE="$(cat /private/tmp/claude-501/pp-prompt.txt)"
EXT="${BASE} Follow-up question number 1, please continue."

SRV=""
# shellcheck disable=SC2329  # invoked by the EXIT trap below
cleanup() { [[ -n "${SRV}" ]] && kill "${SRV}" 2>/dev/null || true; }
trap cleanup EXIT

"${BIN}" -m "${MODEL}" --host 127.0.0.1 --port "${PORT}" \
  -ctk q8_0 -ctv q8_0 -ngl 999 --ctx-size 65536 -ub 2048 -b 2048 \
  --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.7 \
  --spec-draft-backend-sampling \
  --chat-template-file "${TEMPLATE}" > "${LOG}" 2>&1 &
SRV=$!

waited=0
until grep -q 'listening on' "${LOG}" 2>/dev/null; do
  sleep 3; waited=$((waited + 3))
  if ! kill -0 "${SRV}" 2>/dev/null || [[ ${waited} -gt 2400 ]]; then
    echo "server failed to start"; tail -5 "${LOG}"; exit 1
  fi
done

ask() {
  curl -sS -m 1800 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg p "$1" --argjson n "${NGEN}" --argjson c "$2" \
          '{messages:[{role:"user",content:$p}],max_tokens:$n,temperature:0,top_k:1,seed:1234,cache_prompt:$c}')"
}

# prime, so the extend below goes through the checkpoint restore path
ask "${BASE}" true > /dev/null
a="$(ask "${EXT}" true)"
b="$(ask "${EXT}" false)"

ta="$(jq -r '(.choices[0].message.reasoning_content // "") + (.choices[0].message.content // "")' <<< "${a}")"
tb="$(jq -r '(.choices[0].message.reasoning_content // "") + (.choices[0].message.content // "")' <<< "${b}")"

printf 'cached   prompt_n=%s  bytes=%s\n' "$(jq -r '.timings.prompt_n' <<< "${a}")" "${#ta}"
printf 'uncached prompt_n=%s  bytes=%s\n' "$(jq -r '.timings.prompt_n' <<< "${b}")" "${#tb}"

if [[ "${ta}" == "${tb}" ]]; then
  echo "IDENTICAL - checkpoint restore is exact"
else
  echo "DIFFERS - first divergence:"
  cmp <(printf '%s' "${ta}") <(printf '%s' "${tb}") 2>&1 | head -2
  printf '  cached  : %.140s\n' "${ta}"
  printf '  uncached: %.140s\n' "${tb}"
fi

rss_kb="$(ps -o rss= -p "${SRV}" 2>/dev/null | tr -d ' ')"
printf 'steady RSS: %s MB\n' "$(( ${rss_kb:-0} / 1024 ))"

kill "${SRV}" 2>/dev/null || true
wait "${SRV}" 2>/dev/null || true
SRV=""
