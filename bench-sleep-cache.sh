#!/usr/bin/env bash
set -euo pipefail

# What survives, and what throws the prompt cache away?
#
# Two events cost a full cold prefill unless something is done about them. The idle sleep:
# handle_sleeping_state() calls destroy(), and load_model() then rebuilds prompt_cache with
# make_unique, so everything cached before the sleep is freed - and the cached KV blobs
# referenced the destroyed context anyway. And a restart, which loses the cache outright
# because it only ever lived in RAM.
#
# Sends the same large prompt four times: cold, warm, after an idle sleep, and after the
# process has been killed and started again. A warm repeat costs a handful of tokens; any
# step back at the cold figure is a case that is not covered.
#
# The two fixes are opt-in flags, so pass them via EXTRA_ARGS to see them work:
#   EXTRA_ARGS="--sleep-preserve-cache"                     covers step 3 (PR #28022)
#   EXTRA_ARGS="--sleep-preserve-cache --cache-disk /path"  covers step 4 too (PR #28092)
#
# SLEEP_IDLE is deliberately small so the test takes a minute rather than the hour the ini uses.
#
# Usage: ./bench-sleep-cache.sh

BIN="${SERVER_BIN:-/Users/samm/git/llama.cpp-pr27836/build/bin/llama-server}"
MODEL="${MTP_MODEL:-${HOME}/git/sammcj/llamacpp-ini/models/Qwen3.8-Flash-Next-MTP-Merged-GGUF/Qwen3.8-Flash-Next-MTP-UD-IQ4_XS-00001-of-00004.gguf}"
TEMPLATE="${HOME}/git/Qwen-Fixed-Chat-Templates/chat_template.jinja"
PROMPT="${PROMPT_FILE:-/tmp/claude/px/a_taskA.txt}"
LOG="${TMPDIR:-/tmp}/sleepcache.log"
PORT="${PORT:-8975}"
SLEEP_IDLE="${SLEEP_IDLE:-30}"
STARTUP=""

[[ -r "${PROMPT}" ]] || { echo "prompt file not readable: ${PROMPT}" >&2; exit 1; }
mkdir -p "$(dirname "${LOG}")"

SRV=""
# shellcheck disable=SC2329  # invoked by the EXIT trap below
cleanup() { [[ -n "${SRV}" ]] && kill "${SRV}" 2>/dev/null || true; }
trap cleanup EXIT

# EXTRA_ARGS is intentionally unquoted so a caller can pass several flags in one string.
# shellcheck disable=SC2086
start_server() {
  # Truncate before forking. The child's own > redirect happens after the parent has
  # already reached the grep below, so on the phase-4 relaunch it would otherwise match
  # the previous run's "listening on" and hand back a server still loading weights.
  : > "${LOG}"

  local t0 t1
  t0="${EPOCHREALTIME/,/.}"

  "${BIN}" -m "${MODEL}" --host 127.0.0.1 --port "${PORT}" \
    -ctk q8_0 -ctv q8_0 -ngl 999 --ctx-size 65536 --cache-ram 16384 \
    -ub 2048 -b 2048 --sleep-idle-seconds "${SLEEP_IDLE}" \
    --spec-type draft-mtp --spec-draft-n-max 6 --draft-p-min 0.7 \
    --spec-draft-backend-sampling \
    --temp 1.0 --top-k 20 --min-p 0.0 \
    ${EXTRA_ARGS:-} \
    --chat-template-file "${TEMPLATE}" > "${LOG}" 2>&1 &
  SRV=$!

  local waited=0
  until grep -q 'listening on' "${LOG}" 2>/dev/null; do
    sleep 3; waited=$((waited + 3))
    if ! kill -0 "${SRV}" 2>/dev/null || [[ ${waited} -gt 2400 ]]; then
      echo "server failed to start"; tail -20 "${LOG}"; exit 1
    fi
  done

  # Reported separately from the request timings: restoring the disk cache happens in
  # server_prompt_cache's constructor, before the port opens, so it lands here and not
  # in the ask() figure below. Without this, --cache-disk looks free at restart.
  t1="${EPOCHREALTIME/,/.}"
  STARTUP="$(awk "BEGIN{printf \"%.1f\", ${t1} - ${t0}}")"
}

stop_server() {
  kill "${SRV}" 2>/dev/null || true
  wait "${SRV}" 2>/dev/null || true
  SRV=""
}

# Wall time matters as much as prompt_ms: waking from sleep reloads ~94 GB of weights, and
# a restart reloads them and reads the cache off disk, neither of which shows up in timings.
ask() {
  local t0 t1 out
  t0="${EPOCHREALTIME/,/.}"
  out="$(curl -sS -m 1800 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --rawfile p "${PROMPT}" '{messages:[{role:"user",content:$p}],max_tokens:1,seed:1234,cache_prompt:true}')" \
  | jq -r '"\(.timings.prompt_n // "?") tokens  \((.timings.prompt_ms // 0) | .*10 | round / 10) ms prefill"')"
  t1="${EPOCHREALTIME/,/.}"
  printf '%s, %s s wall\n' "${out}" "$(awk "BEGIN{printf \"%.1f\", ${t1} - ${t0}}")"
}

# ask() runs in a command substitution, so set -e cannot see it fail - a dead server or
# an unreadable prompt would otherwise print a blank row and look like a fast result.
report() {
  local label="${1}" r
  r="$(ask)" || { echo "request failed during '${label}'" >&2; exit 1; }
  [[ -n "${r}" ]] || { echo "empty response during '${label}'" >&2; exit 1; }
  printf '%-28s %s\n' "${label}" "${r}"
}

start_server
echo "   startup: ${STARTUP} s"
report "1. cold:"
report "2. warm repeat:"

# leave the server idle past its sleep threshold, then confirm it actually slept
echo "   idling ${SLEEP_IDLE}s + margin for the server to sleep..."
slept=0
for _ in $(seq 1 $((SLEEP_IDLE + 40))); do
  sleep 1
  if grep -q 'entering sleeping state' "${LOG}"; then slept=1; break; fi
done

if [[ ${slept} -eq 1 ]]; then
  echo "   server reported entering sleeping state"
else
  echo "   WARNING: server never reported sleeping; result below is not the sleep case"
fi

report "3. after sleep+wake:"

stop_server
echo "   restarting the server..."
start_server
echo "   startup: ${STARTUP} s (includes any disk-cache restore)"

report "4. after restart:"

stop_server
