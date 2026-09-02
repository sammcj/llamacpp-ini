#!/usr/bin/env bash
set -euo pipefail

# The real agent shape: a full message history that grows by an assistant reply (INCLUDING
# its reasoning_content - see below, getting this wrong invalidated a whole round of results) plus a
# new user turn each round.
#
# The obvious wrong way to model this, and the one an earlier harness here used, is to
# grow a single user message and never send the assistant's answers back - then the
# prefix being reused is only ever the user text. Reusing the previous *answer* is the case ggml-org/llama.cpp#28049 is about:
# draft-mtp leaves accepted tokens past the end-of-generation token in the slot, the slot
# stops being a prefix of the next request, and a hybrid model has to fall back to a
# context checkpoint and re-prefill the whole answer.
#
# f_keep in the server log is the metric that actually shows it - 1.000 means the answer
# was kept. The tokens responsible are invisible in anything the API returns.
#
# Usage: SERVER_BIN=<path> ./sweep-chat.sh [outfile]

BIN="${SERVER_BIN:?set SERVER_BIN to the llama-server to test}"
MODEL="${MTP_MODEL:-${HOME}/git/sammcj/llamacpp-ini/models/Qwen3.8-Flash-Next-MTP-Merged-GGUF/Qwen3.8-Flash-Next-MTP-UD-IQ4_XS-00001-of-00004.gguf}"
TEMPLATE="${HOME}/git/Qwen-Fixed-Chat-Templates/chat_template.jinja"
OUT="${1:-/private/tmp/claude-501/chat.txt}"
LOG="${OUT%.txt}.log"
PORT="${PORT:-8961}"
NGEN="${NGEN:-160}"
TURNS="${TURNS:-5}"

# Every acceptance figure in QWEN_NEXT.md was taken with cache_prompt=false, because
# bench-mtp.sh posts it that way to get bit-identical runs. Production serves
# cache_prompt=true, and until now no script that set it parsed acceptance at all - so
# the n-max and p-min choices are tuned on a path the server never actually runs.
#
# CACHE=false runs the same turns uncached, but it is NOT a control arm and this script
# cannot answer the acceptance question: each arm generates its own conversation, so from
# turn 2 they are answering different text, and acceptance here tracks content strongly.
# Measured that way 4 turns and 10 turns gave opposite signs. Use
# verify-cached-acceptance.sh for that. What this script measures correctly is per-turn
# prefill cost, where the arms are directly comparable.
CACHE="${CACHE:-true}"

BASE="$(cat "${PROMPT_FILE:-/private/tmp/claude-501/pp-prompt.txt}")"

# NO_RP=1 drops --reasoning-preserve, to separate the server-side half of the
# round-trip (does the template re-render the thinking block) from the client-side
# half (does the client echo reasoning_content back at all).
rp_opts=(--reasoning-preserve)
[[ -n "${NO_RP:-}" ]] && rp_opts=()

SRV=""
# shellcheck disable=SC2329  # invoked by the EXIT trap below
cleanup() { [[ -n "${SRV}" ]] && kill "${SRV}" 2>/dev/null || true; }
trap cleanup EXIT

"${BIN}" -m "${MODEL}" --host 127.0.0.1 --port "${PORT}" \
  -ctk q8_0 -ctv q8_0 -ngl 999 --ctx-size 65536 -ub "${UB:-2048}" -b 2048 \
  --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.7 \
  --spec-draft-backend-sampling \
  --temp 1.0 --top-k 20 --min-p 0.0 \
  "${rp_opts[@]}" \
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
  echo "chat-history multi-turn $(date '+%Y-%m-%d %H:%M')  bin=${BIN}  cache_prompt=${CACHE}"
  printf '%-8s %10s %12s %10s %10s %10s\n' turn prompt_n prompt_ms f_keep accept tg_t/s
} | tee "${OUT}"

msgs="$(jq -n --arg p "${BASE}" '[{role:"user",content:$p}]')"

for ((t = 1; t <= TURNS; t++)); do
  # Mark the log before the request so acceptance is read from THIS turn's stats block
  # rather than whatever the previous turn left behind - a failed turn would otherwise
  # silently repeat the last good number.
  mark="$(wc -l < "${LOG}")"

  r="$(curl -sS -m 1800 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(jq -n --argjson m "${msgs}" --argjson n "${NGEN}" --argjson c "${CACHE}" \
              '{messages:$m,max_tokens:$n,seed:1234,cache_prompt:$c}')")"

  # This model thinks by default, so the answer arrives split across
  # reasoning_content and content. Reading only content sends an EMPTY assistant
  # turn back, the slot's generated tokens stop being a prefix, and every turn
  # eats a checkpoint restore - which reads as a cache problem when it is a
  # harness bug. Echo both back, as a client with reasoning preserved would.
  reply="$(jq -r '.choices[0].message.content // ""' <<< "${r}")"
  reason="$(jq -r '.choices[0].message.reasoning_content // ""' <<< "${r}")"
  new="$(tail -n +$((mark + 1)) "${LOG}")"
  fk="$(grep -oE 'f_keep = [0-9.]+' <<< "${new}" | tail -1 | grep -oE '[0-9.]+$' || echo '-')"
  acc="$(grep -oE 'draft acceptance = [0-9.]+' <<< "${new}" | tail -1 | grep -oE '[0-9.]+$' || echo '-')"

  printf '%-8s %10s %12s %10s %10s %10s\n' "${t}" \
    "$(jq -r '.timings.prompt_n // "?"' <<< "${r}")" \
    "$(jq -r '.timings.prompt_ms // "?" | if type=="number" then (.*10|round/10) else . end' <<< "${r}")" \
    "${fk}" "${acc}" \
    "$(jq -r '.timings.predicted_per_second // "?" | if type=="number" then (.*100|round/100) else . end' <<< "${r}")" \
    | tee -a "${OUT}"

  # append the assistant turn and a new user turn, exactly as a chat client would
  msgs="$(jq -n --argjson m "${msgs}" --arg a "${reply}" --arg rc "${reason}" \
              --arg u "Thanks. Follow-up ${t}: please continue." \
          '$m + [({role:"assistant",content:$a} + (if $rc == "" then {} else {reasoning_content:$rc} end)),
                 {role:"user",content:$u}]')"
done

kill "${SRV}" 2>/dev/null || true
wait "${SRV}" 2>/dev/null || true
SRV=""

awk 'NR>2 && $1 ~ /^[0-9]+$/ && $1 > 1 {
       s += $3; n++
       if ($5 ~ /^[0-9.]+$/) { a += $5; an++ }
       if ($6 ~ /^[0-9.]+$/) { g += $6; gn++ }
     }
     END {
       if (n)  printf "steady-state mean prompt_ms over %d turns: %.1f\n", n, s/n
       if (an) printf "steady-state mean acceptance over %d turns: %.4f\n", an, a/an
       if (gn) printf "steady-state mean tg t/s over %d turns: %.2f\n", gn, g/gn
     }' "${OUT}" | tee -a "${OUT}"

echo "done -> ${OUT}"
