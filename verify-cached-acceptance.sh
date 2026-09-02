#!/usr/bin/env bash
set -euo pipefail

# Is the acceptance drop on the cached path real, or is it content divergence?
#
# sweep-chat.sh lets each arm generate its own conversation, so by turn 2 the cached and
# uncached arms are answering different text - and acceptance on this model depends
# heavily on what is being generated (tool-call JSON drafts far better than prose). A
# 0.65 vs 0.70 gap measured that way is not attributable.
#
# This pins the content. One server, one fixed message history H, three requests per rep:
#
#   1. prime    - H minus the last user turn, cache_prompt=true, fills the slot
#   2. cached   - H, cache_prompt=true  -> short prefill, restores from checkpoint
#   3. uncached - H, cache_prompt=false -> full re-prefill of the identical history
#
# Sending identical bytes is necessary but NOT sufficient for the arms to be comparable,
# which an earlier version of this script got wrong. The cached arm prefills ~42 tokens in
# one ubatch and the uncached arm ~6000; different reduction orders give slightly
# different logits, and at temp 1.0 / top-k 20 that can diverge the sampled text. Content
# divergence is the exact confound this script exists to remove, so it has to be
# *observed*, not assumed. Hence the sha column: if the two arms' hashes match, the
# comparison is clean; if they differ, the run is comparing two different generations and
# any acceptance delta is uninterpretable.
#
# Three further controls, each of which was missing and each of which mattered:
#   - ignore_eos, so both arms generate exactly NGEN tokens. Without it EOG ends the arms
#     at different lengths and their acceptance means cover different amounts of text.
#   - arm order alternates by rep parity. With a fixed order, thermal drift over the run
#     loads entirely onto the arm axis and reads as a cache effect.
#   - predicted_n is reported, so a length mismatch is visible rather than inferred.
#
# Usage: SERVER_BIN=<path> ./verify-cached-acceptance.sh [outfile]

BIN="${SERVER_BIN:?set SERVER_BIN to the llama-server to test}"
MODEL="${MTP_MODEL:-${HOME}/git/sammcj/llamacpp-ini/models/Qwen3.8-Flash-Next-MTP-Merged-GGUF/Qwen3.8-Flash-Next-MTP-UD-IQ4_XS-00001-of-00004.gguf}"
TEMPLATE="${HOME}/git/Qwen-Fixed-Chat-Templates/chat_template.jinja"
OUT="${1:-/tmp/claude/cached-acceptance.txt}"
LOG="${OUT%.txt}.log"
PORT="${PORT:-8991}"
NGEN="${NGEN:-1024}"
REPEATS="${REPEATS:-4}"
# TEMP=0 makes the target greedy, so the two arms only diverge if the cache path actually
# flips an argmax rather than merely nudging a logit under sampling. That is the run that
# can attribute a delta to the checkpoint-replay path; temp 1.0 cannot, because the arms
# prefill different batch shapes and the sampled text drifts apart on its own.
TEMP="${TEMP:-1.0}"

BASE="$(cat "${PROMPT_FILE:-/private/tmp/claude-501/pp-prompt.txt}")"
mkdir -p "$(dirname "${OUT}")"

SRV=""
# shellcheck disable=SC2329  # invoked by the EXIT trap below
cleanup() { [[ -n "${SRV}" ]] && kill "${SRV}" 2>/dev/null || true; }
trap cleanup EXIT

# f16 KV to match what this model is now served with (samm-mbp.ini).
"${BIN}" -m "${MODEL}" --host 127.0.0.1 --port "${PORT}" \
  -ctk f16 -ctv f16 -ngl 999 --ctx-size 65536 -ub 2048 -b 2048 \
  --ctx-checkpoints 32 \
  --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.7 \
  --spec-draft-backend-sampling \
  --temp "${TEMP}" --top-k 20 --min-p 0.0 \
  --reasoning-format deepseek --reasoning-preserve \
  --chat-template-file "${TEMPLATE}" > "${LOG}" 2>&1 &
SRV=$!

waited=0
until grep -q 'listening on' "${LOG}" 2>/dev/null; do
  sleep 3; waited=$((waited + 3))
  if ! kill -0 "${SRV}" 2>/dev/null || [[ ${waited} -gt 2400 ]]; then
    echo "server failed to start"; tail -20 "${LOG}"; exit 1
  fi
done

# ask <messages-json> <cache_prompt> -> "prompt_n prompt_ms predicted_n accept tg sha"
ask() {
  local msgs="$1" cache="$2" mark new r
  mark="$(wc -l < "${LOG}")"
  r="$(curl -sS -m 1800 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(jq -n --argjson m "${msgs}" --argjson n "${NGEN}" --argjson c "${cache}" \
              '{messages:$m,max_tokens:$n,seed:1234,cache_prompt:$c,ignore_eos:true}')")" || return 1
  new="$(tail -n +$((mark + 1)) "${LOG}")"
  # Hash content and reasoning together: this model splits its answer across both, and
  # hashing only content would call two different generations identical whenever they
  # differ solely in the thinking block.
  printf '%s %s %s %s %s %s\n' \
    "$(jq -r '.timings.prompt_n // "?"' <<< "${r}")" \
    "$(jq -r '.timings.prompt_ms // "?" | if type=="number" then (.*10|round/10) else . end' <<< "${r}")" \
    "$(jq -r '.timings.predicted_n // "?"' <<< "${r}")" \
    "$(grep -oE 'draft acceptance = [0-9.]+' <<< "${new}" | tail -1 | grep -oE '[0-9.]+$' || echo '-')" \
    "$(jq -r '.timings.predicted_per_second // "?" | if type=="number" then (.*100|round/100) else . end' <<< "${r}")" \
    "$(jq -r '(.choices[0].message.reasoning_content // "") + (.choices[0].message.content // "")' <<< "${r}" \
       | shasum | cut -c1-8)"
}

{
  echo "cached vs uncached acceptance, identical history  $(date '+%Y-%m-%d %H:%M')"
  echo "bin=${BIN}  ngen=${NGEN} (ignore_eos)  repeats=${REPEATS}  temp=${TEMP}"
  printf '%-6s %-10s %10s %12s %12s %10s %10s %10s\n' \
    rep arm prompt_n prompt_ms predicted_n accept tg_t/s sha
} | tee "${OUT}"

prefix_msgs="$(jq -n --arg p "${BASE}" '[{role:"user",content:$p}]')"
full_msgs="$(jq -n --arg p "${BASE}" \
  '[{role:"user",content:$p},
    {role:"assistant",content:"Understood. I have read the material above."},
    {role:"user",content:"Now summarise the three most important points and explain why each matters."}]')"

emit() {
  # shellcheck disable=SC2086  # word splitting is the point: ask() returns six fields
  printf '%-6s %-10s %10s %12s %12s %10s %10s %10s\n' "$1" "$2" $3 | tee -a "${OUT}"
}

for ((i = 1; i <= REPEATS; i++)); do
  # Prime: fills the slot and lays down a checkpoint near the end of the prefix. Runs
  # before both arms every rep, so the cached arm always has a slot to restore from.
  ask "${prefix_msgs}" true > /dev/null || { echo "prime failed on rep ${i}" >&2; exit 1; }

  if (( i % 2 == 1 )); then
    c="$(ask "${full_msgs}" true)"  || { echo "cached failed on rep ${i}"   >&2; exit 1; }
    u="$(ask "${full_msgs}" false)" || { echo "uncached failed on rep ${i}" >&2; exit 1; }
  else
    # Even reps run uncached first. A cache_prompt=false request re-prefills the whole
    # prompt, which also re-primes the slot, so the cached arm after it still hits.
    u="$(ask "${full_msgs}" false)" || { echo "uncached failed on rep ${i}" >&2; exit 1; }
    c="$(ask "${full_msgs}" true)"  || { echo "cached failed on rep ${i}"   >&2; exit 1; }
  fi

  emit "${i}" cached   "${c}"
  emit "${i}" uncached "${u}"
done

kill "${SRV}" 2>/dev/null || true
wait "${SRV}" 2>/dev/null || true
SRV=""

# Guard every numeric field on its own: a "?" in one column must not silently contribute
# a zero to another column's mean.
awk 'NR>3 && ($2 == "cached" || $2 == "uncached") {
       arm = $2
       if ($6 ~ /^[0-9.]+$/) { a[arm] += $6; an[arm]++
                               if (amin[arm] == "" || $6 < amin[arm]) amin[arm] = $6
                               if (amax[arm] == "" || $6 > amax[arm]) amax[arm] = $6 }
       if ($7 ~ /^[0-9.]+$/) { g[arm] += $7; gn[arm]++ }
       sha[arm "" NR] = $8
       shas[arm] = shas[arm] " " $8
     }
     END {
       for (arm in an)
         if (an[arm])
           printf "%-8s acceptance mean %.4f  min %.4f  max %.4f  (n=%d), tg mean %.2f t/s\n",
                  arm, a[arm]/an[arm], amin[arm], amax[arm], an[arm], (gn[arm] ? g[arm]/gn[arm] : 0)
       if (an["cached"] && an["uncached"])
         printf "delta: acceptance %+.4f, tg %+.2f t/s (cached minus uncached)\n",
                a["cached"]/an["cached"] - a["uncached"]/an["uncached"],
                g["cached"]/gn["cached"] - g["uncached"]/gn["uncached"]
       print "cached   sha:" shas["cached"]
       print "uncached sha:" shas["uncached"]
       print "If the two sha lists do not match pairwise, the arms generated different text"
       print "and the acceptance delta is NOT attributable to the cache path."
     }' "${OUT}" | tee -a "${OUT}"

echo "done -> ${OUT}  (server log at ${LOG})"
