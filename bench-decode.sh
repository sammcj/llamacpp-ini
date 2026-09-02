#!/usr/bin/env bash
set -euo pipefail

# Decode benchmark, for the acceptance question rather than the prefill one.
#
# Decode speed on this model tracks mean accepted draft length (r = +0.87 over real agent
# traffic), not context depth. The draft is greedy - common/speculative.cpp takes
# cur_p->data[0].id - while the target samples at whatever --temp is set. Verification is
# plain token equality (common/sampling.cpp, `if (draft[i] != id) break;`), so a stochastic
# target only accepts when its draw lands exactly on the draft's argmax.
#
# TEMP=0 makes the target greedy too. If acceptance jumps, the gap is sampling-induced and
# proper rejection sampling has something to recover. If it does not, the draft is simply
# wrong that often and no verification scheme will help.
#
# Usage: TEMP=1.0 ./bench-decode.sh <label>

LABEL="${1:?usage: bench-decode.sh <label>}"
BIN="${SERVER_BIN:-/Users/samm/git/llama.cpp-pr27836/build/bin/llama-server}"
MODEL="${MTP_MODEL:-${HOME}/git/sammcj/llamacpp-ini/models/Qwen3.8-Flash-Next-MTP-Merged-GGUF/Qwen3.8-Flash-Next-MTP-UD-IQ4_XS-00001-of-00004.gguf}"
TEMPLATE="${HOME}/git/Qwen-Fixed-Chat-Templates/chat_template.jinja"
OUT="/tmp/claude/decode-${LABEL}.txt"
LOG="/tmp/claude/decode-${LABEL}.log"
PORT="${PORT:-8973}"
TEMP="${TEMP:-1.0}"
NGEN="${NGEN:-400}"

# Several different prompts: acceptance depends heavily on what is being generated
# (tool-call JSON drafts far better than prose), so a single prompt says little.
PROMPTS=(
  "Write a short Python function that merges two sorted lists, then explain how it works."
  "Explain in a few paragraphs why unified memory helps large language model inference on Apple silicon."
  "List the steps to set up a systemd service unit for a long-running Go binary, with a brief note on each."
)

# PREFIX_FILE prepends a document to every prompt, so the same three generations run at
# agent-like depth instead of from an empty context. Anything whose cost scales with KV
# length - QSA attention, mask uploads - is invisible without it.
PREFIX=""
CTX="${CTX:-32768}"
if [[ -n "${PREFIX_FILE:-}" ]]; then
  [[ -r "${PREFIX_FILE}" ]] || { echo "prefix file not readable: ${PREFIX_FILE}" >&2; exit 1; }
  PREFIX="$(cat "${PREFIX_FILE}")"
  # A ~30k prefix plus NGEN would sit right on the 32k default, and an overflow
  # triggers a context shift that quietly changes what is being measured.
  CTX="${CTX_PREFIX:-65536}"
fi

SRV=""
# shellcheck disable=SC2329  # invoked by the EXIT trap below
cleanup() { [[ -n "${SRV}" ]] && kill "${SRV}" 2>/dev/null || true; }
trap cleanup EXIT

"${BIN}" -m "${MODEL}" --host 127.0.0.1 --port "${PORT}" \
  -ctk q8_0 -ctv q8_0 -ngl 999 --ctx-size "${CTX}" -ub 2048 -b 2048 \
  --spec-type draft-mtp --spec-draft-n-max 6 --draft-p-min 0.7 \
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

{
  echo "decode  ${LABEL}  temp=${TEMP}  $(date '+%Y-%m-%d %H:%M')"
  printf '%-8s %10s %10s %10s\n' prompt tg_t/s accept mean_len
} | tee "${OUT}"

i=0
for p in "${PROMPTS[@]}"; do
  i=$((i + 1))
  full="${p}"
  [[ -n "${PREFIX}" ]] && full="${PREFIX}"$'\n\n'"${p}"

  # Only read log lines this request produced. curl exits 0 on an HTTP 500, so without
  # the window a failed request silently reports the previous prompt's numbers again.
  mark="$(wc -l < "${LOG}")"

  curl -sS -m 1800 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg p "${full}" --argjson n "${NGEN}" \
          '{messages:[{role:"user",content:$p}],max_tokens:$n,seed:1234}')" > /dev/null

  new="$(tail -n +$((mark + 1)) "${LOG}")"

  # the server logs acceptance and mean length per request; the API response does not
  tg="$(grep -oE 'eval time =.*\(.*,[ ]*[0-9.]+ tokens per second' <<< "${new}" | tail -1 | grep -oE '[0-9.]+ tokens per second' | grep -oE '^[0-9.]+' || echo '-')"
  acc="$(grep -oE 'draft acceptance = [0-9.]+' <<< "${new}" | tail -1 | grep -oE '[0-9.]+$' || echo '-')"
  mlen="$(grep -oE 'mean len = +[0-9.]+' <<< "${new}" | tail -1 | grep -oE '[0-9.]+$' || echo '-')"

  printf '%-8s %10s %10s %10s\n' "p${i}" "${tg}" "${acc}" "${mlen}" | tee -a "${OUT}"
done

kill "${SRV}" 2>/dev/null || true
wait "${SRV}" 2>/dev/null || true
SRV=""

awk 'NR>2 && $1 ~ /^p[0-9]+$/ {tg+=$2; a+=$3; m+=$4; n++}
     END {if (n) printf "mean over %d: tg %.1f t/s, acceptance %.3f, mean len %.2f\n", n, tg/n, a/n, m/n}' \
  "${OUT}" | tee -a "${OUT}"

echo "done -> ${OUT}"
