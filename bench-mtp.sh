#!/usr/bin/env bash
set -euo pipefail

# A/B two llama-server configurations on the grafted Qwen3.8-Flash-Next MTP
# model, reporting prefill t/s, generation t/s, draft acceptance and steady
# state RSS.
#
# llama-server, not llama-cli: the `draft acceptance` line - the metric that
# says whether MTP is actually working - is only printed server side, and
# llama-cli's timing format is a one-line summary with no acceptance at all.
#
# Each arm runs in BOTH orders (A B B A). The first run after a model switch or
# on a cold page cache reads 10-15 t/s low and that penalty lands on whichever
# arm goes first, so compare per-arm means, never single runs.
#
# The two arms can be the same binary with different flags, which is the right
# shape for testing an opt-in option (no build difference to confound it):
#   EXTRA_B="--lazy-mode on-direct" ./bench-mtp.sh "$S" "$S"
#
# Usage: ./bench-mtp.sh <serverA> <serverB> [rounds]
# Env:   EXTRA_A / EXTRA_B  extra server args per arm
#        PROMPT_FILE        file to use as the prompt (for a meaningful prefill
#                           number - the default prompt is far too short)
#        NGEN, PORT, MTP_MODEL

BIN_A="${1:?usage: bench-mtp.sh <serverA> <serverB> [rounds]}"
BIN_B="${2:?usage: bench-mtp.sh <serverA> <serverB> [rounds]}"
ROUNDS="${3:-1}"

MODEL="${MTP_MODEL:-${HOME}/git/sammcj/llamacpp-ini/models/Qwen3.8-Flash-Next-MTP-Merged-GGUF/Qwen3.8-Flash-Next-MTP-UD-IQ4_XS-00001-of-00004.gguf}"
NGEN="${NGEN:-400}"
PORT="${PORT:-8919}"
TEMPLATE="${HOME}/git/Qwen-Fixed-Chat-Templates/chat_template.jinja"

for b in "${BIN_A}" "${BIN_B}"; do
  [[ -x "${b}" ]] || { echo "not executable: ${b}" >&2; exit 1; }
done
[[ -f "${MODEL}" ]] || { echo "model not found: ${MODEL}" >&2; exit 1; }

if [[ -n "${PROMPT_FILE:-}" ]]; then
  [[ -f "${PROMPT_FILE}" ]] || { echo "prompt file not found: ${PROMPT_FILE}" >&2; exit 1; }
  PROMPT="$(cat "${PROMPT_FILE}")"
else
  PROMPT='Write a C++ function that merges two sorted std::vector<int> into one sorted vector without using std::merge, then explain its complexity.'
fi

read -r -a EXTRA_A_ARR <<< "${EXTRA_A:-}"
read -r -a EXTRA_B_ARR <<< "${EXTRA_B:-}"

SRV=""
cleanup() { [[ -n "${SRV}" ]] && kill "${SRV}" 2>/dev/null || true; }
trap cleanup EXIT

# Mirrors the served config in samm-mbp.ini: q8_0 KV, draft-mtp n-max 6 with
# backend sampling, ub 2048, Qwen 3.8 sampler. Acceptance is sampler dependent,
# so these numbers only compare against runs at these same settings.
run_one() {
  local bin="${1}"; shift
  local log; log="$(mktemp)"
  "${bin}" -m "${MODEL}" --host 127.0.0.1 --port "${PORT}" \
    -ctk q8_0 -ctv q8_0 -ub 2048 -ngl 999 --ctx-size 32768 \
    --spec-type draft-mtp --spec-draft-n-max 6 --spec-draft-p-min 0.7 \
    --spec-draft-backend-sampling \
    --temp 1.0 --top-k 20 --min-p 0.0 \
    --chat-template-file "${TEMPLATE}" \
    "$@" > "${log}" 2>&1 &
  SRV=$!

  local waited=0
  until grep -q 'listening on' "${log}" 2>/dev/null; do
    sleep 2; waited=$((waited + 2))
    if ! kill -0 "${SRV}" 2>/dev/null || [[ ${waited} -gt 400 ]]; then
      echo "server failed to start; tail:" >&2; tail -6 "${log}" >&2
      rm -f "${log}"; SRV=""; echo "NA NA NA NA"; return 0
    fi
  done

  curl -sS "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg p "${PROMPT}" --argjson n "${NGEN}" \
          '{messages:[{role:"user",content:$p}],n_predict:$n,seed:1234,cache_prompt:false}')" \
    > /dev/null

  # Steady state RSS while the model is still resident. Mostly file backed
  # pages for an mmap'd model, so read it as "how much is paged in", which is
  # the number that matters when comparing mmap against direct reads.
  local rss_kb; rss_kb="$(ps -o rss= -p "${SRV}" 2>/dev/null | tr -d ' ')"

  kill "${SRV}" 2>/dev/null || true
  wait "${SRV}" 2>/dev/null || true
  SRV=""

  local pp tg acc
  pp="$(grep -oE 'prompt eval time =.*,[[:space:]]*[0-9.]+ tokens per second' "${log}" \
        | tail -1 | grep -oE '[0-9.]+ tokens per second' | grep -oE '^[0-9.]+')"
  tg="$(grep -E '^\[?.*\beval time =' "${log}" | grep -v 'prompt eval' \
        | tail -1 | grep -oE '[0-9.]+ tokens per second' | grep -oE '^[0-9.]+')"
  acc="$(grep -oE 'draft acceptance = [0-9.]+' "${log}" | tail -1 | grep -oE '[0-9.]+$')"
  rm -f "${log}"
  echo "${pp:-NA} ${tg:-NA} ${acc:-NA} $(( ${rss_kb:-0} / 1024 ))"
}

a_pp=(); a_tg=(); a_acc=(); a_rss=()
b_pp=(); b_tg=(); b_acc=(); b_rss=()

for ((i = 1; i <= ROUNDS; i++)); do
  for arm in A B B A; do
    if [[ "${arm}" == A ]]; then
      read -r pp tg acc rss < <(run_one "${BIN_A}" "${EXTRA_A_ARR[@]}")
      a_pp+=("${pp}"); a_tg+=("${tg}"); a_acc+=("${acc}"); a_rss+=("${rss}")
    else
      read -r pp tg acc rss < <(run_one "${BIN_B}" "${EXTRA_B_ARR[@]}")
      b_pp+=("${pp}"); b_tg+=("${tg}"); b_acc+=("${acc}"); b_rss+=("${rss}")
    fi
    printf 'round %d  arm %s  pp=%-8s tg=%-8s acc=%-9s rss=%sMB\n' \
      "${i}" "${arm}" "${pp}" "${tg}" "${acc}" "${rss}"
  done
done

mean() { printf '%s\n' "$@" | grep -v NA | awk '{s+=$1; n++} END {if (n) printf "%.2f", s/n; else printf "NA"}'; }

echo
printf 'A %s %s\n' "${BIN_A}" "${EXTRA_A:-}"
printf '   pp %s t/s   tg %s t/s   acceptance %s   rss %s MB\n' \
  "$(mean "${a_pp[@]}")" "$(mean "${a_tg[@]}")" "$(mean "${a_acc[@]}")" "$(mean "${a_rss[@]}")"
printf 'B %s %s\n' "${BIN_B}" "${EXTRA_B:-}"
printf '   pp %s t/s   tg %s t/s   acceptance %s   rss %s MB\n' \
  "$(mean "${b_pp[@]}")" "$(mean "${b_tg[@]}")" "$(mean "${b_acc[@]}")" "$(mean "${b_rss[@]}")"
