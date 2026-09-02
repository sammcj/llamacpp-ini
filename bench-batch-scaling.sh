#!/usr/bin/env bash
set -euo pipefail

# What does an n-token batch cost against a 1-token batch, at depth?
#
# Every decode number in QWEN_NEXT.md before this came from `-p 0 -n 32`, which is
# single-token decode. MTP verifies a whole draft in one batch, so the quantity that
# actually sets its economics is T(n): the cost of verifying n tokens. If T(n) is nearly
# flat over n=1..16, decode is overhead-bound and the lever is node count and CPU
# orchestration. If T(n) is linear, it is weight-bound and the only lever is bytes moved.
#
# The kernel-level half of this was already measured - `test-backend-ops perf -o
# MUL_MAT_ID` at the real 512-expert/10-active geometry gives iq3_s 1.68 TFLOPS at bs=1
# and 1.93 at bs=8, so the expert matmul is close to linear in n. But experts are only
# ~2.8 ms of a ~31 ms step, so that settles under a tenth of the question. This measures
# the whole step.
#
# IMPORTANT: llama-bench does not run the MTP head. T(n) here is the *trunk verify* cost
# only. Do not add a draft-step cost to these numbers - the draft is a separate ~8%
# measured elsewhere. This is what a verify of width n costs, nothing more.
#
# Arm 2 is the KV cache type, never A/B'd on this model. `-ctk q8_0 -ctv q8_0` was adopted
# on 2026-08-30 as a benchmarking convention to match the served config, not because it
# measured faster. Only 12 of 48 layers attend, so KV is small and f16 may well win once
# Metal flash-attn no longer has to dequantise it in-kernel.
#
# Usage: ./bench-batch-scaling.sh [outfile]

BIN="${LLAMA_BENCH_BIN:-${HOME}/git/llama.cpp-pr27836/build/bin/llama-bench}"
MODEL="${MTP_MODEL:-${HOME}/git/sammcj/llamacpp-ini/models/Qwen3.8-Flash-Next-MTP-Merged-GGUF/Qwen3.8-Flash-Next-MTP-UD-IQ4_XS-00001-of-00004.gguf}"
OUT="${1:-/tmp/claude/batch-scaling.txt}"

[[ -x "${BIN}" ]] || { echo "llama-bench not found at ${BIN}" >&2; exit 1; }
[[ -r "${MODEL}" ]] || { echo "model not readable: ${MODEL}" >&2; exit 1; }
mkdir -p "$(dirname "${OUT}")"

# -ub 2048 -b 2048 to match the served preset. At n<=16 the ubatch never binds, but the
# graph is built for it, so keeping it identical removes one difference from production.
COMMON=(-m "${MODEL}" -ngl 999 -ub 2048 -b 2048 -r 5)

VER="$("${BIN}" --version 2>&1 || true)"

{
  echo "batch scaling + KV type  $(date '+%Y-%m-%d %H:%M')"
  # Capture first, filter second. Piping the binary straight into head/grep lets the
  # reader close the pipe early, and SIGPIPE plus pipefail then kills the whole run
  # before a single benchmark starts - which is exactly what happened on the first try.
  printf '%s\n' "${VER}" | grep -i version || true
  echo
  echo "## Arm 1: T(n) trunk verify cost, q8_0 KV, depth 8192"
  echo
} | tee "${OUT}"

"${BIN}" "${COMMON[@]}" -n 0 -p 1,2,4,8,16 -d 8192 -ctk q8_0 -ctv q8_0 2>&1 | tee -a "${OUT}"

{
  echo
  echo "## Arm 2: KV cache type, single-token decode at depth"
  echo
} | tee -a "${OUT}"

for kv in q8_0 f16; do
  echo "### -ctk ${kv} -ctv ${kv}" | tee -a "${OUT}"
  "${BIN}" "${COMMON[@]}" -p 0 -n 32 -d 4096,32768 -ctk "${kv}" -ctv "${kv}" 2>&1 | tee -a "${OUT}"
  echo | tee -a "${OUT}"
done

{
  echo
  echo "## Arm 3: T(n) with f16 KV, depth 8192 (does KV type change the slope)"
  echo
} | tee -a "${OUT}"

"${BIN}" "${COMMON[@]}" -n 0 -p 1,2,4,8,16 -d 8192 -ctk f16 -ctv f16 2>&1 | tee -a "${OUT}"

{
  echo
  echo "## Arm 4: the cliff, f16 KV, depth 8192, r=8"
  echo
} | tee -a "${OUT}"

# Arms 1 and 3 step 1,2,4,8,16 and so straddle the cliff without locating it. This arm
# walks 7..16 at higher repetition to pin it: the step lands between 8 and 9, matching the
# `ne11 <= 8` bound on mul_mv_ext in ggml-metal-ops.cpp. Kept here rather than run
# ad hoc, because this is the row that drives both the served n-max and an upstream
# candidate, and an unreproducible number has no business doing either.
"${BIN}" -m "${MODEL}" -ngl 999 -ub 2048 -b 2048 -r 8 \
  -n 0 -p 7,8,9,10,12,14,16 -d 8192 -ctk f16 -ctv f16 2>&1 | tee -a "${OUT}"

echo "done -> ${OUT}"
