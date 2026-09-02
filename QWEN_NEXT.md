# Qwen3.8-Flash-Next MTP setup

Qwen3.8-Flash-Next (arch `qwen4exp`, 125B-A6B) ships a jointly-trained single-layer MTP draft head, but llama.cpp master deliberately drops the `mtp.*` tensors at conversion, so no public GGUF quant can self-draft. This repo works around that with a grafted head and a PR-branch build. Result on the M5 Max: 41.6 t/s plain, ~70 t/s with `draft-mtp` at ~0.70-0.76 draft acceptance.

Note the model's much-hyped "native ngram layers" (`qwen4exp.ple.*`, the `per_layer_token_embd` hashed 2/3-gram tables - ~95GiB at bf16, 26.8GiB at the IQ4_NL/Q4_0 most quants use, 50.7GiB at Q8_0 - the "51GB" figure community posts quote) are a capacity mechanism in the forward pass, not speculative decoding. The MTP head is a separate DeepSeek/GLM-style nextn block (llama.cpp's current implementation drafts with dense attention; the sparse/QSA draft path is an open TODO upstream).

## What is in place

- `graft-mtp-head.py` grafts a standalone MTP head GGUF onto the Unsloth 3-split quant as a 4th split. It rewrites shard 1 (block_count 48 to 49, adds `nextn_predict_layers = 1`, pads `compress_ratios`, split.count 4), hardlinks the two weight shards (no copy), and rewrites the head keeping only `blk.48.*` tensors, renaming the standalone `output_hc_{norm,down,up}` to `blk.48.nextn.hc_head_*`.
- `models/Qwen3.8-Flash-Next-MTP-Merged-GGUF/` is the grafted 4-split model (3.9GB extra disk).
- `heads/Qwen3.8-Flash-Next-MTP-GGUF/` holds the source head from HF `agentionai/Qwen3.8-Flash-Next-MTP-Q8_0-GGUF`. It lives outside `models/` on purpose: the router would otherwise list it as a servable model, and it cannot load standalone.
- `~/git/llama.cpp-pr27836/` is a git worktree building llama.cpp PR #27836 (qwen4exp NextN/MTP support). The main `~/git/llama.cpp` checkout stays on master.
- `samm-mbp.env` exports `LLAMA_SERVER_BIN` pointing at the worktree build; `run-llama-server.sh` sources it, so the whole router runs on the PR binary.
- `samm-mbp.ini` carries the `[Qwen3.8-Flash-Next-MTP-Merged-GGUF]` section with `spec-type = draft-mtp` (no ngram-mod - see below), `spec-draft-n-max = 6` and `spec-draft-backend-sampling = 1` (GPU-side draft sampling, +6% tg at 4K depth). Loosening `spec-draft-p-min` below the global 0.7 measured clearly worse (31 vs 44 t/s at 0.3); tightening to 0.75 or 0.85 measured as a wash once run order was controlled (the first llama-cli run after a model switch pays a 10-15 t/s cold-cache penalty, so A/B in both orders before believing a sweep).

The original `models/Qwen3.8-Flash-Next-GGUF/` entry is untouched (ngram-mod only), so the two can be compared by model name.

## Why the graft instead of `-md head.gguf`

PR #27836 creates the MTP context against the target model, never against an external draft, so the nextn tensors must be in the target's own file set. The `-md` external-head flow only exists in an unrelated fork (older #27739 design, ~62% acceptance vs ~89% here). Requanting from safetensors would need the ~250GB original; the graft gets the same result for 3.9GB.

The merged file carries the MTP block and nothing else: shard 4 has six `blk.48.nextn.*` tensors, no `shared_head_head`, and `token_embd.weight` / `output.weight` appear once each in shard 2 as the trunk's own. The head reaches them through the graph-level fallback at `src/models/qwen4exp.cpp:682`.

That is worth stating because [unsloth/llama.cpp#142](https://github.com/unslothai/llama.cpp/pull/142) exists to give the *sidecar* path the same property - an opt-in `--mtp-shared-embd` that omits `token_embd`/`output`/`output_norm` from a NextN sidecar and resolves them against the target, taking ggml-org's Qwen3.8-27B sidecar from 1.565GiB to 0.233GiB with byte-identical output. The 6.7x it reports is exactly the duplication the graft never had, so there is no size or speed gain here for us. The draw would be dropping the graft step entirely: a ~230MB stripped sidecar plus any quant of the base, instead of re-running `graft-mtp-head.py` over 90GB per quant. Not available yet - it is unmerged in Unsloth's fork against a pushed copy of upstream `6fe749801`, our `qwen4exp` is not in its arch list (though it has the needed fallback, and the loader half is arch-agnostic), and stacking it on #27836 across two bases is the real cost. Revisit if it lands upstream.

## Benchmarks

M5 Max 128GB, UD-IQ4_XS, C++ source-code prompts (prose lands a few t/s lower with the same ordering). Current config on the preset (LTO) build. Benchmark convention from 2026-08-30 on: always pass `-ctk q8_0 -ctv q8_0` to match the served config (the router's global `cache-type-k/v = q8_0`); numbers before that date used f16 KV. Target-KV q8_0 is safe on this arch - the crash reports concern the draft KV quant only.

| context      | no spec | draft-mtp | speedup |
| ------------ | ------- | --------- | ------- |
| short (n=200)| 41.6    | ~70       | +68%    |

A UD-Q4_K_XL graft (`models/Qwen3.8-Flash-Next-MTP-Q4KXL-Merged-GGUF/`, q8_0 KV) measured 68.8 t/s short and 30.3 at 32K vs IQ4_XS's 70.1 / 34.1 (f16 KV) - parity short, ~11% slower at depth. Decode is bandwidth-bound, so the larger working set (~77 vs ~60GB) outweighs the faster Q4_K Metal kernels. Caveat on that pair: the arms differ in KV type as well as model quant, so the depth gap is not cleanly attributable to the weights. f16 against q8_0 KV has never been A/B'd on this model - q8_0 was adopted to match the served config, not because it measured better - and with only 12 attention layers the KV is small enough that the Metal flash-attn in-kernel dequant could plausibly cost more at depth than the smaller cache saves. It stays available as a quality option; IQ4_XS remains the speed default. Cold-cache warning: first load after a reboot fault-storms the SSD (single-digit t/s, OS lag) - warm it with a throwaway generation, and never benchmark two large quants alternately (page-cache churn hard-locked the machine once).

Total memory, measured at 32K context:

- **mlock (old config)**: ~89GB wired - the whole 87GB file plus ~1.9GB context buffers. Nothing evictable.
- **mmap, ngram table demand-paged from SSD (current config)**: ~62-65GB steady state - ~60GB of non-PLE weights (read every token, so effectively always resident) + ~2GB context + a few GB of hot ngram rows. Only ~2GB of that is dirty/wired; the rest is file cache macOS can evict under pressure.

Saving: roughly 25GB, the cold portion of the 27.4GB ngram table, at no measured speed cost (mlock vs mmap A/B: 46.3 vs 47.0 t/s avg). Context buffers at 32K break down as 768 MiB KV (12 attention layers, f16) + 288 MiB QSA indexer + 64 MiB MTP layer + 113 MiB recurrent state + ~700 MiB compute (~1.1 GiB of that scales with context length). The `ubatch-size = 2048` setting lifts the compute buffer to 1.8 GiB Metal + 0.3 GiB CPU (read from the load log; peak process memory not re-measured), so steady state lands ~1GB above the figures above - still ~63-66GB all up.

The saving only exempts the ngram table - the MoE experts get touched broadly across a generation, so all non-PLE weights effectively stay resident. That caps which quants fit on 128GB. PLE sizes below were read from each file's GGUF header (`scratchpad gguf-remote-ple.py` trick: HTTP range requests, no download); the quantisers vary the PLE type per tier, so subtracting a fixed number misleads:

| quant               | file      | PLE quant/size  | non-PLE working set | verdict on 128GB                     |
|---------------------|-----------|-----------------|---------------------|--------------------------------------|
| UD-IQ4_XS (current) | 87.2 GiB  | IQ4_NL 26.8 GiB | ~60 GiB             | comfortable                          |
| UD-Q4_K_XL          | 103.7 GiB | IQ4_NL 26.8 GiB | ~77 GiB             | comfortable                          |
| Q5_K_S (bartowski)  | 119.3 GiB | Q5_1 35.8 GiB   | ~84 GiB             | fits, ~90GB all up                   |
| UD-Q5_K_XL          | 147.4 GiB | Q8_0 50.7 GiB   | ~97 GiB             | fits the 112GB budget, solo model    |

Budget note: up to ~112GB total (model + KV) is acceptable on this 128GB machine, leaving ~16GB for the OS. UD-Q5_K_XL lands around 100-103GB all up with context and hot ngram rows - the practical ceiling. UD-Q6_K_XL (157.5GiB) and beyond exceed it.

For CUDA boxes the same trick applies via `-ot "per_layer_token_embd=CPU"` (PLE gathers stay on CPU). On 2x RTX 3090 (47.5GB usable) the non-PLE weights + ~2-3GB context must fit VRAM: bartowski IQ1_M (~40GiB non-PLE) fits, IQ2_XXS (~43GiB) barely, IQ3_XXS (~55GiB) does not. With enough system RAM, `--n-cpu-moe` spilling routed experts as well lifts the quality ceiling at reduced speed.

The n-max sweep (plain Release build, before the ngram-mod drop; the preset build lifted the winner from 64.7 to ~70):

| spec-type           | n-max | gen t/s |
| ------------------- | ----- | ------- |
| none                | -     | 41.6    |
| ngram-mod           | -     | 42.1    |
| draft-mtp           | 3     | 59.2    |
| draft-mtp,ngram-mod | 3     | 59.9    |
| draft-mtp,ngram-mod | 4     | 60.0    |
| draft-mtp,ngram-mod | 6     | 64.7    |
| draft-mtp,ngram-mod | 8     | 46.4    |

The single-layer head drafts recursively, so n-max is not capped at 1. It was trained for ~4-step drafting; 6 still wins on throughput, 8+ collapses because every rejected deep draft wastes a full verify batch.

Acceptance sits around 0.70-0.76, not the 0.90 first recorded here. That 0.90 was 96/106 tokens - one short sample, and wrong. Larger runs at the served settings (temp 1.0, top-k 20, n-max 6): 0.758 over 4196 draft events on a 4K agent generation, 0.695 over a 300-token `bench-mtp.sh` run. For scale, unsloth's own head measures 0.66 greedy over ~10,300 events, and acceptance falls as temperature rises - so 0.70-0.76 at temp 1.0 is the honest figure and still a clear win.

At context depth (llama-bench for PP and plain TG; llama-cli with source-code prompts for spec TG):

| depth | PP t/s | TG plain | TG ngram-mod | TG draft-mtp | TG mtp+ngram |
| ----- | ------ | -------- | ------------ | ------------ | ------------ |
| 4K    | 1010   | 39.2     | 38.5         | 44.3         | 44.3         |
| 16K   | 781    | 34.3     | 33.9         | 40.6         | 39.4         |
| 32K   | 632    | 29.4     | 30.3         | 34.1         | 36.1         |

TG spec columns are the preset (LTO) build; the earlier plain Release build measured 41.4/34.0/31.2 combined, so the preset build is worth ~+16% at depth (LTO helps the per-step CPU-side draft/verify orchestration). PP and TG-plain columns are still from the plain build and read slightly low. The MTP gain thins from +45% (short) to ~15-20% at depth because verify batches pay the full attention/KV cost per step. Chat workloads with higher acceptance land higher still.

**ubatch (2026-08-30, on the #27992 build):** `-ub 2048` lifts 32K prefill 570 -> 661 t/s (+16%; 1024 lands at 627) with TG unchanged, costing ~1.1GiB extra compute buffer (1.8GiB Metal at 2048 vs ~0.7 default). Wired as `ubatch-size = 2048` in both MTP model sections.

**Long-context perf PRs #27992 vs #27977 (2026-08-30):** qwen4exp's PLE lookup calls `get_prev_tokens()` every decode step, which upstream scans all used KV cells - a cost that grows with context. Two open PRs attack it: #27992 (per-seq kv-cell position index) and #27977 (scan early-exit + QSA gather windows for generation + indexer sum-shape fix + bitmap used-cells). Either one beats the plain PR build at depth (+18% TG at 74K in the first A/B). Head to head on M5 Max (both-orders, q8 KV, ub 2048): TG identical within noise at 32K/74K/115K, but #27977 wins PP consistently at every depth - 685 vs 663 (32K), 511 vs 488 (74K), 409 vs 387 t/s (115K). **Carrying both regressed TG ~-18% at 32K** (27.6/29.0 vs 34.8/34.4), so `update-mtp-build.sh` merges exactly one: #27977. The CUDA-side headline gains (2.7x at 240K) do not transfer to Metal - unified memory already makes the CPU scan cheap - but the PP win and the correctness-adjacent fixes are free. Swap back to #27992 only if #27977 stops merging cleanly.

At depth on novel code, draft-mtp alone and mtp+ngram-mod are within noise of each other (ngram-mod barely fires). Where they differ, ngram-mod loses: short context 70.1 vs 65.8 avg, echo-heavy agent task 51.8 vs 48.0 avg (both-orders A/B). Same mechanism as ngram-cache below - draftless ngram drafts take per-step priority, so wherever ngram-mod matches, its lower-acceptance drafts displace the higher-acceptance MTP drafts. Hence `spec-type = draft-mtp` alone for this model; ngram-mod stays right for the models with no MTP head.

For calibration, the only published M5 Max 128GB numbers for this model (heretik.io) are tg 33 t/s and pp512 966 at depth 0 - this setup is ~20% ahead plain and ~2.1x ahead with MTP on the preset build (69.3 t/s) at short context. Ruled out as bottlenecks: flash attention (auto resolves on, ±1 t/s), the PR build itself (~1% vs master), and the lazy-read ngram table (`--lazy-mode off`, formerly `--tensor-read-lazy`, made no difference, matching PR #27794's claim). The ~40 t/s plain decode is what a 125B model with per-token ngram gathers does on this hardware.

The ngram table (`per_layer_token_embd`, 27.4GB in this quant) is lazy-read upstream (#27794): excluded from load prefetch, `madvise(RANDOM)`, pages fault in from SSD on demand and only hot rows stay cached. Verified here - a spec run's peak dirty footprint is 1.6GB and the full file is never resident. mlock defeats this by wiring the whole mapping, so the model's ini section sets `load-mode = mmap` (overriding the global `mmap+mlock`), keeping tens of GB unwired at no measured speed cost. That headroom, plus the fact that the whole mapping is evictable under pressure, is what makes a larger trunk quant (UD-Q4_K_XL, 104GB file) feasible on 128GB.

Two cautions from community reports: quantised KV cache crashes this arch (leave the commented `cache-type-*-draft = q8_0` lines in samm-mbp.ini commented), and very long contexts can exceed the default Metal wired limit (a 262K run needed `iogpu.wired_limit_mb` raised; our 131072 ctx-size has headroom but is worth remembering).

## Moving forward

- After pulling llama.cpp master, run `./update-mtp-build.sh` to bring the worktree along: it fetches the latest PR head, merges origin/master and PR #27977 (falls back gracefully when either stops merging cleanly), rebuilds, and exits early when nothing changed. It also detects the PR merging upstream and prints the retirement steps. It builds with the main repo's `local` cmake preset (LTO, native; symlinked into the worktree since `CMakeUserPresets.json` is untracked) and never runs `cmake --install`, so the PR binaries stay out of `~/.local`.
## PR triage (2026-09-01)

`update-mtp-build.sh` merges a list of extra PRs on top of #27836; `EXTRA_PRS` in that script is the single source of truth for which, with the reason for each beside it. It also re-applies `patches/` after the merges, since `checkout -B` discards working-tree edits every run. `bench-mtp.sh` A/Bs two llama-server binaries in A B B A order on TG and acceptance; with a fixed seed and `cache_prompt=false` the runs come out bit-identical, so a real difference shows up immediately.

- **#28123 recurrent state rollback: merged, already in the build.** It is commit `0eadefebd`. Before it, an MTP draft had to serialise the whole recurrent state to host memory - the PR's author measured MTP as *slower than no drafting at all* on prose (83 vs 108 t/s). Anything measured here before 2026-09-01 predates it.
- **#27941 follow-up fixes: added, neutral on speed, kept for correctness.** A/B against the previous build measured 56.68 vs 56.70 t/s and identical acceptance - no gain, no regression, single stream. It stays for two reasons that do not show up in a one-request benchmark: blocks are keyed per sequence rather than by position alone (wrong under `kv-unified`, which is what the router's 4 auto slots run), and eight `GGML_ASSERT` sites reachable from a hand-edited GGUF become throws - this model *is* hand-edited.
- **#28086 metal iq3_xxs: does not apply.** It patches `mul_mv_iq3_xxs`. This model's tensors are q8_0 523, iq3_s 94, iq4_nl 44, bf16 24, iq4_xs 2, q6_K 1 - no `iq3_xxs`, and `iq3_s` is a different kernel. Nothing else in the farm is IQ3_XXS either. Reports of a large Metal win come from IQ3_XXS models.
- **unsloth MTP head GGUFs (`unsloth/Qwen3.8-Flash-Next-GGUF/MTP/`): not an upgrade.** They need unslothai/llama.cpp PR #144 and the `-md` external-draft path; their README states plainly they do not work on mainline as of `0eadefebd`. Acceptance 0.66 greedy against 0.70-0.76 here at temp 1.0. Two of their findings do transfer: `shared-` heads borrow the embedding and output projection from the target (the same reason the graft pairs with any quant), and BF16 heads are bigger *and* slower because the draft step is dominated by the output projection, which is cheaper at 8 bits - Q8_0 is the right tier.
- **#28097 draft-head-only GGUFs: watch, high value.** Teaches #27836's loader to accept a head-only GGUF (`mtp_only` detection). If it lands, `graft-mtp-head.py`, the 4-split model and its 3.9GB all retire, while keeping the in-target MTP context that earns the higher acceptance. Draft; conflicts with this stack today.
- **#28104 MTP port to master: watch.** A different implementation of the same feature; would supersede #27836. Draft; conflicts here, as expected.
- **#28136 direct PLE reads: measured, no effect on Metal, dropped.** The PR's >2x prefill claim comes from GB10, where mmap misbehaves. A/B on one binary a flag apart (~5K-token prompt, 250 gen, A B B A): pp 1039.72 vs 1039.91 t/s, tg 48.86 vs 49.39, acceptance unchanged, RSS 70421 vs 69760 MB. Unified memory already makes the PLE gather cheap, matching the earlier `--lazy-mode off` and Metal-buffer results. It also does not merge cleanly in a semantic sense: another PR in the stack rewrapped the PLE weight lookup as an `if`-scoped pointer while #28136 expects a function-scope value, so every build needs a hand-applied `ple_w` fix that `checkout -B` then destroys. No measured gain, recurring manual cost - removed from `EXTRA_PRS`. Note the option was renamed upstream from `--tensor-read-lazy` to `--lazy-mode`.
- **#28121 noscan ssm_a flags: merges clean, 4 lines, untested.**

## Knob re-tune after #28123 (2026-09-02)

Re-measured because n-max 6 and p-min 0.7 were both chosen on a build predating #28123. Before it a rejected draft serialised the whole recurrent state to host memory, so deep drafts and a tight confidence gate were hedges against expensive rejection; #28123 made rejection cheap, which is a direct reason to expect the optima moved. They did not. Build 9359, ~6K-token prompt, 250 generated, three requests per config with the first discarded as warmup.

- baseline (n-max 6, p-min 0.7): pp 1032.64, tg 50.05, acceptance 0.73
- n-max 4: pp 1030.58, tg 49.82, acceptance 0.79
- n-max 8: pp 1024.34, tg 43.14, acceptance 0.73
- p-min 0.0: pp 1024.88, tg 43.59, acceptance 0.42
- parallel 1: pp 1037.90, tg 50.12, acceptance 0.72

Every wired setting survives, so nothing in samm-mbp.ini changes. What the sweep adds:

- n-max 8 still costs 14%. Cheaper rejection did not buy draft depth - what a rejection wastes is the verify batch, and #28123 did not change that.
- `spec-draft-p-min` is the highest-value knob on this model. Ungating costs 13% tg and halves acceptance (0.73 to 0.42): drafting while the model is unconfident is actively harmful, not merely neutral.
- Acceptance is a trap metric. n-max 4 posts the best acceptance in the table and is not the fastest - shorter drafts are accepted more often per token while returning fewer tokens per verify batch. Rank configs by tg, never by acceptance.
- `parallel 1` is neutral (+0.1%, inside noise) with acceptance unchanged. That is **not** a clean read on slot count: `n_parallel < 0` sets `n_parallel = 4` *and* `kv_unified = true`, so passing `--parallel 1` also drops to non-unified KV and changes per-slot context sizing. The arm compared "4 slots, unified" against "1 slot, non-unified", and the honest conclusion is only that the auto default is not costing anything - not that slot count specifically is free. Isolating it needs `--parallel 4 --kv-unified` against `--parallel 1 --kv-unified`. It also does not settle #27572, which is about acceptance collapse under *concurrent* requests; every request here ran alone.
- **Every acceptance number in this document was taken with `cache_prompt=false`.** `bench-mtp.sh` posts it that way deliberately, for bit-identical runs. Production serves `cache_prompt=true`, and no script that sets it parses acceptance at all, so acceptance on the cached path has never been observed. The knob choices above are therefore tuned on the uncached shape and extrapolated to the served one. That extrapolation is untested rather than known-wrong, but it is the assumption to break first if the wired config ever looks worse in practice than it does here.

## Where the decode time actually goes (2026-09-02, measured)

Direct per-context measurement, not inference. The server runs the draft in its own `ctx_dft`, and `llama_perf_context()` already times both contexts; a logging patch in `tools/server/server-context.cpp` (`spec split` line) prints them. It is not in the working tree - recover it with `git stash list | grep 'spec-split instrumentation'`, since stash indices shift on every push. Per request, 250 tokens, baseline config:

- draft steps: **1.94 ms each**, 208 steps, **402 ms total - about 8% of the 5219 ms generation time**
- target single-token decodes: 31.4 ms each, 24 of them, 755 ms

**A draft step costs 1.94 ms, not the 12-16ms two different inferred models claimed.** Making draft steps entirely free would gain at most 8%, so every optimisation aimed at draft-step cost is chasing that ceiling: vocabulary shortlisting, cheaper draft heads, fewer draft steps. Do not spend effort there.

This retro-explains both failed experiments exactly. p-min 0.8 removed 39 draft steps, worth 39 x 1.94 = 75ms of 5134ms = 1.5%, i.e. noise - which is what it measured. And the Q4_0 head cut part of a 1.94ms step, so its +3.4% cannot be draft speed; it came from acceptance rising 0.722 to 0.760, which removes *verify* passes.

The lever is therefore **mean accepted length**, because it divides the verify-pass count directly. Every config measured lines up with it: mean len 3.52 gives tg 48.7-50.8, 2.90 gives 45.1, 2.53 gives 47.9.

Caveat on the rest: `n_eval` counts only single-token decodes, so multi-token verify batches land in `n_p_eval` mixed with prompt prefill. Summing all four counters exceeds wall-clock time, so the remaining ~77% is not cleanly attributable between prefill and verify without a further counter. Three cost models have now died from inferring this rather than measuring it - do not attempt a fourth from aggregate numbers.

## Where the cold prefill time goes (2026-09-02, measured)

Opening a fresh agent session against a 32k repo context costs ~48 s before the first token, an order of magnitude more than any single generation in the same session. `bench-prefill.sh` measures it: production server args, distinct prompts so every one is a genuine cold prefill, rebuilds in place.

| prompt tokens | ms | tok/s | ms/token |
|---|---|---|---|
| 4356 | 4227 | 1031 | 0.970 |
| 8538 | 8599 | 993 | 1.007 |
| 20745 | 24570 | 844 | 1.184 |
| 30386 | 41385 | 734 | 1.362 |
| 34006 | 48012 | 708 | 1.412 |

Fitting `ms/token = 0.905 + 1.49e-5 n` splits the cost: a flat **1105 tok/s ceiling** worth 66% of a 32k prefill, and depth scaling worth the other 34%. The ceiling, not the scaling, is the thing to attack.

Three candidates were eliminated by direct A/B, each a null:

- **PR #28040**, `get_prev_tokens` from a full cell walk to an indexed lookup: 701.7 vs 702.6 tok/s. Its author reports the same for pp.
- **The QSA sparse-attention indexer** (`build_qsa_top_k`, a `ggml_top_k` across every `n_kv` block per token per layer): forced off behind an env var so attention ran dense, 704.9 vs 702.6 tok/s.
- **Attention generally**, by implication - forcing the dense path, which is strictly more work at 32k, changed nothing measurable.
- **QSA gather (#28213)** goes the other way, gathering only the ~2048 selected KV entries rather than masking the full cache, and is also a null here - slightly negative to 32k, level at 64k, +1.4% at 128k. Its author measures +6% at 31k and +50% at 130k on dual A6000. Two ablations in opposite directions both landing on nothing is the strongest evidence yet that QSA attention is simply not what this machine is waiting on. See UPSTREAM-CANDIDATES.md for the numbers.

### The idle sleep was the actual problem (2026-09-02, measured)

Before any of the per-op work below matters, this: `handle_sleeping_state()` calls `destroy()`, and `load_model()` then rebuilds `prompt_cache` with `make_unique`. Waking discards every cached prefix.

`bench-sleep-cache.sh`, one 28903-token prompt sent three times (build 9359, before any of the PRs below):

| | tokens prefilled | time |
|---|---|---|
| cold | 28903 | 38.1 s |
| warm repeat | 4 | **0.076 s** |
| after sleep + wake | 28903 | 38.1 s |

`run-llama-server.sh` passed `--sleep-idle-seconds 1200`, so any break over 20 minutes made the next agent prompt fully cold.

**Fixed properly by PR #28022** (`--sleep-preserve-cache`), which writes the idle slot states into the prompt cache before sleeping and restores them on wake. It is off by default, and merging the PR without the flag changes nothing - the first run here still measured 38.2 s after a wake. Same build, same prompt, with the flag:

| | tokens prefilled | prefill | wall |
|---|---|---|---|
| cold | 28903 | 38.3 s | 38.3 s |
| warm repeat | 4 | 0.086 s | 0.1 s |
| after sleep + wake | 4 | **0.100 s** | **3.2 s** |

The 3.2 s wall against 0.1 s of prefill is the weight reload, which the timings block does not cover - worth measuring, because it is the entire remaining cost of a sleep. With the cache surviving, the idle window no longer has to span a working day, so `LLAMA_SLEEP_IDLE` defaults to an hour: the ~94 GB comes back over lunch and overnight, and waking costs a few seconds rather than 38.

A restart is the other way the cache dies, and #28022 does nothing for it - the fourth step of the same test measures a full 39.7 s. **PR #28092** (`--cache-disk`) writes the state under a directory and reloads it on start, taking that to 0.09 s prefill, with the restore costing nothing at startup either (6.0 s to first listen on the restart against 9.0 s cold). About 900 MiB of disk per ~33k prompt, capped at 32 GiB.

It is set per model by `sync-models.py`, not by the launcher: the router builds each child's arguments from its own argv, and a child deletes every entry in its cache directory whose key is not its own, so one shared path would mean each model wiping the last one's cache.

Both flags are probed against `--help` before being passed, since neither PR is upstream yet and an unknown flag is fatal on the stock binary the MBA uses.

Prefix reuse itself works well and did not need fixing. Two prompts sharing a 28.9k prefix and differing only in their last ~20 tokens: the first cost 28903 tokens / 38.1 s, the second **2047 tokens / 3.6 s**, a 10.6x saving. Note the 2047 is not incidental - it is `4 + n_ubatch`, the fallback checkpoint distance, and this is the one shape where the `checkpoint_offsets` patch in `patches/` genuinely helps: it takes that 3.6 s to 0.45 s.

With #28022 in, the cost order for an agent workload becomes: the first prompt of the day (38 s, unavoidable), then a wake (3.2 s), then prefix divergence (3.6 s, or 0.45 s with the checkpoint patch), then everything in the per-op profile.

### Where it actually goes, per op (2026-09-02, measured)

`llama-profile-ops` (a tool added under `examples/profile-ops/`, not in the working tree - `git stash list | grep profile-ops`) sets `params.cb_eval`, which makes `ggml_backend_sched` compute one node at a time with a `ggml_backend_synchronize` before handing it back, so every node can be timed. One 2048-token ubatch at 8k context.

The measurement calibrates itself: VIEW, RESHAPE and PERMUTE are metadata-only no-ops, and all three report **152.6 us/call**. That is the per-node synchronisation floor. Subtract it from every op and those three fall to 0.0 ms, which is the check that the correction is right. 1236 ms of the 3001 ms measured was instrumentation; 1765 ms was real work.

| op | net ms | share |
|---|---|---|
| MUL_MAT_ID (experts) | 492.1 | 27.9% |
| MUL_MAT (dense projections) | 431.5 | 24.4% |
| MUL | 171.3 | 9.7% |
| GATED_DELTA_NET | 138.5 | 7.8% |
| ADD | 131.9 | 7.5% |
| CONT | 64.7 | 3.7% |
| FLASH_ATTN_EXT | 61.9 | 3.5% |
| UNARY | 52.0 | 2.9% |
| RMS_NORM | 47.1 | 2.7% |
| TOP_K + ARGSORT | 32.7 | 1.9% |
| everything else | ~140 | ~8% |

Grouped by role rather than op: MoE ~32%, the GDN block and its projections (`z`, `linear_attn_out`, `conv_output_*`) ~15%, the hyper-connection scaffolding (`hc_inject` 51.6, `hc_norm` 31.7, `hc_gate` 26.8, `hc_combine` 24.9) ~7.6% across 338 calls, attention including flash-attn ~6.5%.

**There is no dominant term.** Attention plus the QSA indexer is 5.4%, which independently confirms the dense-ablation null rather than contradicting it. The graph runs ~7000 nodes for 48 layers, ~145 per layer, and elementwise MUL/ADD/UNARY/SCALE together are 20.6% across 2562 calls - the largest fusion target, but still only a fifth.

**Read that 20.6% as an upper bound, not a measurement.** `cb_eval` forces one node at a time, which disables Metal op fusion as well as serialising. The 152.6 us/call correction above removes the synchronisation cost but not the fusion effect, so a normal run already fuses some of those ops and the gain still on the table is smaller than 20.6% by an unknown amount. `GGML_METAL_FUSION_DISABLE` (`ggml/src/ggml-metal/ggml-metal-context.m:137`) makes this a one-run check against a normal build; it has not been done. The "no dominant term" conclusion does not depend on it - the expert-count ablation below reaches the same place independently.

The practical consequence: nothing here yields a 2x. Removing the hyper-connection machinery entirely would take 48 s to 44 s. Effort is better spent not paying the cold prefill at all than making it faster.

The profile did pay off once, though, by saying where to look. GDN at ~15% is the second-largest role, and PR #25788 fuses the recurrent-state snapshot write into the Metal `gated_delta_net` kernel instead of a per-layer `cpy` - **+3.5% decode at every depth from 4k to 32k, and +2.3% cold prefill**. Small, but it is the only change so far that helps both halves, and it is the size the profile predicts: a few percent, not a multiple.

A cross-check by ablation agrees on the shape. Routing to 1 expert instead of 10 via `--override-kv qwen4exp.expert_used_count=int:1` - a 90% cut in expert FLOPs - moves cold prefill only 702.6 -> 805.6 tok/s, putting MoE at ~14% of a 32k prefill. The profile's 32% is at 8k with no depth term and with overlap removed by serialisation; both agree MoE is a minority.

### On the quant kernels specifically

The expert matmul is **not** limited by the quant type. `test-backend-ops perf -o MUL_MAT -b MTL0` at m=4096, n=512, k=14336 puts IQ3_S at **36.73 TFLOPS**, inside 5% of f16 (38.08) and of every other quant in the set (34.8-41.7). At prefill batch sizes the dequant is fully hidden by the MMA pipeline. An earlier reading of this file blamed the IQ3_S dequant kernel; that was wrong, and it came from quoting `MUL_MAT` numbers taken at n=2..5, which is the matvec regime and not what prefill runs.

The cost is the **MoE routing shape**, and it is quant-independent. `MUL_MAT_ID` on Metal, 128 mats / 8 used / m=768 / k=2048:

| n (rows per expert) | q4_0 | q4_K | q8_0 | f16 |
|---|---|---|---|---|
| 32 | 2.52 | 2.16 | 1.72 | - |
| 64 | 4.34 | 3.76 | 2.90 | 1.61 |
| 128 | 8.12 | - | - | 3.12 |

Against 36-38 TFLOPS for a dense matmul of similar total size, that is a **10x loss**, and f16 sits in the same hole, so it is a property of the MoE kernel rather than of any quant.

That table is **not** our shape, and reading it as ours was a mistake. It is the qwen3-30b case: 128 mats, 8 used. With `test-backend-ops` cases added for this model's real geometry - 512 mats, 10 used, m=640, k=2560, `n_embd` 2560 and `n_ff_exp` 640 read off the GGUF rather than guessed - the picture at our served ubatch is completely different:

| rows/expert source | IQ3_S | Q4_K | IQ4_NL | f16 |
|---|---|---|---|---|
| bs=1 | 1.68 | 2.26 | 2.23 | 0.93 |
| bs=8 | 1.93 | 2.00 | 2.09 | 0.60 |
| bs=512 | 9.75 | 9.99 | 9.34 | 3.92 |
| **bs=2048 (our ubatch)** | **19.18** | 19.21 | 18.48 | 11.88 |
| bs=4096 | 22.57 | 23.00 | 22.05 | 12.88 |

At the ubatch we actually serve, the routed matmul runs at **19 TFLOPS**, not the 2.5-4.3 the wrong-shape table suggested, and IQ3_S is level with Q4_K. Expert-count matters enormously: 512 experts behave nothing like 128.

The bs=1 and bs=8 rows are the **MTP verify-width** case, and they say the draft width buys nothing in this kernel: 1.68 to 1.93 TFLOPS across the whole range an MTP draft can occupy, against 19.18 at bs=2048. With 512 experts and 10 active, eight tokens almost never land on the same expert, so there are no weight reads to amortise and the expert matmul cost scales close to linearly with draft width. Amortisation only appears once each expert gets many rows, which needs a prefill-sized batch. This is the reason acceptance, not draft depth, is the lever on this model - a longer draft costs proportionally more to verify. It also means an end-to-end `llama-bench -p 1,2,4,8,16 -d <depth>` would be measuring CPU-side orchestration, since the kernel half of that question is now answered.

Raising `ubatch` to buy rows per expert does not work, measured twice. At ubatch 4096 cold prefill drops to **634.9 tok/s from 702.6** - whatever the wider tile buys back is more than lost elsewhere.

So there is no quant-kernel lever and no MoE-occupancy lever. Both were hypotheses built by reading a benchmark taken at the wrong shape, which is the same failure mode as the three cost models before them. The per-op profile above is what should be trusted instead.

`MUL_MAT_ID` had no `test-backend-ops` coverage for IQ3_S or IQ4_NL, nor for any 512-expert geometry. Cases for both were added (`tests/test-backend-ops.cpp`, the qwen3.8-flash-next block); note `ffn_down_exps` has k=640, which is not a multiple of QK_K, so only block-32 types are legal for that half - which is exactly why that tensor is IQ4_NL in the GGUF and not a K-quant.

## Q4_0 draft LM head: rejected on multi-prompt retest (2026-09-02)

**Verdict: do not adopt.** The +3.4% below held on exactly one prompt. Retested with `verify-draft-head.sh` over four prompts at the production sampler: mean tg 53.31 against baseline 53.72 (**0.8% slower**), mean acceptance 0.72 against 0.73. Per prompt the baseline wins three of four. The gain was one prompt's artefact, which is precisely the n=1 acceptance caveat that was flagged and then not respected.

The mechanism is now clear and predicts the result. `qwen4exp.cpp` falls back to `model.output` when `nextn.shared_head_head` is absent, so the draft head and the target's LM head are *the same tensor*. Acceptance measures agreement between draft and target, so a perfectly matched head is already the optimum - any requant can only reduce agreement. Do not expect a draft-head precision change to help on this architecture; it can only help where draft and target use genuinely different weights.

Related correction to the quality claim: greedy output is **not** byte-identical between the two heads. Two of four prompts diverge, each with an identical prefix and then a single flip that cascades (char 589 of 1301, char 715 of 1617). This is not the draft leaking into output - `common_sampler_sample_and_accept_n` still emits only the target's own sampled token. Different acceptance patterns produce different verify *batch shapes*, and float reduction order on Metal varies with batch shape, so logits differ in the low bits and a near-tie argmax can flip. Both outputs are valid greedy decodes of identical target weights, but "identical output regardless of draft" is too strong a claim: the draft cannot choose a token, yet it does perturb reproducibility.

## Q4_0 draft LM head: the original single-prompt result (2026-09-02)

`qwen4exp.cpp` picks `layer.nextn.shared_head_head` when present and falls back to `model.output` otherwise, and the graft writes no `shared_head_head` - so the MTP draft step was reusing the trunk's Q6_K head (497MB, 635.7M params at n_embd 2560 x vocab 248320). `add-draft-head.py` writes a Q4_0 copy (341MB) into the head shard; the target keeps Q6_K untouched. Splits 2-3 are hardlinked, so it costs ~3GB.

Measured over 5 requests per arm, back to back in one session: tg 50.81 vs 49.16 (**+3.4%**), acceptance 0.760 vs 0.722, pp unchanged. The distributions do not overlap - every Q4_0 request (50.51-51.18) beat every baseline request (49.00-49.49).

Quality is unaffected by construction, not by argument. `common_sampler_sample_and_accept_n` (common/sampling.cpp) emits the token the *target* sampled and uses the draft token only as an equality test - `result.push_back(id)`, never `push_back(draft[i])`. A worse draft head lowers acceptance and slows generation; it cannot change the text. Byte-identical output versus no speculation is not guaranteed, because a rejected position is sampled, discarded and later re-sampled, which shifts the RNG stream.

Caveat on the acceptance figure: it is deterministic for a given prompt and seed, so the five identical 0.75962 readings are one sample of acceptance, not five. Whether a Q4_0 head raises acceptance generally needs several prompts.

Not adopted into the farm yet - the +3.4% is real but the mechanism is not understood (see below), and the model would need regenerating on every requant.

## p-min above 0.7 and the cost-model failures (2026-09-02)

`spec-draft-p-min` had only ever been tested downward (0.0, much worse) and at 0.7. Tested upward on build 9359, same harness:

- p-min 0.70: tg 48.73, acceptance 0.72, 71.0 verify passes, 223 draft steps, 5134ms
- p-min 0.80: tg 48.33, acceptance 0.81, 71.4 verify passes, 184 draft steps, 5176ms
- p-min 0.90: tg 45.09, acceptance 0.84, 86.2 verify passes, 151 draft steps, 5487ms

0.7 stays optimal, so nothing changes. The useful part is what it did to two cost models, both of which are now dead:

- **Weight bytes per draft step.** Predicted +13.6% from a Q4_0 draft head (see below); measured +3.4%, with acceptance moving *up* (0.722 to 0.760) rather than down as predicted. Most of that gain was probably the accidental acceptance change, not the 156MB saved.
- **Draft-step count.** Going 0.7 to 0.8 removed 39 draft steps (-17%) at identical verify count and token count, and saved *no* time. At an inferred 12-14ms per draft step that should have been ~550ms.

Solving `T = V*T_verify + D*T_draft` across config pairs gives contradictory answers (14ms/31ms from the n-max pair, 6.2ms/52.8ms from the p-min 0.7/0.9 pair, and a negative T_draft from 0.7/0.8), so the linear model is invalid: verify cost scales with draft width, which is a third unknown the configs confound. Do not attribute decode time to draft versus verify from aggregate t/s - two hypotheses have now died that way.

The direct measurement instead: the server keeps the draft in a separate `ctx_dft`, and `llama_perf_context()` already tracks `t_eval_ms`/`n_eval` per context. Printing it beside the existing timing line gives the split exactly, with no inference and no compute change.

Also corrected: the ~41.6 t/s plain-decode figure quoted above predates this build. Solving from measured verify/draft counts puts plain decode nearer 30 t/s, so do not use 41.6 as a baseline for cost models.

Measurement notes for whoever repeats this. A request is ~12s (prefill 5.75s for 5963 tokens, then 250 tokens at ~50 t/s) while a model load is ~45 min once the page cache is cold, so the load count is the entire runtime - sweep configs, never A/B pairs, and reuse one server for all requests of a config. An earlier pairwise attempt (16 loads) also drove the machine into swap and produced 16% within-arm spread on tg and one 6.66 t/s prefill outlier, against a 0.12% same-binary noise floor measured when idle. If arm-internal spread exceeds ~1%, the machine is thrashing and the run is void.

- #27977 conflicts with master as of 2026-09-01, in two comment-only hunks: master adopted the PR's `left > 0` early-exit (`llama-kv-cells.h`) and its summed-slice loop (`qwen4exp.cpp`) but reworded both comments. The code is identical either side, so the resolution is "keep master's wording". That resolution is recorded in git rerere (enabled globally) and replays automatically, which is what makes the script's merge succeed - `checkout -B` rebuilds the merge from the PR head every run, so without rerere it would need hand-resolving each time. The cache lives in `~/git/llama.cpp/.git/rr-cache` and is never committed: wipe it, or move to another machine, and the merge conflicts again. The script then says so loudly and names the files rather than dropping the PR quietly. To re-record: `cd ~/git/llama.cpp-pr27836 && git merge refs/pr/27977`, take master's comment in both hunks, `git commit`.
- Watch [PR #27836](https://github.com/ggml-org/llama.cpp/pull/27836). When it merges: update the normal llama.cpp install, delete the `LLAMA_SERVER_BIN` override in `samm-mbp.env`, and remove the worktree (`git -C ~/git/llama.cpp worktree remove ../llama.cpp-pr27836`). The grafted model keeps working on the merged code.
- The merged model does not load on master builds. If the server ever fails on it with "MTP" or `nextn` tensor errors, the binary is not the PR build.
- Related fixes to watch: [#27897](https://github.com/ggml-org/llama.cpp/pull/27897) (combined draft-mtp + external draft init, fetched locally as branch `pr-27897-fix`; not needed for our no-`-md` path) and [#27572](https://github.com/ggml-org/llama.cpp/issues/27572) (acceptance collapse with parallel slots `-np N`; retest before enabling parallel serving on this model).
- For the MBA or a new quant: rerun `graft-mtp-head.py <shard1> heads/Qwen3.8-Flash-Next-MTP-GGUF/Qwen3.8-Flash-Next-MTP-Q8_0.gguf <out-dir> <basename>`. The head pairs with any quant of the base model because it shares the trunk's embeddings and LM head.
- Once upstream converters keep the MTP tensors, freshly converted quants will carry nextn embedded and the graft (plus `heads/`) can be retired.
- Do not expect the 70-100 t/s that Qwen 3.5 122B-A10B managed with a 4B draft model: a dense drafter runs 8-16 tokens deep per verify, while the MTP head's recursion decays past its 4-step training depth (useful max ~6), and Flash-Next's verify step also pays for ngram gathers, a 248K-vocab output matmul and GDN state rollback on rejection. If a small Qwen4-family sibling with the 248K tokeniser ever appears, benchmark it against MTP.

## Prompt-cache continuation: one ubatch per turn on stock, 74 tokens with the patch (2026-09-02, measured)

Everything above this line ran `cache_prompt=false`, the cold-start worst case. With caching on, in the shape agent traffic actually has - a conversation growing by one short message per turn, 5963-token prompt:

| build | per-turn prompt_n | per-turn prompt_ms |
|---|---|---|
| stock `{4 + n_ubatch, 4}` | 2058 | 2393 |
| local `{4 + n_ubatch, 64, 4}` | 74 | 476 |

On stock the replay is one ubatch, so it does scale with `-ub` - the earlier `-ub 1024` reasoning was right about the mechanism. With the patch the replay is 74 tokens and `-ub` stops mattering (470.5 ms at 2048 vs 469.4 at 1024), so `ubatch-size = 2048` stays on the cold-prefill tuning. If the patch is ever dropped, revisit that line.

**One artefact to know about.** `sweep-cache.sh` reports a 2054-token replay per extend even on the patched build, because it extends a conversation whose *previous* request was an exact repeat - a repeat prefills 4 tokens and lays down no usable checkpoint. That shape does not occur in agent traffic; the per-turn figures above come from a growing-conversation harness instead.

**Mechanism.** `n_rs_seq` is not the cause. The replay length is set by `tools/server/server-context.cpp:3538`:

```
static const int checkpoint_offsets[] = {4 + n_ubatch, 4};
```

Prefill deliberately breaks `4 + n_ubatch` and `4` tokens before the end so checkpoints land there, and `near_prompt_end` (`:3559`, `:3598-3601`) forces both through regardless of `checkpoint_min_step` - which is why that knob and `ctx-checkpoints` were inert in every arm tested (32/8192 and 64/512 gave a byte-identical 2054). On restore, the `N-4` checkpoint is rejected whenever the request has new tokens, because `hparams.n_swa == 0` for this arch makes `pos_min_thold == pos_next` (`:3270`, `:3323-3336`) and that checkpoint has already consumed the divergent token. So the fallback is the `4 + n_ubatch` one. The arithmetic closes exactly: 5963 - min(2048, 2052) = 3915, and 5969 - 3915 = 2054.

**Raising `n_rs_seq` is a dead end** if the thought comes up again. One recurrent snapshot row for this model is **112.57 MiB** per sequence (36 GDN layers, `n_embd_r` 30720 + `n_embd_s` 786432 f32, plus PLE conv state), and rows are `mem_size * (1 + n_rs_seq)`. Covering a whole turn needs thousands of slots. At `n_rs_seq` 6 with `--parallel 4` this already costs 3.08 GiB; `--parallel 1` would return 2.3 GiB of that if the router is ever single-user.

## The client must send the thinking block back, or every turn re-prefills the answer (2026-09-02, measured)

This is the one that matters for agent traffic. It is mostly a client-side property, but the earlier claim here that it is "not a server tuning knob" was wrong - there are two, and one of them is already on.

- `--reasoning-format none` (`common/arg.cpp:3707`) leaves thoughts unparsed inside `content`, so even a content-only client round-trips the thinking block for free. It changes the API shape for every client on the router, and it has **not** been tested.
- `--reasoning-preserve` (`common/arg.cpp:3761`) keeps the reasoning trace across the *whole* history rather than just the last assistant message, where the template supports it. It is wired on in `samm-mbp.ini`.

This model thinks by default, so a reply arrives split across `reasoning_content` and `content`. A client that appends only `content` to the conversation sends back an assistant turn missing its thinking block. The slot's generated tokens then stop being a prefix of the next request, and because the recurrent half cannot be trimmed to an arbitrary position the server restores a context checkpoint and re-prefills the answer. Every turn.

Measured with `sweep-chat.sh`, 5963-token prompt, `NGEN=1024`, stock build:

| client behaviour | steady-state per turn | `f_keep` |
|---|---|---|
| echoes `reasoning_content` + `content` | 20-23 tok / ~280 ms | **1.000** |
| echoes `content` only | 747-835 tok / ~1200 ms | 0.857-0.942 |

The gap is the answer length, so it grows with how much the model writes - at 160-token answers it is invisible (28 tok/turn, which is why an earlier pass here wrongly concluded there was nothing to fix), and at realistic agent answer lengths it dominates the per-turn cost.

`f_keep` in the server log is the diagnostic: **1.000 means the answer was kept**. Anything less means it is being re-prefilled, and nothing the API returns shows it. This is the same mechanism as [#28049](https://github.com/ggml-org/llama.cpp/issues/28049) reached by a different route - there the tail tokens come from `draft-mtp` leaving accepted tokens past the EOG, here from the client dropping the thinking block.

Unresolved: whether the server's `--reasoning-preserve` matters once the client does echo. Three turns gave 469 ms with it against 802 ms without, `f_keep` 1.000 in both, with per-turn counts too variable (387/861/23) to call. Needs more turns before anything is changed in the ini.

Also unpriced: echoing the thinking block keeps `f_keep` at 1.000, but it appends up to `reasoning-budget` (8192, wired in `samm-mbp.ini`) tokens of reasoning per turn to the history, and every later prefill and decode carries them. That cost has never been measured against the alternative of re-prefilling the answer. Both sides of the trade need a number before the current setting can be called optimal rather than merely better on the one metric that was checked.

**Action:** check what the router actually sends. If it forwards only `content`, fixing that is worth more than anything else measured in this document.

## Cached turn cost also depends on the message shape (2026-09-02, measured)

Separately from the thinking-block issue: a conversation grown as one ever-longer user message, with assistant turns never returned, diverges from the slot well before the end of its content and forces a checkpoint restore a full ubatch back - 2058 tok / 2393 ms per turn. Adding a third checkpoint offset, `{4 + n_ubatch, 64, 4}`, takes that to 74 tok / 476 ms. That patch **is** applied now (`patches/`), though for the session-start case rather than this one; no client here sends this shape. See [UPSTREAM-CANDIDATES.md](UPSTREAM-CANDIDATES.md).

Not a correctness concern: cached output already diverges from an uncached run on **stock upstream** (`verify-checkpoint.sh`, greedy - stock diverges at char 641, patched at char 700), the known batch-shape float-reduction effect.

## Methodology trap: copying the binary does not isolate the build (2026-09-02)

Read this before running any A/B that differs by *build* rather than by flags. `build/bin/llama-server` has exactly one `LC_RPATH`, an absolute path to `/Users/samm/git/llama.cpp-pr27836/build/bin`. Copying the binary, or even the whole `build/bin` directory, elsewhere still loads the dylibs from the live build directory - and the server logic lives in `libllama-server-impl.dylib`, the Metal kernels in `libggml-metal.dylib`. Both arms of such an A/B run whatever was last built.

This invalidated two results here before it was caught, and the tell was in the server log: a prefill breaking at `N-64` when the binary under test supposedly had no 64 offset. `bench-mtp.sh`'s two-binary interface is only safe for arms that differ by *flags*. To A/B two builds, rebuild in place between arms, or relink with `-Wl,-rpath,@loader_path` / build statically.

Consequences, both now corrected above: the `{4 + n_ubatch, 64, 4}` offset was first written off as null when it is a 5x win, and the `N_R0` test below compared a binary against itself.

**Untested, previously mis-reported as null:** `N_R0_IQ3_S` 4 to 8 and `N_R0_IQ4_NL` 2 to 4 in `ggml/src/ggml-metal/ggml-metal-impl.h`, the transferable half of PR #28086. The A/B that produced "pp 1044.75 to 1045.72, tg 49.46 to 49.45" loaded one `libggml-metal.dylib` for both arms and means nothing. Correctness did pass `test-backend-ops` on MUL_MAT and MUL_MAT_ID. Needs redoing with rebuilds in place.

## The model is not IQ4_XS (2026-09-02)

Worth knowing before anyone optimises for the name on the tin. Actual tensor-type composition:

| type | GB | share | where |
|---|---|---|---|
| IQ4_NL | 49.1 | 52.4% | `per_layer_token_embd` 28.8GB (get_rows, not matmul) + `ffn_down_exps` |
| IQ3_S | 33.9 | 36.2% | `ffn_gate_exps` + `ffn_up_exps` |
| Q8_0 | 8.9 | 9.5% | attention |
| IQ4_XS | 0.9 | 1.0% | two tensors |

"UD-IQ4_XS" is Unsloth's naming convention, not a description of the weights. Per-token expert traffic is IQ3_S 662 MB, IQ4_NL 396 MB, IQ4_XS 17 MB - so doubling IQ4_XS kernel speed would buy roughly 1.5% of decode.

IQ3_S is the largest share of expert traffic. It is also the slowest kernel *in the dense matvec shape*: `test-backend-ops perf -o MUL_MAT` at m=4096,k=14336 gives 2.38 / 2.44 / 2.56 / 2.55 TFLOPS at n=2/3/4/5, where IQ4_NL climbs 2.64 to 4.09 and Q4_K holds ~3.5. Treat that as background only - it is `MUL_MAT` at a shape this model does not run, which is the exact reading error corrected above. At the real MoE geometry IQ3_S is level with Q4_K and IQ4_NL from bs=8 upwards, which covers prefill and most of the MTP verify batch. It is ~25% behind them at bs=1 (1.68 against 2.26 and 2.23), and bs=1 is the single-token decode step, so a small IQ3_S-specific gap does exist in the narrowest decode case. It closes by bs=8 and is worth at most a few percent of decode, which is why the kernel work below is still not worth chasing - but the flat "IQ3_S is fine everywhere" reading is too strong.

IQ3_S also has the `ix = tiisg` idle-lane pattern that PR #28086 fixed for IQ3_XXS (`ggml/src/ggml-metal/kernels/mul_mv.metal:2249`), and at `ne00=2560` the guard `32 % nb32 == 0` does not fire, so a generalised `ntx = gcd(32, nb32)` split is still available. The `N_R0` half of that PR measured null here, so the split half would need measuring on its own before assuming it helps. (An earlier version of this line said `test-backend-ops` has no IQ3_S/IQ4_NL coverage for `MUL_MAT_ID` and that the MoE path could not be micro-benchmarked without adding cases. The cases were added and run - see the 512-expert table above.)

## Context depth does not shift the draft/verify split (2026-09-02, measured)

`src/models/qwen4exp.cpp:508` notes the MTP block attends densely while the trunk's QSA prunes past a 2048-token budget, with a `TODO: wire up QSA here`. That predicts the draft step growing linearly in context while the target stays flat, which would have made the "draft is only 8% of decode" finding an artefact of benchmarking at 6K. It does not happen. One load, prompts at increasing depth, per-context counters diffed between requests:

| prompt | draft ms/step | target ms/decode |
|---|---|---|
| 4K | 1.80 | 28.9 |
| 16K | 1.94 | 32.7 |
| 32K | 2.12 | 36.8 |

+18% draft against +27% target over 8x the context - the draft's share shrinks slightly with depth. The MTP block is one block of 48, so its dense attention is small next to the whole trunk, and the trunk pays its own O(n_ctx) scoring pass in `build_qsa_top_k`. Implementing QSA in the draft is not a throughput win.

Acceptance did fall over the same runs (0.773 / 0.759 / 0.718), which would matter since mean accepted length is the real lever. Do not trust it yet: the three prompts are different truncations of one source file, so content varies with depth and the comparison is confounded. Isolating it needs a fixed trailing instruction behind varying filler.

## Explored and ruled out (2026-08-29)

Save yourself the retry - all measured on this machine, controlled A/B where it matters. The experiment scripts were removed after the verdicts; the graft trick they used (rewrite the 11MB metadata shard, hardlink the weight shards) is the same one `graft-mtp-head.py` demonstrates.

- **Qwen3.5-4B as `-md` drafter**: vocab is identical (the 3.8 generation kept the 248320-token 3.5 tokeniser) and llama.cpp accepts the pairing, but cross-generation acceptance is only ~70%, and the target's 6B-active decode is too fast for a 4B to pay its way: 29-30 t/s vs 41.6 undrafted.
- **QSA in the MTP head via metadata** (compress_ratios[48]=4 in a variant shard): the draft graph is hard-wired dense ("v1 simplification" in `src/models/qwen4exp.cpp`), and the flip measured worse in controlled A/B (31.6 vs 35.6 at 16K).
- **Expert-reduced self-draft** (same weights, `expert_used_count` 10 to 4 via a rewritten metadata shard, mmap-shared pages): elegant on paper, dead on 128GB - Metal wires each model instance's full mapping, so target + draft = 2x 90GB wired and the first decode OOMs (`kIOGPUCommandBufferCallbackErrorOutOfMemory`). Viable only with ~256GB, or if llama.cpp ever shares one model instance between target and draft contexts with per-context hparams.
- **Downloading a trained EAGLE3/DFlash/DSpark draft**: none exists for qwen4exp (they exist for Qwen3.8-27B and others, wrong target). No small Qwen4-family sibling has been released.

- **PLE ngram table in a Metal buffer** (`-ot "per_layer_token_embd\.weight=MTL0"`, hoping to kill the per-token CPU gather+sync): wash at short context (66.4 vs 65.7 avg over 4 runs/arm), worse at 32K (30.0 vs 34.2), costs 27GB wired. Unified memory already makes the CPU-side gather cheap; default placement stands.

- **`spec-draft-n-min` above 0**: gating speculation on a minimum confident draft length measured monotonically worse (16K: 41.2/38.2/37.1 t/s at n-min 0/2/3) - at this acceptance rate a skipped spec round costs more than the verify batch it saves.

- **Lower temperature for draft acceptance** (temp 0.6 vs the model card's 1.0): the sharper-target-distribution-lifts-acceptance theory did not show up in throughput - short context straddles noise (67.3/75.3 vs 73.5), 32K measured worse (26.5 vs 32.3, single pair). Acceptance was thought to sit ~90% at the time (later corrected to 0.70-0.76), so the headroom looked small; the throughput conclusion stands either way. Temp stays 1.0. Sampler chain ordering was checked too: llama.cpp already runs top-k first (cheap 248K-vocab cull), nothing to reorder.

- **`ngram-cache` in the spec-type list**: measured on an agent-shaped file-rewrite task, it makes things worse (46.2 t/s current config, 38.0 with cache added, 34.0 replacing ngram-mod). Draftless ngram types take per-step priority, so the cache's frequency-ranked drafts displace the higher-quality MTP/ngram-mod drafts. A warmed static cache does not change the displacement problem.

- **Implementing the QSA TODO in the MTP draft graph** (attempted 2026-08-29, no viable minimal change): the MTP context gets a plain blk.48 `llama_kv_cache` and cannot cheaply obtain the indexer context QSA requires; switching it to `llama_memory_hybrid_idx` breaks the draft driver's per-cycle partial rollback (`llama_memory_recurrent::seq_rm` refuses partial tail removal with `n_rs_seq = 0`, silently corrupting draft state), and the msa cache alternative means a 400+ line refactor of the trunk QSA path. Decisive point: per-draft-step cost at 32K is dominated by the 248K-vocab LM head (~600MB weight read) + MoE (~50MB); dense attention over 32K KV is ~32MB, under 5% of the step, so the ceiling is low single digits of t/s - below our benchmark noise floor. If ever revisited, the smallest path is making zero-layer recurrent `seq_rm` succeed on partial tail removal, then a hybrid-idx MTP context and `build_layer_attn` in `graph_mtp` (~60-80 lines) - but the audit of recurrent cell bookkeeping for the zero-layer case is the real cost. Worth a comment on PR #27836 noting the rollback blocker and cost profile.

No levers left standing: the wired config (draft-mtp alone, n-max 6, p-min 0.7, backend sampling, preset build + #27992) is the measured optimum for this model in llama.cpp today. The next real speedup arrives from upstream (MTP draft-step batching, or a small Qwen4-family sibling model).
