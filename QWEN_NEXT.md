# Qwen3.8-Flash-Next MTP setup

Qwen3.8-Flash-Next (arch `qwen4exp`, 125B-A6B) ships a jointly-trained single-layer MTP draft head, but llama.cpp master deliberately drops the `mtp.*` tensors at conversion, so no public GGUF quant can self-draft. This repo works around that with a grafted head and a PR-branch build. Result on the M5 Max: 41.6 t/s plain, ~70 t/s with `draft-mtp` at ~90% draft acceptance.

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

## Benchmarks

M5 Max 128GB, UD-IQ4_XS, C++ source-code prompts (prose lands a few t/s lower with the same ordering). Current config on the preset (LTO) build. Benchmark convention from 2026-08-30 on: always pass `-ctk q8_0 -ctv q8_0` to match the served config (the router's global `cache-type-k/v = q8_0`); numbers before that date used f16 KV. Target-KV q8_0 is safe on this arch - the crash reports concern the draft KV quant only.

| context      | no spec | draft-mtp | speedup |
| ------------ | ------- | --------- | ------- |
| short (n=200)| 41.6    | ~70       | +68%    |

A UD-Q4_K_XL graft (`models/Qwen3.8-Flash-Next-MTP-Q4KXL-Merged-GGUF/`, q8_0 KV) measured 68.8 t/s short and 30.3 at 32K vs IQ4_XS's 70.1 / 34.1 (f16 KV) - parity short, ~11% slower at depth. Decode is bandwidth-bound, so the larger working set (~77 vs ~60GB) outweighs the faster Q4_K Metal kernels. It stays available as a quality option; IQ4_XS remains the speed default. Cold-cache warning: first load after a reboot fault-storms the SSD (single-digit t/s, OS lag) - warm it with a throwaway generation, and never benchmark two large quants alternately (page-cache churn hard-locked the machine once).

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

The single-layer head drafts recursively, so n-max is not capped at 1. It was trained for ~4-step drafting; 6 still wins on throughput, 8+ collapses because every rejected deep draft wastes a full verify batch. Server-side acceptance measured 96/106 (90%).

At context depth (llama-bench for PP and plain TG; llama-cli with source-code prompts for spec TG):

| depth | PP t/s | TG plain | TG ngram-mod | TG draft-mtp | TG mtp+ngram |
| ----- | ------ | -------- | ------------ | ------------ | ------------ |
| 4K    | 1010   | 39.2     | 38.5         | 44.3         | 44.3         |
| 16K   | 781    | 34.3     | 33.9         | 40.6         | 39.4         |
| 32K   | 632    | 29.4     | 30.3         | 34.1         | 36.1         |

TG spec columns are the preset (LTO) build; the earlier plain Release build measured 41.4/34.0/31.2 combined, so the preset build is worth ~+16% at depth (LTO helps the per-step CPU-side draft/verify orchestration). PP and TG-plain columns are still from the plain build and read slightly low. The MTP gain thins from +45% (short) to ~15-20% at depth because verify batches pay the full attention/KV cost per step. Chat workloads with higher acceptance land higher still.

**ubatch (2026-08-30, on the #27992 build):** `-ub 2048` lifts 32K prefill 570 -> 661 t/s (+16%; 1024 lands at 627) with TG unchanged, costing ~1.1GiB extra compute buffer (1.8GiB Metal at 2048 vs ~0.7 default). Wired as `ubatch-size = 2048` in both MTP model sections.

**Long-context perf PRs #27992 vs #27977 (2026-08-30):** qwen4exp's PLE lookup calls `get_prev_tokens()` every decode step, which upstream scans all used KV cells - a cost that grows with context. Two open PRs attack it: #27992 (per-seq kv-cell position index) and #27977 (scan early-exit + QSA gather windows for generation + indexer sum-shape fix + bitmap used-cells). Either one beats the plain PR build at depth (+18% TG at 74K in the first A/B). Head to head on M5 Max (both-orders, q8 KV, ub 2048): TG identical within noise at 32K/74K/115K, but #27977 wins PP consistently at every depth - 685 vs 663 (32K), 511 vs 488 (74K), 409 vs 387 t/s (115K). **Carrying both regressed TG ~-18% at 32K** (27.6/29.0 vs 34.8/34.4), so `update-mtp-build.sh` merges exactly one: #27977. The CUDA-side headline gains (2.7x at 240K) do not transfer to Metal - unified memory already makes the CPU scan cheap - but the PP win and the correctness-adjacent fixes are free. Swap back to #27992 only if #27977 stops merging cleanly.

At depth on novel code, draft-mtp alone and mtp+ngram-mod are within noise of each other (ngram-mod barely fires). Where they differ, ngram-mod loses: short context 70.1 vs 65.8 avg, echo-heavy agent task 51.8 vs 48.0 avg (both-orders A/B). Same mechanism as ngram-cache below - draftless ngram drafts take per-step priority, so wherever ngram-mod matches, its lower-acceptance drafts displace the 90%-acceptance MTP drafts. Hence `spec-type = draft-mtp` alone for this model; ngram-mod stays right for the models with no MTP head.

For calibration, the only published M5 Max 128GB numbers for this model (heretik.io) are tg 33 t/s and pp512 966 at depth 0 - this setup is ~20% ahead plain and ~2.1x ahead with MTP on the preset build (69.3 t/s) at short context. Ruled out as bottlenecks: flash attention (auto resolves on, ±1 t/s), the PR build itself (~1% vs master), and the lazy-read ngram table (`--tensor-read-lazy off` made no difference, matching PR #27794's claim). The ~40 t/s plain decode is what a 125B model with per-token ngram gathers does on this hardware.

The ngram table (`per_layer_token_embd`, 27.4GB in this quant) is lazy-read upstream (#27794): excluded from load prefetch, `madvise(RANDOM)`, pages fault in from SSD on demand and only hot rows stay cached. Verified here - a spec run's peak dirty footprint is 1.6GB and the full file is never resident. mlock defeats this by wiring the whole mapping, so the model's ini section sets `load-mode = mmap` (overriding the global `mmap+mlock`), keeping tens of GB unwired at no measured speed cost. That headroom, plus the fact that the whole mapping is evictable under pressure, is what makes a larger trunk quant (UD-Q4_K_XL, 104GB file) feasible on 128GB.

Two cautions from community reports: quantised KV cache crashes this arch (leave the commented `cache-type-*-draft = q8_0` lines in samm-mbp.ini commented), and very long contexts can exceed the default Metal wired limit (a 262K run needed `iogpu.wired_limit_mb` raised; our 131072 ctx-size has headroom but is worth remembering).

## Moving forward

- After pulling llama.cpp master, run `./update-mtp-build.sh` to bring the worktree along: it fetches the latest PR head, merges origin/master and PR #27977 (falls back gracefully when either stops merging cleanly), rebuilds, and exits early when nothing changed. It also detects the PR merging upstream and prints the retirement steps. It builds with the main repo's `local` cmake preset (LTO, native; symlinked into the worktree since `CMakeUserPresets.json` is untracked) and never runs `cmake --install`, so the PR binaries stay out of `~/.local`.
- Watch [PR #27836](https://github.com/ggml-org/llama.cpp/pull/27836). When it merges: update the normal llama.cpp install, delete the `LLAMA_SERVER_BIN` override in `samm-mbp.env`, and remove the worktree (`git -C ~/git/llama.cpp worktree remove ../llama.cpp-pr27836`). The grafted model keeps working on the merged code.
- The merged model does not load on master builds. If the server ever fails on it with "MTP" or `nextn` tensor errors, the binary is not the PR build.
- Related fixes to watch: [#27897](https://github.com/ggml-org/llama.cpp/pull/27897) (combined draft-mtp + external draft init, fetched locally as branch `pr-27897-fix`; not needed for our no-`-md` path) and [#27572](https://github.com/ggml-org/llama.cpp/issues/27572) (acceptance collapse with parallel slots `-np N`; retest before enabling parallel serving on this model).
- For the MBA or a new quant: rerun `graft-mtp-head.py <shard1> heads/Qwen3.8-Flash-Next-MTP-GGUF/Qwen3.8-Flash-Next-MTP-Q8_0.gguf <out-dir> <basename>`. The head pairs with any quant of the base model because it shares the trunk's embeddings and LM head.
- Once upstream converters keep the MTP tensors, freshly converted quants will carry nextn embedded and the graft (plus `heads/`) can be retired.
- Do not expect the 70-100 t/s that Qwen 3.5 122B-A10B managed with a 4B draft model: a dense drafter runs 8-16 tokens deep per verify, while the MTP head's recursion decays past its 4-step training depth (useful max ~6), and Flash-Next's verify step also pays for ngram gathers, a 248K-vocab output matmul and GDN state rollback on rejection. If a small Qwen4-family sibling with the 248K tokeniser ever appears, benchmark it against MTP.

## Explored and ruled out (2026-08-29)

Save yourself the retry - all measured on this machine, controlled A/B where it matters. The experiment scripts were removed after the verdicts; the graft trick they used (rewrite the 11MB metadata shard, hardlink the weight shards) is the same one `graft-mtp-head.py` demonstrates.

- **Qwen3.5-4B as `-md` drafter**: vocab is identical (the 3.8 generation kept the 248320-token 3.5 tokeniser) and llama.cpp accepts the pairing, but cross-generation acceptance is only ~70%, and the target's 6B-active decode is too fast for a 4B to pay its way: 29-30 t/s vs 41.6 undrafted.
- **QSA in the MTP head via metadata** (compress_ratios[48]=4 in a variant shard): the draft graph is hard-wired dense ("v1 simplification" in `src/models/qwen4exp.cpp`), and the flip measured worse in controlled A/B (31.6 vs 35.6 at 16K).
- **Expert-reduced self-draft** (same weights, `expert_used_count` 10 to 4 via a rewritten metadata shard, mmap-shared pages): elegant on paper, dead on 128GB - Metal wires each model instance's full mapping, so target + draft = 2x 90GB wired and the first decode OOMs (`kIOGPUCommandBufferCallbackErrorOutOfMemory`). Viable only with ~256GB, or if llama.cpp ever shares one model instance between target and draft contexts with per-context hparams.
- **Downloading a trained EAGLE3/DFlash/DSpark draft**: none exists for qwen4exp (they exist for Qwen3.8-27B and others, wrong target). No small Qwen4-family sibling has been released.

- **PLE ngram table in a Metal buffer** (`-ot "per_layer_token_embd\.weight=MTL0"`, hoping to kill the per-token CPU gather+sync): wash at short context (66.4 vs 65.7 avg over 4 runs/arm), worse at 32K (30.0 vs 34.2), costs 27GB wired. Unified memory already makes the CPU-side gather cheap; default placement stands.

- **`spec-draft-n-min` above 0**: gating speculation on a minimum confident draft length measured monotonically worse (16K: 41.2/38.2/37.1 t/s at n-min 0/2/3) - at 90% acceptance a skipped spec round costs more than the verify batch it saves.

- **Lower temperature for draft acceptance** (temp 0.6 vs the model card's 1.0): the sharper-target-distribution-lifts-acceptance theory did not show up in throughput - short context straddles noise (67.3/75.3 vs 73.5), 32K measured worse (26.5 vs 32.3, single pair). Acceptance already sits ~90%, so the headroom was small. Temp stays 1.0. Sampler chain ordering was checked too: llama.cpp already runs top-k first (cheap 248K-vocab cull), nothing to reorder.

- **`ngram-cache` in the spec-type list**: measured on an agent-shaped file-rewrite task, it makes things worse (46.2 t/s current config, 38.0 with cache added, 34.0 replacing ngram-mod). Draftless ngram types take per-step priority, so the cache's frequency-ranked drafts displace the higher-quality MTP/ngram-mod drafts. A warmed static cache does not change the displacement problem.

- **Implementing the QSA TODO in the MTP draft graph** (attempted 2026-08-29, no viable minimal change): the MTP context gets a plain blk.48 `llama_kv_cache` and cannot cheaply obtain the indexer context QSA requires; switching it to `llama_memory_hybrid_idx` breaks the draft driver's per-cycle partial rollback (`llama_memory_recurrent::seq_rm` refuses partial tail removal with `n_rs_seq = 0`, silently corrupting draft state), and the msa cache alternative means a 400+ line refactor of the trunk QSA path. Decisive point: per-draft-step cost at 32K is dominated by the 248K-vocab LM head (~600MB weight read) + MoE (~50MB); dense attention over 32K KV is ~32MB, under 5% of the step, so the ceiling is low single digits of t/s - below our benchmark noise floor. If ever revisited, the smallest path is making zero-layer recurrent `seq_rm` succeed on partial tail removal, then a hybrid-idx MTP context and `build_layer_attn` in `graph_mtp` (~60-80 lines) - but the audit of recurrent cell bookkeeping for the zero-layer case is the real cost. Worth a comment on PR #27836 noting the rollback blocker and cost profile.

No levers left standing: the wired config (draft-mtp alone, n-max 6, p-min 0.7, backend sampling, preset build + #27992) is the measured optimum for this model in llama.cpp today. The next real speedup arrives from upstream (MTP draft-step batching, or a small Qwen4-family sibling model).
