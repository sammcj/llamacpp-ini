# Upstream candidates

Local llama.cpp changes considered for a PR to ggml-org/llama.cpp, kept separate from QWEN_NEXT.md so the case for each can be judged on its own. What is applied to the build lives in `patches/` and is re-applied by `update-mtp-build.sh` on every rebuild.

Status legend: **proposed** (validated, worth a PR), **parked** (real effect, insufficient case), **dropped** (measured as not worth it).

---

## Extra context checkpoint offset - PROPOSED

**Status:** un-parked 2026-09-02 and **applied to the local build**. Not submitted yet.

It was parked because the shape that motivated it - a conversation grown as one ever-longer user message, with the assistant replies never sent back - is not what agents send. That reasoning was right about the shape and wrong about the conclusion, because a second shape does hit it and is entirely realistic: **a new agent session that shares a long prefix with an earlier one and diverges near the end.**

Measured with two prompts sharing a 28903-token prefix, differing only in their last ~20 tokens:

| offsets | second prompt |
|---|---|
| `{4 + n_ubatch, 4}` | 2047 tokens / 3605 ms |
| `{4 + n_ubatch, 64, 4}` | **63 tokens / 454 ms** |

8x. The 2047 is exactly `4 + n_ubatch`: the `-4` checkpoint sits past the divergence point and is rejected, so the only survivor is a full ubatch back. A 64-token offset lands clear of the chat template's generation prompt (~5 tokens) and is accepted.

Cost is one extra context checkpoint per prefill per slot, ~112 MiB of host state each, capped by `--swa-checkpoints`. 64 was the first value tried and remains unswept, but the remaining headroom is only ~48 tokens (~0.05 s), so sweeping it is not worth much. Before submitting, the case should be framed as bounding worst-case checkpoint distance on hybrid models, not as a speedup.

### The change

One line, at `checkpoint_offsets` in `tools/server/server-context.cpp`:

```diff
-  static const int checkpoint_offsets[] = {4 + n_ubatch, 4};
+  static const int checkpoint_offsets[] = {4 + n_ubatch, 64, 4};
```

The surrounding code is from [#20288](https://github.com/ggml-org/llama.cpp/pull/20288), which added the two-checkpoint scheme. No upstream PR or issue proposes adding a third offset - searched `checkpoint_offsets` across ggml-org/llama.cpp issues and PRs, one unrelated hit ([#28049](https://github.com/ggml-org/llama.cpp/issues/28049), see below).

### Why it looked worth having

llama-server creates context checkpoints during prompt processing at two offsets before the end of the prompt: `4 + n_ubatch` and `4`. On a cached follow-up request the `4` one is rejected whenever the request carries new tokens - `hparams.n_swa == 0` for `qwen4exp` makes `pos_min_thold == pos_next` (`:3270`, `:3323-3336`), and that checkpoint has already consumed the divergent token. The only remaining candidate is a full ubatch back.

This matters on a hybrid model specifically. The recurrent half cannot be trimmed to an arbitrary position, so when the slot is not a usable prefix the server has no option but to restore a checkpoint and replay forward.

A 64-token offset lands past the chat template's generation prompt (`<|im_end|>\n<|im_start|>assistant\n`, ~5 tokens) and is accepted.

### Measured, on the shape that motivated it

Growing single-user-message shape, 5963-token prompt, real rebuilds between arms (see the rpath note in QWEN_NEXT.md - copying the binary does not isolate a build):

| offsets | per-turn prefill | cold prefill |
|---|---|---|
| `{4 + n_ubatch, 4}` | 2058 tok / 2393 ms | 5812 ms |
| `{4 + n_ubatch, 64, 4}` | 74 tok / 476 ms | 6030 ms |

5.0x, cold prefill unchanged.

### History, superseded

Everything below this heading was written while the patch was parked, and the measurement above overrides its conclusion. It is kept only for the harness bug it records, which is worth remembering.


The original reasoning was that this shape is not what agents send, which held for multi-turn traffic (append-only, `f_keep` 1.000, nothing for the extra checkpoint to improve) but missed the session-start case entirely.

That shape is not what agents send. Growing a single user message and never returning the assistant's replies makes each request diverge from the slot well before the end of its content. A chat client appends turns instead, and `sweep-chat.sh` models that.

The first version of that comparison was itself wrong, and the correction matters more than the patch. It read only `.choices[0].message.content`, and this model thinks by default, so it echoed an **empty** assistant turn every round and reported 28 tok / 346 ms for both arms - which was read here as "stock already reuses the prefix almost completely". It was not: `f_keep` 0.974 was the discarded answer, and a checkpoint restore was happening every turn. With the harness fixed to echo `reasoning_content` as well, stock reaches `f_keep` 1.000 and 20-23 tok / ~280 ms per turn. See QWEN_NEXT.md.

The parking decision was then kept, for a different reason than first given - superseded by the session-start measurement above, which the argument below does not cover. With the answer round-tripping there is no restore for the extra checkpoint to improve on. Without it, the divergence sits at the end of the previous prompt, where the existing `-4` checkpoint is already the nearest one and a 64 offset changes nothing. Either way the 5x is confined to clients that mutate or extend an earlier message rather than appending.

Two corrections to the mechanism paragraph above, from review: the `4` offset is **not** rejected whenever a request carries new tokens - the chat data shows it being accepted (28 = 24 new + 4 replayed). Rejection needs the divergence point to fall inside the last 4 tokens of the previous prompt. The `n_swa == 0` claim holds but is not what makes the checkpoint fail; divergence position is.

Against that, each context checkpoint on this model is ~112 MiB of host state (36 GDN layers; `n_embd_r` 30720 + `n_embd_s` 786432 f32 per layer, plus PLE conv state), and this adds one per prefill per slot.

### Still open

The `f_keep` 0.974 gap in append-only traffic at 160-token answers was never chased down and may widen at longer answer lengths.

---

## Metal `mul_mv_ext` batch bound caps speculative decoding at width 8 - INVESTIGATE

Not a proposed patch yet: the cliff is measured, the fix is not.

`ggml/src/ggml-metal/ggml-metal-ops.cpp:2494` in `ggml_metal_op_mul_mat` gates the `mul_mv_ext` fast path to `ne11 >= 2 && ne11 <= 8` for block-32 types, and `ne11 >= 4 && ne11 <= 8` for K-quants. At `ne11 = 9` the dense projections and the LM head drop onto the generic kernel.

Measured on Qwen3.8-Flash-Next, f16 KV, depth 8192, r=8, ms per batch:

| n | 7 | 8 | 9 | 10 | 16 |
|---|---|---|---|---|---|
| ms | 58.07 | 61.11 | **139.71** | 142.49 | 161.63 |

A +78 ms step at exactly the bound. It is a step in fixed cost, not a change in scaling: the local marginal cost per token is 3.04 ms just below the bound (58.07 to 61.11) and 3.1 ms above it (10 to 16). Both kernels scale the same way; the generic one simply starts ~78 ms higher.

Why this matters beyond this model: every speculative decoder on Metal submits a verify batch of `n_draft + 1`. The bound therefore caps useful draft width at 7 on any model and any backend-sampling scheme, and it does so invisibly - it presents as "longer drafts measure worse", which reads like an acceptance problem. On this model it is the entire reason `n-max 6` benchmarks as optimal and `n-max 8` costs 14%. That optimum is a kernel artefact.

What would need doing before this is a PR:

- Establish why the bound is 8. The surrounding code carries a `TODO: determine the optimal parameters based on grid utilization` and an explicit "I still don't know why we should not always use the maximum available threads", so 8 may be tuning inherited rather than a hardware limit.
- Measure `mul_mv_ext` with the bound raised to 16 and 32, for correctness first and then speed. If it degrades above 8 for a real reason, the fix is instead a mid-range path for 9-32 rather than widening this one.
- Check the K-quant lower bound of 4 while there. Nothing here explains why K-quants need `ne11 >= 4` when block-32 types manage from 2.

## Related upstream work, not ours

- **[#28049](https://github.com/ggml-org/llama.cpp/issues/28049)** (open issue, no PR as of 2026-09-02) - `draft-mtp` leaves accepted tokens past the EOG token in the slot, so the slot stops being a prefix and a hybrid model re-prefills the whole previous answer. The author proposes a 7-line fix cutting `accepted` at the first EOG before `n_rollback` is computed. Applied and measured here as no change - but that test **could not have detected anything**, for three reasons: with thinking on and a 160-token cap the generation never reaches EOG so the loop never fires; the harness was echoing an empty assistant turn, so answer reuse could not happen either way; and the issue's repro ran `enable_thinking:false`, the opposite configuration. Treat the null as void, not as evidence against the fix. Superseded by #28232 below, which is the same fix as a real PR and is now in the build. Note also that #28049's body cites `checkpoint_offsets` directly and explains the `+4` as this same mechanism, so it is not an unrelated hit.
- **[#28040](https://github.com/ggml-org/llama.cpp/pull/28040)** (merged 2026-09-01) - `get_prev_tokens` from O(n) to O(log n), benchmarked by its author on Qwen3.8-Flash-Next at +4.9% generation at 71k. Not in our build; cherry-picks with two conflicts, both resolvable to upstream's side, kept as a stash in the worktree. Measured here 2026-09-02 against cold prefill with `bench-prefill.sh`, three distinct ~32k prompts, rebuild in place: **701.7 vs 702.6 tok/s, no effect**. The author reports the same ("pp unchanged"), so this is a confirmation rather than a contradiction - the win is decode-side and still untested on its own terms here. See QWEN_NEXT.md for what does bound prefill.
- **Upstream scan 2026-09-02**, 24 commits between our HEAD and `origin/master`. Nothing addresses our measured bottlenecks. Two are templates for work that would apply here: **[#25952](https://github.com/ggml-org/llama.cpp/pull/25952)** fuses the MoE weighted-expert reduction on CUDA and **[#28202](https://github.com/ggml-org/llama.cpp/pull/28202)** does MUL_MAT/MUL_MAT_ID fusion on Hexagon; the Metal equivalent would target `ffn_moe_weighted`, which the per-op profile puts at ~2%. **[#27449](https://github.com/ggml-org/llama.cpp/pull/27449)** fixes IQ3_S mat-vec at batch >4 on Vulkan, which is the shape our MTP verify batches use - but Metal shows no equivalent problem, IQ3_S measuring 1.93 TFLOPS at bs=8 against Q4_K's 2.00. **[#27883](https://github.com/ggml-org/llama.cpp/pull/27883)** fixes Metal autoreleasepool leaks and is worth having on a long-running server. The M5 Max already has fa-vec tunings upstream, so that gap is closed.
- **[#28022](https://github.com/ggml-org/llama.cpp/pull/28022)** (open) - `--sleep-preserve-cache`. The one that mattered. Merging it alone changes nothing, because the flag is off by default: measured 38.2 s after a wake with the PR in and no flag, then **0.1 s prefill / 3.2 s wall** with it. Now passed by `run-llama-server.sh` behind a `--help` probe, since the stock binary on the other machine would reject the flag outright.
- **[#28232](https://github.com/ggml-org/llama.cpp/pull/28232)** (open) - the EOG truncation fix for #28049, this time as a real PR. In `EXTRA_PRS` as a correctness fix: accepted tokens left past EOG stop the slot being a prefix, so the next turn re-prefills the previous answer. **Still not measured.** A `bench-decode.sh` run at `NGEN=400` came back at tg 55.8 / acceptance 0.716 / mean len 3.75, identical to baseline - but that script sends three independent single-message requests and never sends a second turn, so there is no next-turn prefill for the fix to affect. Second void null on the same issue, for a second reason. A real test needs a two-turn conversation where turn 2 echoes turn 1's completed answer, comparing `f_keep` and prompt tokens on turn 2.
- **[#28213](https://github.com/ggml-org/llama.cpp/pull/28213)** (open) - gathers the ~2048 indexer-selected KV entries for QSA decode instead of masking the full cache. The author measures +6% at 31k and +50% at 130k on dual A6000. On Metal it is a small **regression** up to 32k, A/B'd in one binary via its own `QWEN4EXP_QSA_GATHER` switch with `llama-bench -p 0 -n 32`: 40.51 vs 40.66 at d4096, 35.99 vs 37.07 at d16384, 32.97 vs 33.31 at d32768, against a ±0.25 spread. Deeper results below. A server-level A/B first showed +2.5%, but acceptance moved with it (0.762 to 0.782) - at temp 0 the two paths still generate different tokens past ~150, so speculative throughput cannot attribute anything. That is why the measurement moved to `llama-bench`, which has no draft model to confound it.
- **[#28092](https://github.com/ggml-org/llama.cpp/pull/28092)** (open) - `--cache-disk`, a prompt cache that persists to a directory and reloads on start. Covers the case #28022 does not: a restart, which every rebuild of this worktree causes. Four-phase run of `bench-sleep-cache.sh`, same 28903-token prompt: without it the fourth step is a full **39.7 s**, with it **0.09 s prefill**. Two caveats on that figure, one of which turned out not to matter. The restore runs inside `server_prompt_cache`'s constructor, before the port opens, so it is not in the prefill number at all - the script now times startup separately for that reason, and it is not hiding anything: **9.0 s to first listen on the cold start against 6.0 s on the restart that restored the cache**. The one that does matter: `prompt_save` runs on a new task, on LRU slot eviction, or on entering sleep, never on shutdown, so what this covers is a restart *after* a sleep or a later request, not a restart straight after the last one.

Cold prefill measured 707.4 tok/s with it against 682.3 without, back to back. Single sample each, ordering uncontrolled, on a machine that had been running the model for an hour - enough to rule out a large cost, not enough to call it free. About 900 MiB per ~33k prompt, capped at 32 GiB.

Set **per model** by `sync-models.py`, not in the launcher. The router builds each child's args from its own argv (`server_models` constructs `base_preset` from `argc/argv`, and `--cache-disk` is not in `unset_reserved_args`), so a single path given to the router reaches every child - and a child deletes every entry in its directory whose cache key is not its own. One shared path means loading a second model wipes the first model's cache. Confirmed in source: on startup `server_prompt_cache_read_metadata` returns false when the stored key is not the server's own, and the caller responds with `server_prompt_cache_remove_files` (`tools/server/server-task.cpp`, the restore loop).

Restricted to Qwen 3.8 models 2026-09-03 (`cache_disk_models` in `sync-models.py`, matched as a case-insensitive substring so the `mtp-` and `-no-mmproj` variants come along). The cap is **per model**, so enabling it for all 17 put ~544 GiB of potential disk use behind a feature only the big slow models benefit from - a small model reloads faster than its cache restores. Now 7 models, ~224 GiB worst case. Note `--cache-disk` exists **only** in #28092 and is not in upstream master, so if that PR is ever dropped from `EXTRA_PRS` these ini keys become dead config that a stock binary rejects.
  - Needs two local fixes. It conflicts with #28022 in `server-context.cpp`, resolved here by keeping #28092's disk/RAM split, #28022's short-write checks on the RAM branch (using #28092's `discard()`), and #28022's sleep-preserve guard around the cache rebuild; rerere replays it. And it does not compile on macOS: `key << modified.time_since_epoch().count()` is ambiguous because libc++'s `file_clock` has an `__int128` rep, fixed with a `(long long)` cast. **Both reported upstream 2026-09-03**: [comment on #28092](https://github.com/ggml-org/llama.cpp/pull/28092#issuecomment-5517314832) with the macOS numbers and the shared-directory note, and the cast submitted as a PR into the author's own branch, [yitizi/llama.cpp#1](https://github.com/yitizi/llama.cpp/pull/1) (base `pr2-standalone`). The cast is now upstream: the author force-pushed #28092 to `6deebd45f` carrying it as `static_cast<long long>`, at which point our half stopped applying and `update-mtp-build.sh` aborted with "local patch no longer applies". Dropped from the patch the same day; only the checkpoint-offset hunk remains. The conflict fix is still needed and is still replayed by rerere.
- **[#25788](https://github.com/ggml-org/llama.cpp/pull/25788)** (open) - Metal `gated_delta_net` cache fusion, mirroring the CUDA path: the kernel writes recurrent-state snapshots straight into the KV cache rather than doing a per-layer `cpy`. 36 of our 48 trunk layers are GDN, and it is the first thing tried here that helps both halves. `llama-bench -p 0 -n 32`, against the same baseline as #28213:

| depth | baseline | #25788 | |
|---|---|---|---|
| 4096 | 40.66 | 42.14 | +3.6% |
| 16384 | 37.07 | 38.43 | +3.7% |
| 32768 | 33.31 | 34.40 | +3.3% |
| 65536 | 27.40 | 28.14 | +2.7% |
| 131072 | 20.13 | 20.53 | +2.0% |

`bench-prefill.sh` over three ~32k prompts: **718.6 tok/s against 702.6, +2.3%**. In `EXTRA_PRS`.
- **[#28098](https://github.com/ggml-org/llama.cpp/pull/28098)** (merged 2026-09-03) - Metal sparse Flash Attention: gathers the finite mask entries into per-row index lists and runs the vec kernels over those rows only, instead of masking the whole cache. The author's DSv4 numbers on M2 Ultra are 1.5x prefill at 16k and 3x at 65k. The kernel arrives through `origin/master`, but the PR leaves `qwen4exp` passing `n_kv_max = 0` behind a "TODO: enable sparse attention when we are ready", so on its own it does nothing for this model. `patches/0002-qwen4exp-sparse-fa.patch` flips that one line to `top_k->ne[0]`. Measured 2026-09-03, `bench-prefill.sh`, three ~33k cold prompts, rebuilt in place, browser tests running alongside: **760 vs 664 tok/s cold prefill, +14%**, with the sparse arm flat at 757-763 across all three prompts where the baseline drifted 692 to 630. `llama-bench -p 0 -n 32`, q8_0 KV: 39.42 ± 0.24 vs 35.00 ± 5.24 at d16384 and 35.14 ± 0.29 vs 34.50 ± 0.48 at d32768, so no decode regression and possibly a small gain. At depth the prefill gap widens: `llama-bench -p 2048 -n 32 -d 65536,131072 -r 2`, pp2048 **388 -> 702 t/s at d65536 (1.81x)** and **340 -> 589 at d131072 (1.73x)**, tg32 28.3 -> 28.7 and 20.6 -> 20.7. Temp-0 generation over a 36k-token prompt produced the same text on both builds. This is a larger prefill effect than the ~5% attention share in the per-op profile predicts, because the sparse path also skips the dense F16 KV dequant pass and the full-width mask multiply, neither of which the profile attributes to attention. Contrast #28213 above, whose gather sits on the decode side only and lost on Metal.
- **[unsloth/llama.cpp#142](https://github.com/unslothai/llama.cpp/pull/142)** - lets an MTP sidecar borrow the target's `token_embd`/`output`/`output_norm`. No benefit here; our graft already has that property. See QWEN_NEXT.md.
