# Local patches

Applied to the PR #27836 worktree by `update-mtp-build.sh`, after the `EXTRA_PRS` merges and before the build. They live here rather than as uncommitted edits or stashes because `checkout -B` rebuilds the branch from the PR head on every run and would discard either one - silently, in the case of the checkpoint offset.

Regenerate after changing them by hand:

    git -C ~/git/llama.cpp-pr27836 diff -- tools/server/server-context.cpp > patches/0001-server-context-local.patch
    git -C ~/git/llama.cpp-pr27836 diff -- src/models/qwen4exp.cpp > patches/0002-qwen4exp-sparse-fa.patch

0002 and 0003 both touch `src/models/qwen4exp.cpp`; regenerate them one hunk at a time (stage the other first, then `git diff` gives only the unstaged hunk).

## 0001-server-context-local.patch

One change, in `tools/server/server-context.cpp`.

**A third context checkpoint at 64 tokens.** `checkpoint_offsets` upstream is `{4 + n_ubatch, 4}`, so a prefix that diverges anywhere inside a ubatch falls back to replaying the whole ubatch. On the shape a new agent session actually produces this measured 2047 tokens / 3.6 s against 63 tokens / 0.45 s with the extra offset. See UPSTREAM-CANDIDATES.md.

Retired 2026-09-03: a `(long long)` cast on `last_write_time().count()` in PR #28092's disk cache key, without which the server did not compile on macOS at all (libc++'s `file_clock` has an `__int128` rep and no `operator<<` accepts it). Submitted upstream as [yitizi/llama.cpp#1](https://github.com/yitizi/llama.cpp/pull/1) into #28092's own branch; the author force-pushed it in as `static_cast<long long>` (head `6deebd45f`), at which point our half stopped applying and had to be dropped by hand. The PR was closed as superseded on 2026-09-07. A force-push to a PR that a local patch touches is the failure mode to expect here.

## 0002-qwen4exp-sparse-fa.patch

One line, in `src/models/qwen4exp.cpp`.

**Sparse Flash Attention on the QSA layers.** Upstream [#28098](https://github.com/ggml-org/llama.cpp/pull/28098) (merged 2026-09-03) added the Metal kernel that gathers the indexer-selected KV rows and runs FA over those alone, but left the `qwen4exp` call passing `n_kv_max = 0` behind a "TODO: enable sparse attention when we are ready". This patch passes `top_k->ne[0]` instead, which is what the commented-out line above it already does. Measured 2026-09-03 with `bench-prefill.sh`, three ~33k cold prompts, rebuilt in place, browser tests running in the background: **760 vs 664 tok/s cold prefill (+14%)**, and every prompt landing at 757-763 where the baseline drifted 692 to 630. Decode at depth with `llama-bench -p 0 -n 32` does not regress (35.1 vs 34.5 at d32768; 39.4 vs 35.0 ± 5.2 at d16384). At depth, pp2048 goes 388 -> 702 t/s at d65536 and 340 -> 589 at d131072, with tg32 unchanged. Temp-0 output on a 36k-token prompt matched across the two builds. See UPSTREAM-CANDIDATES.md. Drop this patch the day upstream uncomments the line itself.

## 0003-qwen4exp-mtp-hc-head-norm-shape.patch

One line, in `src/models/qwen4exp.cpp`.

**MTP head norm shape after upstream #28896.** Upstream [#28896](https://github.com/ggml-org/llama.cpp/pull/28896) (merged 2026-09-16) changed the trunk's hyper-connection gammas (`hc_attn_norm`, `hc_ffn_norm`, `hc_head_norm`) from flat `{hc_dim}` to `{n_embd, hc}` with `TENSOR_ALLOW_RESHAPE`, so `build_hc_mix` can multiply the 3D `[n_embd, hc, T]` stream directly and fuse rms_norm + mul. PR #27836's `nextn.hc_head_norm` is created by its own code and stayed `{hc_dim}`, so the MTP graph aborts at context creation with `GGML_ASSERT(ggml_can_repeat(b, a))` inside `build_hc_mix`. This patch creates it as `{n_embd, hc}` to match. `nextn.hnorm` stays flat: the MTP graph reshapes to 2D before that multiply. Drop this patch once #27836 rebases past #28896.
