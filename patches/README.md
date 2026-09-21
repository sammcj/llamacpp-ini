# Local patches

Applied to the PR #27836 worktree by `update-mtp-build.sh`, after the `EXTRA_PRS` merges and before the build. They live here rather than as uncommitted edits or stashes because `checkout -B` rebuilds the branch from the PR head on every run and would discard either one - silently, in the case of the checkpoint offset.

Regenerate after changing them by hand:

    git -C ~/git/llama.cpp-pr27836 diff -- tools/server/server-context.cpp > patches/0001-server-context-local.patch
    git -C ~/git/llama.cpp-pr27836 diff -- src/models/qwen4exp.cpp > patches/0003-qwen4exp-mtp-hc-head-norm-shape.patch

## 0001-server-context-local.patch

One change, in `tools/server/server-context.cpp`.

**A third context checkpoint at 64 tokens.** `checkpoint_offsets` upstream is `{4 + n_ubatch, 4}`, so a prefix that diverges anywhere inside a ubatch falls back to replaying the whole ubatch. On the shape a new agent session actually produces this measured 2047 tokens / 3.6 s against 63 tokens / 0.45 s with the extra offset. See UPSTREAM-CANDIDATES.md.

Retired 2026-09-03: a `(long long)` cast on `last_write_time().count()` in PR #28092's disk cache key, without which the server did not compile on macOS at all (libc++'s `file_clock` has an `__int128` rep and no `operator<<` accepts it). Submitted upstream as [yitizi/llama.cpp#1](https://github.com/yitizi/llama.cpp/pull/1) into #28092's own branch; the author force-pushed it in as `static_cast<long long>` (head `6deebd45f`), at which point our half stopped applying and had to be dropped by hand. The PR was closed as superseded on 2026-09-07. A force-push to a PR that a local patch touches is the failure mode to expect here.

## 0002-qwen4exp-sparse-fa.patch (retired 2026-09-22)

Passed `top_k->ne[0]` as `n_kv_max` to `build_attn_mha` in `src/models/qwen4exp.cpp`, so the QSA layers ran the sparse Flash Attention kernel from [#28098](https://github.com/ggml-org/llama.cpp/pull/28098) instead of masking the whole cache. Measured 2026-09-03: **760 vs 664 tok/s cold prefill (+14%)**, pp2048 388 -> 702 t/s at d65536, decode unchanged. Upstream [#28770](https://github.com/ggml-org/llama.cpp/pull/28770) (merged 2026-09-20 as `3cf03257f`) made the identical one-line change, so it now arrives through `origin/master` and the patch was dropped. Full numbers stay in UPSTREAM-CANDIDATES.md.

## 0003-qwen4exp-mtp-hc-head-norm-shape.patch

One line, in `src/models/qwen4exp.cpp`.

**MTP head norm shape after upstream #28896.** Upstream [#28896](https://github.com/ggml-org/llama.cpp/pull/28896) (merged 2026-09-16) changed the trunk's hyper-connection gammas (`hc_attn_norm`, `hc_ffn_norm`, `hc_head_norm`) from flat `{hc_dim}` to `{n_embd, hc}` with `TENSOR_ALLOW_RESHAPE`, so `build_hc_mix` can multiply the 3D `[n_embd, hc, T]` stream directly and fuse rms_norm + mul. PR #27836's `nextn.hc_head_norm` is created by its own code and stayed `{hc_dim}`, so the MTP graph aborts at context creation with `GGML_ASSERT(ggml_can_repeat(b, a))` inside `build_hc_mix`. This patch creates it as `{n_embd, hc}` to match. `nextn.hnorm` stays flat: the MTP graph reshapes to 2D before that multiply. Drop this patch once #27836 rebases past #28896.
