# Local patches

Applied to the PR #27836 worktree by `update-mtp-build.sh`, after the `EXTRA_PRS` merges and before the build. They live here rather than as uncommitted edits or stashes because `checkout -B` rebuilds the branch from the PR head on every run and would discard either one - silently, in the case of the checkpoint offset.

Regenerate after changing them by hand:

    git -C ~/git/llama.cpp-pr27836 diff -- tools/server/server-context.cpp > patches/0001-server-context-local.patch

## 0001-server-context-local.patch

Two changes, both in `tools/server/server-context.cpp`.

**A third context checkpoint at 64 tokens.** `checkpoint_offsets` upstream is `{4 + n_ubatch, 4}`, so a prefix that diverges anywhere inside a ubatch falls back to replaying the whole ubatch. On the shape a new agent session actually produces this measured 2047 tokens / 3.6 s against 63 tokens / 0.45 s with the extra offset. See UPSTREAM-CANDIDATES.md.

**A `(long long)` cast on `last_write_time().count()`** in PR #28092's disk cache key. libc++'s `file_clock` has an `__int128` rep on macOS and no `operator<<` accepts it, so without this the server does not compile at all. Submitted upstream 2026-09-03 as [yitizi/llama.cpp#1](https://github.com/yitizi/llama.cpp/pull/1), a PR into #28092's own branch (`pr2-standalone`). Delete this half once that merges and `update-mtp-build.sh` picks up the new head.
