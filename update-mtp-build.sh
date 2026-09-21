#!/usr/bin/env bash
set -euo pipefail

# Update the llama.cpp PR #27836 worktree build that serves Qwen3.8-Flash-Next
# MTP (see QWEN_NEXT.md). Fetches the latest PR head, merges in origin/master
# so the worktree keeps pace with daily master pulls, rebuilds, and tells you
# when the PR has merged upstream so the whole arrangement can be retired.
#
# Usage: ./update-mtp-build.sh    (nothing changed: prompts to rebuild anyway on a
#                                  terminal, skips silently when non-interactive)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${LLAMA_REPO:-${HOME}/git/llama.cpp}"
WORKTREE="${LLAMA_MTP_WORKTREE:-${HOME}/git/llama.cpp-pr27836}"
PR=27836
BRANCH="pr-${PR}-qwen4exp-mtp"
PR_REF="refs/pr/${PR}"
# Extra PRs merged on top of the base, in order. Stacking is NOT free - #27992
# plus #27977 together regressed TG ~18% at 32K - so each entry earns its place
# with a both-orders A/B before it is added, and the reason lives beside it.
#   28022 - preserves the prompt cache across an idle sleep. Without it, waking
#           frees every cached prefix and the next prompt is fully cold: measured
#           at 28903 tokens / 38.1 s after a sleep against 4 tokens / 0.08 s on a
#           warm repeat. This is the single largest agent-workload win found.
#   28232 - truncates speculative results at EOG before rollback, so draft-mtp
#           stops leaving accepted tokens past the end-of-generation token in the
#           slot (upstream issue #28049). Retest of an earlier void null.
#   28092 - --cache-dir (named --cache-disk until 2026-09-14): the prompt cache
#           persists to disk and reloads on start, covering the case #28022 does
#           not - a restart or reboot, which is what every rebuild of this
#           worktree causes. Its head moves; when it does, the rerere resolution
#           against 28022 goes stale and the script builds WITHOUT it, which
#           the router then reports as "option 'cache-dir' not recognized".
#           Resolve once by hand (keep 28092's disk/RAM split, put 28022's
#           short-write checks on the RAM branch using discard()).
#   28473 - fixes draft-mtp cross-slot content contamination with --parallel > 1
#           (upstream issue #28286). samm-mbp.ini runs parallel auto (4 slots), so
#           this is a correctness fix, not a speed one. Output stays plausible when
#           it misfires, which is why nobody noticed.
#   28333 - zeroes the MTP carrier at sequence start. Without it the carrier
#           persists between requests and identical deterministic requests can
#           produce different tokens, which invalidates paired A/B runs against a
#           long-lived server. Five lines.
#   28305 - keeps the backend sampling subgraph a fixed shape across ubatches. The
#           spec-dec verify step samples the whole accepted draft window, so the
#           row count varies per verify and can trigger a ggml-alloc realloc; we
#           run spec-draft-backend-sampling = 1. Measured 2026-09-08 with
#           bench-decode.sh ABAB: null (tg 57.0/56.9/56.8/56.5). Kept as harmless.
#   28439 - Metal flash-attn wide query tile (8 -> 16 rows) when ne01 >= 64 and
#           head size pads to a multiple of 128. Our DK=DV=256 hits the gate. The
#           author's M5 numbers are -34% at 16K and -47% at 64K KV on hs 256, but
#           the vec and sparse paths are untouched, so the sparse-FA QSA layers
#           gain nothing and attention is ~5% of prefill here. Measured as part of
#           the four-PR arm 2026-09-08: null at 33k cold prefill (see
#           UPSTREAM-CANDIDATES.md), and null again at d65536/d131072 with
#           llama-bench. The sparse path carries our attention at every depth.
#           Kept as harmless; first to drop if it ever conflicts.
#   28007 - falls back to full reprocessing when hybrid seq_rm refuses a rollback
#           past the RS ring instead of aborting the server (upstream issue #27931).
#           13 lines, safety only.
#   28785 - ggml-cpu skips the threadpool wake-up when the graph has no CPU work,
#           which is every graph at -ngl 999. ggml-cpu.c only. Measured 2026-09-13
#           as part of the four-PR arm: null on short decode (57.2 vs 56.7-57.4)
#           and on cold prefill (777.9 vs 783.2 tok/s). Kept as harmless.
#   27694 - probabilistic drafter with rejection sampling for draft-mtp, opt-in via
#           --spec-draft-sampling probabilistic (set per-model in samm-mbp.ini).
#           Lossless on the output distribution. Three paired runs at temp 1.0:
#           acceptance 0.698 -> 0.735, mean len 3.50 -> 3.55, tg +1.2%. Small
#           because p-min 0.7 already cuts the chain. Conflicts with 28473 in
#           common/speculative.cpp (one hunk, rerere-resolved).
#   28699 - incremental pooled-key cache for the QSA indexer via set_rows, so each
#           QSA layer stops regathering the whole context per decoded token. The
#           largest decode win in this file. Same-binary A/B via its kill switch
#           LLAMA_QSA_NO_POOLED_CACHE=1, llama-bench tg32: 34.8 -> 40.3 at d32768
#           (+16%), 28.1 -> 37.6 at d65536 (+34%); MTP decode at 37k depth 53.0 ->
#           57.4 t/s (+8%); cold prefill and 4k decode null; greedy output at 32k
#           depth identical. Draft PR with an open n_dirty assert on image input,
#           which text-only serving never hits.
# Candidates not yet taken: 27210 (adaptive MTP draft depth) conflicts with 28473
# in common/speculative.cpp and needs spec-draft-n-max >= 7 (we run 5), so it is
# a retune, not a drop-in. 25592 (hybrid checkpoint validity) rewrites the same
# checkpoint-selection predicate 28092 does; semantic conflict, parked.
# Merged upstream, so they now arrive through origin/master and are no longer
# listed: 27941 (qwen4exp follow-up fixes) and 28121 (ssm_a/ggml_scan flag), both
# squash-merged 2026-09-01; 28330 (indexer KV cache drops its unused V half,
# 612 -> 204 MiB at 131072 ctx), squash-merged 2026-09-10 as 311d4211b. A squash
# lands the code under a new SHA, so the ancestry check below never fired for
# any of them and they were being re-merged on every run; the GitHub state check
# is what caught them.
# Dropped: 27977 (closed upstream). 25788 (Metal gated_delta_net cache fusion,
# closed 2026-09-12): its hunk landed on master through ggerganov's 28164
# (single-source fusion table, merged 2026-09-11 as a2878d30d), which was never
# carried here. 28136 (--lazy-mode on-direct) - null on Metal
# at both a 5K and a 32K prompt, and needs a hand-applied ple_w compile fix every
# run. 28213 (QSA gather) - the author's +6% at 31k and +50% at 130k are CUDA; on
# Metal it measured slightly negative to 32k, null at 64k and +1.4% only at 128k,
# so it loses at the depths this machine actually runs. 28301 (Metal mul_mm_id
# half-tile skip) - costs 4.1% prefill and 5.1% decode here, reproducibly. 28118
# (on-device speculative checkpoints) - null on Metal and it aborts the server on
# the first cached follow-up. See QWEN_NEXT.md.
EXTRA_PRS=(28022 28232 28092 28473 28333 28305 28439 28007 28785 27694 28699)
MARKER="${WORKTREE}/.last-mtp-build"

die() {
  echo "Error: ${1}" >&2
  exit 1
}

# Ancestry against origin/master cannot tell a PR that was closed without merging
# from one that is still open - both stay non-ancestors forever - so a dead PR
# keeps being merged in silently (#27977 did exactly that). Ask GitHub instead.
# Best-effort: no gh, no auth or no network leaves every state empty and the rest
# of the script behaves as before.
REPO_SLUG="$(git -C "${REPO}" remote get-url origin 2>/dev/null \
  | sed -E 's#^.*github\.com[:/]##; s#\.git$##')"

pr_state() {
  [[ -n "${REPO_SLUG}" ]] || return 0
  # An exported GITHUB_TOKEN outranks gh's own stored credentials and 401s when it
  # is stale. The interactive shell hides this behind a `gh` function that blanks
  # the variable (shell_config/9-functions.rc); a script inherits the variable but
  # not the function, so blank it here too and let gh use its keyring auth.
  GITHUB_TOKEN="" GH_TOKEN="" gh pr view "${1}" --repo "${REPO_SLUG}" --json state --jq .state 2>/dev/null || true
}

[[ -d "${REPO}/.git" ]] || die "llama.cpp repo not found at ${REPO}"
[[ -d "${WORKTREE}" ]] || die "worktree not found at ${WORKTREE} (see QWEN_NEXT.md)"

echo "Fetching origin/master, PR #${PR} and ${#EXTRA_PRS[@]} extra PR(s)..."
fetch_args=(origin master "+refs/pull/${PR}/head:${PR_REF}")
for p in "${EXTRA_PRS[@]}"; do
  fetch_args+=("+refs/pull/${p}/head:refs/pr/${p}")
done
git -C "${REPO}" fetch "${fetch_args[@]}"

pr_head="$(git -C "${REPO}" rev-parse "${PR_REF}")"
master_head="$(git -C "${REPO}" rev-parse origin/master)"
# Parallel to EXTRA_PRS; indexed arrays rather than an associative one so the
# ordering stays explicit (merge order changes the result).
extra_heads=()
for p in "${EXTRA_PRS[@]}"; do
  extra_heads+=("$(git -C "${REPO}" rev-parse "refs/pr/${p}")")
done

# Queried before the already-built skip below, because closing a PR does not move
# its head - the marker stays valid and a no-op run would otherwise never mention it.
base_state=""
extra_states_upstream=()
closed_prs=()
if command -v gh >/dev/null 2>&1; then
  base_state="$(pr_state "${PR}")"
  for p in "${EXTRA_PRS[@]}"; do
    extra_states_upstream+=("$(pr_state "${p}")")
  done
  if [[ -z "${base_state}" ]]; then
    echo "warning: could not read PR state from GitHub (gh unauthenticated or offline);" >&2
    echo "         skipping the closed-PR check." >&2
  fi
else
  echo "note: gh not installed; skipping the closed-PR check." >&2
fi

if [[ "${base_state}" == "CLOSED" ]]; then
  echo "warning: PR #${PR} was CLOSED upstream without merging. This whole build" >&2
  echo "         exists to carry it - check whether it was superseded before you" >&2
  echo "         keep rebuilding against a dead branch." >&2
fi

for i in "${!EXTRA_PRS[@]}"; do
  if [[ "${extra_states_upstream[$i]:-}" == "CLOSED" ]]; then
    closed_prs+=("${EXTRA_PRS[$i]}")
    echo "warning: PR #${EXTRA_PRS[$i]} was CLOSED upstream without merging; it is still" >&2
    echo "         merged here. Drop it from EXTRA_PRS unless you mean to keep carrying it." >&2
  fi
done

if git -C "${REPO}" merge-base --is-ancestor "${PR_REF}" origin/master; then
  echo "PR #${PR} has MERGED upstream."
  echo "Retire this setup: build main as usual, delete the LLAMA_SERVER_BIN"
  echo "override in samm-mbp.env, then: git -C ${REPO} worktree remove ${WORKTREE}"
  exit 0
fi

if [[ "${base_state}" == "MERGED" ]]; then
  echo "warning: PR #${PR} shows MERGED upstream but its head is not an ancestor of" >&2
  echo "         master - a squash or rebase merge. The code is probably in master" >&2
  echo "         already; check before rebuilding, then retire this setup." >&2
fi

# The marker records the revisions built plus each extra PR's outcome, so a
# skipped run can still say the existing binary is missing one.
marker_key="${pr_head}+${master_head}"
for h in "${extra_heads[@]}"; do
  marker_key+="+${h}"
done

if [[ -f "${MARKER}" ]] && [[ "$(cut -d' ' -f1 "${MARKER}")" == "${marker_key}" ]]; then
  echo "Already built against this PR head and master."
  missing="$(cut -d' ' -f2- "${MARKER}" | tr ' ' '\n' | grep ':MISSING$' || true)"
  if [[ -n "${missing}" ]]; then
    echo "warning: that build is MISSING ${missing//:MISSING/}" >&2
  fi
  # Only offer the rebuild when someone is there to answer; piped or scheduled
  # runs keep the old skip-and-exit behaviour rather than blocking on read.
  if [[ ! -t 0 ]]; then
    echo "Nothing to do."
    exit 0
  fi
  read -r -p "Build anyway? (y/N) " reply || reply=""
  if [[ ! "${reply}" =~ ^[Yy]([Ee][Ss])?$ ]]; then
    echo "Nothing to do."
    exit 0
  fi
  echo "Rebuilding at the same revisions."
fi

# The previous run left patches/ applied, which is a dirty tree as far as the guard
# below is concerned. Reverse them first: if that restores a clean tree, the only
# edits were ours and there is nothing to warn about. Anything that fails to reverse
# was not ours and still trips the guard.
shopt -s nullglob
patches=("${SCRIPT_DIR}"/patches/*.patch)
shopt -u nullglob
# --index because the forward apply below uses --3way, which stages what it applies -
# reverting the working tree alone would leave the index dirty and trip the guard.
# Not --3way here: that insists the file already matches the index. Falling back to a
# working-tree-only revert covers a tree that was reset but not unstaged.
for patch in "${patches[@]}"; do
  git -C "${WORKTREE}" apply -R --index "${patch}" >/dev/null 2>&1 \
    || git -C "${WORKTREE}" apply -R "${patch}" >/dev/null 2>&1 \
    || true
done

# checkout -B rebuilds the branch from the PR head every run, so a tracked file
# edited by hand (a local compile fix, say) both blocks the checkout with a bare
# git error and cannot survive anyway. Say which files and how to park them.
dirty="$(git -C "${WORKTREE}" status --porcelain --untracked-files=no)"
if [[ -n "${dirty}" ]]; then
  {
    echo "Error: ${WORKTREE} has uncommitted changes:"
    echo "  ${dirty//$'\n'/$'\n'  }"
    echo "checkout -B would discard them. Park or drop them first:"
    echo "  git -C ${WORKTREE} stash push -m 'wip'"
    echo
    echo
    echo "The local patches this build needs are applied from patches/ automatically;"
    echo "they are not what this is complaining about. If you edited one by hand,"
    echo "regenerate it (see patches/README.md) rather than leaving it in the tree."
  } >&2
  exit 1
fi

echo "Updating ${BRANCH} to PR head ${pr_head:0:9}..."
git -C "${WORKTREE}" checkout -q -B "${BRANCH}" "${PR_REF}"

# Merge a ref, reporting what actually broke. The old version sent conflict
# output to /dev/null, so a dropped merge looked like a one-line warning with no
# way to tell a trivial comment clash from a real code divergence.
# rerere is enabled on this repo (rerere.enabled/autoupdate), so a conflict you
# resolve by hand once is replayed automatically on later runs - worth doing,
# since checkout -B above discards the branch every time.
try_merge() {
  local ref="${1}" label="${2}" out conflicts
  if out="$(git -C "${WORKTREE}" merge --no-edit "${ref}" 2>&1)"; then
    echo "Merged ${label}."
    return 0
  fi
  conflicts="$(git -C "${WORKTREE}" diff --name-only --diff-filter=U)"
  # rerere replays a recorded resolution and stages it, but `git merge` still
  # exits non-zero and leaves the merge uncommitted. Nothing unmerged plus a
  # live MERGE_HEAD means exactly that, so finish the commit rather than abort
  # the merge rerere just fixed.
  if [[ -z "${conflicts}" ]] && git -C "${WORKTREE}" rev-parse -q --verify MERGE_HEAD >/dev/null; then
    git -C "${WORKTREE}" commit --no-edit -q
    echo "Merged ${label} (conflicts replayed from rerere)."
    return 0
  fi
  {
    echo "warning: ${label} does not merge cleanly; building WITHOUT it."
    if [[ -n "${conflicts}" ]]; then
      echo "         conflicting files:"
      echo "           ${conflicts//$'\n'/$'\n'           }"
      echo "         resolve once by hand and rerere will replay it next run:"
      echo "           cd ${WORKTREE} && git merge ${ref}"
    else
      echo "         ${out//$'\n'/$'\n'         }"
    fi
  } >&2
  git -C "${WORKTREE}" merge --abort
  return 1
}

try_merge origin/master "origin/master (${master_head:0:9})" || true

# Ancestry only catches a plain merge upstream; a squash or rebase merge lands
# the same code under a new SHA and still fails this test, so the merge below is
# what actually decides.
extra_states=()
for i in "${!EXTRA_PRS[@]}"; do
  p="${EXTRA_PRS[$i]}"
  h="${extra_heads[$i]}"
  if git -C "${REPO}" merge-base --is-ancestor "refs/pr/${p}" origin/master; then
    echo "PR #${p} has merged upstream; skipping its merge (drop it from EXTRA_PRS)."
    extra_states+=("#${p}:upstream")
    continue
  fi
  if [[ "${extra_states_upstream[$i]:-}" == "MERGED" ]]; then
    echo "PR #${p} shows MERGED upstream under a different SHA (squash or rebase);" >&2
    echo "         its merge below is likely a no-op. Drop it from EXTRA_PRS." >&2
  fi
  if try_merge "refs/pr/${p}" "PR #${p} (${h:0:9})"; then
    extra_states+=("#${p}:merged")
  else
    extra_states+=("#${p}:MISSING")
  fi
done

# Local patches, applied after the merges and before the build. checkout -B above
# rebuilds the branch from the PR head every run, so these cannot be carried as
# working-tree edits or stashes - one is a silent 8x on cached turns, the other is
# the difference between the MTP graph loading and aborting. --3way lets them
# survive upstream moving the surrounding code. See patches/README.md.
for patch in "${patches[@]}"; do
  git -C "${WORKTREE}" apply --3way "${patch}" \
    || die "local patch $(basename "${patch}") no longer applies; fix it before building"
  echo "Applied $(basename "${patch}")."
done

# Use the same cmake preset as the main repo's build.sh (untracked file, so the
# worktree does not inherit it). Deliberately no `cmake --install`: the PR build
# must not overwrite the master binaries in ~/.local.
[[ -e "${WORKTREE}/CMakeUserPresets.json" ]] \
  || ln -s "${REPO}/CMakeUserPresets.json" "${WORKTREE}/CMakeUserPresets.json"

# Re-resolve OpenSSL on every configure. LLAMA_OPENSSL defaults ON, and CMake
# caches the absolute library path it finds; Homebrew deletes the old Cellar
# directory on upgrade, after which FindOpenSSL still reports OpenSSL_FOUND (the
# cache vars are non-empty) but skips creating OpenSSL::SSL (the dylib is gone),
# so configure fails with "links to OpenSSL::SSL but the target was not found".
# Clearing the three path vars re-runs the search; OPENSSL_ROOT_DIR points it at
# the versionless opt symlink, which brew repoints in place across upgrades.
ssl_opts=(-UOPENSSL_INCLUDE_DIR -UOPENSSL_SSL_LIBRARY -UOPENSSL_CRYPTO_LIBRARY)
if ssl_prefix="$(brew --prefix openssl@3 2>/dev/null)" && [[ -d "${ssl_prefix}" ]]; then
  ssl_opts+=(-DOPENSSL_ROOT_DIR="${ssl_prefix}")
fi

# The preset points CMAKE_OSX_SYSROOT at CommandLineTools/SDKs/MacOSX.sdk, which
# a CLT update repointed to a 27.0 SDK on 2026-09-11 while xcode-select still
# resolves the Xcode ld (ld-1267). That ld rejects the new SDK's tbd files
# ("unknown architecture: arm64e.x1-macos"). Ask the toolchain xcode-select
# resolves for its own SDK instead, so ld and SDK always come from the same
# place.
sdk_opts=()
if sdk_path="$(xcrun --sdk macosx --show-sdk-path 2>/dev/null)" && [[ -d "${sdk_path}" ]]; then
  sdk_opts+=(-DCMAKE_OSX_SYSROOT="${sdk_path}")
fi

echo "Building llama-server, llama-cli, llama-bench (preset: local)..."
(cd "${WORKTREE}" && cmake --preset local "${ssl_opts[@]}" "${sdk_opts[@]}" >/dev/null) \
  || die "cmake configure failed"
cmake --build "${WORKTREE}/build" --target llama-server llama-cli llama-bench -j \
  || die "build failed"

echo "${marker_key} ${extra_states[*]}" > "${MARKER}"
"${WORKTREE}/build/bin/llama-server" --version
for s in "${extra_states[@]}"; do
  case "${s##*:}" in
    merged)   echo "${s%%:*}: included." ;;
    upstream) echo "${s%%:*}: merged upstream, no longer carried separately." ;;
    MISSING)  echo "warning: ${s%%:*} is NOT in this build (see QWEN_NEXT.md)." >&2 ;;
  esac
done
# Repeated here because the state query runs before the merges and the build, far
# enough up the output to be scrolled away by the time the build finishes.
if [[ ${#closed_prs[@]} -gt 0 ]]; then
  echo "warning: CLOSED upstream but still carried: ${closed_prs[*]/#/#}" >&2
fi
echo "Done. The router picks this up via LLAMA_SERVER_BIN in samm-mbp.env."
