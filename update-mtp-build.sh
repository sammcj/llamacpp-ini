#!/usr/bin/env bash
set -euo pipefail

# Update the llama.cpp PR #27836 worktree build that serves Qwen3.8-Flash-Next
# MTP (see QWEN_NEXT.md). Fetches the latest PR head, merges in origin/master
# so the worktree keeps pace with daily master pulls, rebuilds, and tells you
# when the PR has merged upstream so the whole arrangement can be retired.
#
# Usage: ./update-mtp-build.sh    (nothing changed: prompts to rebuild anyway on a
#                                  terminal, skips silently when non-interactive)

REPO="${LLAMA_REPO:-${HOME}/git/llama.cpp}"
WORKTREE="${LLAMA_MTP_WORKTREE:-${HOME}/git/llama.cpp-pr27836}"
PR=27836
BRANCH="pr-${PR}-qwen4exp-mtp"
PR_REF="refs/pr/${PR}"
# PR 27977: qwen4exp long-context perf (ngram scan early-exit, QSA gather
# windows, indexer sum fix, bitmap used-cells). Chosen over the overlapping
# PR 27992 (kv-cell index): equal TG everywhere on M5 Max, but consistently
# +3-5.5% PP at 32K-115K. NEVER carry both - the combination regressed TG
# ~-18% at 32K in both-orders A/B.
PR2=27977
PR2_REF="refs/pr/${PR2}"
MARKER="${WORKTREE}/.last-mtp-build"

die() {
  echo "Error: ${1}" >&2
  exit 1
}

[[ -d "${REPO}/.git" ]] || die "llama.cpp repo not found at ${REPO}"
[[ -d "${WORKTREE}" ]] || die "worktree not found at ${WORKTREE} (see QWEN_NEXT.md)"

echo "Fetching origin/master, PR #${PR} and PR #${PR2}..."
git -C "${REPO}" fetch origin master \
  "+refs/pull/${PR}/head:${PR_REF}" "+refs/pull/${PR2}/head:${PR2_REF}"

pr_head="$(git -C "${REPO}" rev-parse "${PR_REF}")"
pr2_head="$(git -C "${REPO}" rev-parse "${PR2_REF}")"
master_head="$(git -C "${REPO}" rev-parse origin/master)"

if git -C "${REPO}" merge-base --is-ancestor "${PR_REF}" origin/master; then
  echo "PR #${PR} has MERGED upstream."
  echo "Retire this setup: build main as usual, delete the LLAMA_SERVER_BIN"
  echo "override in samm-mbp.env, then: git -C ${REPO} worktree remove ${WORKTREE}"
  exit 0
fi

# The marker records the revisions built, plus whether PR #PR2 made it in, so a
# skipped run can still say that the existing build is missing it.
marker_key="${pr_head}+${master_head}+${pr2_head}"

if [[ -f "${MARKER}" ]] && [[ "$(cut -d' ' -f1 "${MARKER}")" == "${marker_key}" ]]; then
  echo "Already built against this PR head and master."
  if [[ "$(cut -d' ' -f2 "${MARKER}")" == "MISSING" ]]; then
    echo "warning: that build does NOT include PR #${PR2} (long-context perf)." >&2
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
if git -C "${REPO}" merge-base --is-ancestor "${PR2_REF}" origin/master; then
  echo "PR #${PR2} has merged upstream; skipping its merge (remove it from this script)."
  pr2_state="upstream"
elif try_merge "${PR2_REF}" "PR #${PR2} (${pr2_head:0:9})"; then
  pr2_state="merged"
else
  pr2_state="MISSING"
fi

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

echo "Building llama-server, llama-cli, llama-bench (preset: local)..."
(cd "${WORKTREE}" && cmake --preset local "${ssl_opts[@]}" >/dev/null) \
  || die "cmake configure failed"
cmake --build "${WORKTREE}/build" --target llama-server llama-cli llama-bench -j \
  || die "build failed"

echo "${marker_key} ${pr2_state}" > "${MARKER}"
"${WORKTREE}/build/bin/llama-server" --version
case "${pr2_state}" in
  merged)   echo "PR #${PR2} (long-context perf): included." ;;
  upstream) echo "PR #${PR2}: merged upstream, no longer carried separately." ;;
  MISSING)  echo "warning: PR #${PR2} (long-context perf) is NOT in this build;" \
                 "expect lower prefill at depth (see QWEN_NEXT.md)." >&2 ;;
esac
echo "Done. The router picks this up via LLAMA_SERVER_BIN in samm-mbp.env."
