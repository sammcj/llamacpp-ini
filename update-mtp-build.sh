#!/usr/bin/env bash
set -euo pipefail

# Update the llama.cpp PR #27836 worktree build that serves Qwen3.8-Flash-Next
# MTP (see QWEN_NEXT.md). Fetches the latest PR head, merges in origin/master
# so the worktree keeps pace with daily master pulls, rebuilds, and tells you
# when the PR has merged upstream so the whole arrangement can be retired.
#
# Usage: ./update-mtp-build.sh          (idempotent; skips rebuild when nothing changed)

REPO="${LLAMA_REPO:-${HOME}/git/llama.cpp}"
WORKTREE="${LLAMA_MTP_WORKTREE:-${HOME}/git/llama.cpp-pr27836}"
PR=27836
BRANCH="pr-${PR}-qwen4exp-mtp"
PR_REF="refs/pr/${PR}"
MARKER="${WORKTREE}/.last-mtp-build"

die() {
  echo "Error: ${1}" >&2
  exit 1
}

[[ -d "${REPO}/.git" ]] || die "llama.cpp repo not found at ${REPO}"
[[ -d "${WORKTREE}" ]] || die "worktree not found at ${WORKTREE} (see QWEN_NEXT.md)"

echo "Fetching origin/master and PR #${PR}..."
git -C "${REPO}" fetch origin master "+refs/pull/${PR}/head:${PR_REF}"

pr_head="$(git -C "${REPO}" rev-parse "${PR_REF}")"
master_head="$(git -C "${REPO}" rev-parse origin/master)"

if git -C "${REPO}" merge-base --is-ancestor "${PR_REF}" origin/master; then
  echo "PR #${PR} has MERGED upstream."
  echo "Retire this setup: build main as usual, delete the LLAMA_SERVER_BIN"
  echo "override in samm-mbp.env, then: git -C ${REPO} worktree remove ${WORKTREE}"
  exit 0
fi

if [[ -f "${MARKER}" ]] && [[ "$(cat "${MARKER}")" == "${pr_head}+${master_head}" ]]; then
  echo "Already built against this PR head and master. Nothing to do."
  exit 0
fi

echo "Updating ${BRANCH} to PR head ${pr_head:0:9}..."
git -C "${WORKTREE}" checkout -q -B "${BRANCH}" "${PR_REF}"

if git -C "${WORKTREE}" merge --no-edit origin/master >/dev/null 2>&1; then
  echo "Merged origin/master (${master_head:0:9})."
else
  git -C "${WORKTREE}" merge --abort
  echo "warning: PR head conflicts with current master; building the plain PR head instead." >&2
fi

# Use the same cmake preset as the main repo's build.sh (untracked file, so the
# worktree does not inherit it). Deliberately no `cmake --install`: the PR build
# must not overwrite the master binaries in ~/.local.
[[ -e "${WORKTREE}/CMakeUserPresets.json" ]] \
  || ln -s "${REPO}/CMakeUserPresets.json" "${WORKTREE}/CMakeUserPresets.json"

echo "Building llama-server, llama-cli, llama-bench (preset: local)..."
(cd "${WORKTREE}" && cmake --preset local >/dev/null)
cmake --build "${WORKTREE}/build" --target llama-server llama-cli llama-bench -j \
  || die "build failed"

echo "${pr_head}+${master_head}" > "${MARKER}"
"${WORKTREE}/build/bin/llama-server" --version
echo "Done. The router picks this up via LLAMA_SERVER_BIN in samm-mbp.env."
