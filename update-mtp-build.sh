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
#   28092 - --cache-disk: the prompt cache persists to disk and reloads on start,
#           covering the case #28022 does not - a restart or reboot, which is what
#           every rebuild of this worktree causes.
#   25788 - Metal gated_delta_net cache fusion, mirroring the CUDA path: the kernel
#           writes recurrent-state snapshots straight into the KV cache instead of a
#           per-layer cpy. 36 of our 48 trunk layers are GDN.
#   27941 - qwen4exp follow-up fixes: indexer keys lost on sequence copy, blocks
#           keyed per-sequence rather than by position alone (wrong under
#           kv-unified, which is what our 4 auto slots run), and hand-edited
#           GGUF asserts turned into throws - our model is a graft.
#   28121 - flags ssm_a tensors not used with ggml_scan (4 lines).
# Dropped: 27977 (closed upstream). 28136 (--lazy-mode on-direct) - null on Metal
# at both a 5K and a 32K prompt, and needs a hand-applied ple_w compile fix every
# run. 28213 (QSA gather) - the author's +6% at 31k and +50% at 130k are CUDA; on
# Metal it measured slightly negative to 32k, null at 64k and +1.4% only at 128k,
# so it loses at the depths this machine actually runs. See QWEN_NEXT.md.
EXTRA_PRS=(28022 28232 28092 25788 27941 28121)
MARKER="${WORKTREE}/.last-mtp-build"

die() {
  echo "Error: ${1}" >&2
  exit 1
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

if git -C "${REPO}" merge-base --is-ancestor "${PR_REF}" origin/master; then
  echo "PR #${PR} has MERGED upstream."
  echo "Retire this setup: build main as usual, delete the LLAMA_SERVER_BIN"
  echo "override in samm-mbp.env, then: git -C ${REPO} worktree remove ${WORKTREE}"
  exit 0
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
    echo "The two patches this build needs are applied from patches/ automatically;"
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
  elif try_merge "refs/pr/${p}" "PR #${p} (${h:0:9})"; then
    extra_states+=("#${p}:merged")
  else
    extra_states+=("#${p}:MISSING")
  fi
done

# Local patches, applied after the merges and before the build. checkout -B above
# rebuilds the branch from the PR head every run, so these cannot be carried as
# working-tree edits or stashes - one of them is the difference between a build that
# compiles on macOS and one that does not, and the other is a silent 8x. --3way lets
# them survive upstream moving the surrounding code. See patches/README.md.
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

echo "Building llama-server, llama-cli, llama-bench (preset: local)..."
(cd "${WORKTREE}" && cmake --preset local "${ssl_opts[@]}" >/dev/null) \
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
echo "Done. The router picks this up via LLAMA_SERVER_BIN in samm-mbp.env."
