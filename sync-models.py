#!/usr/bin/env python3
"""Prepare ~/.lmstudio/models for `llama-server --models-dir` + wire up MTP.

Three things llama-server's own scanner cannot do:

1. It reads only the TOP level of --models-dir (loose *.gguf = one model, each
   immediate sub-dir = one multi-file model). LM Studio nests as
   <publisher>/<repo>/<file>.gguf, so we flatten that into a symlink farm.
   Every quant becomes its own model, named after its GGUF file (sharded models
   after their dir) - never after the containing folder, however few siblings
   it has: the web UI keys favourites on the name, so a folder-derived name
   would silently rename, and unfavourite, a model the moment a second quant
   landed beside it.

2. It does no MTP discovery. We probe each GGUF with llama-gguf and classify it
   (see ModelKind / KIND_SPEC below), which every model needs: the base preset
   turns speculation on globally, and that aborts the load of anything with no
   MTP unless we write it a spec-type of its own.

3. It cannot tell a runnable model from a file that merely looks like one.
   Kept out of the farm entirely:
     - MTP head stubs  - nextn tensors but only a handful of blocks; a head is
                         a draft for some other model, not a model
     - DFlash drafts   - arch = dflash; wire one by hand in the base ini via
                         spec-type = draft-dflash (example in samm-mbp.ini)
     - diffusion archs - cannot load in llama-server at all; listed in
                         .generated/diffusion-models.tsv for llama-diffusion-cli

Output is .generated/router.ini: the per-host base preset (<hostname>.ini) with
one generated section per model appended. Nothing is copied, only symlinked;
re-run whenever models change.

Tunables: paths and size thresholds in Config, each overridable by the env var
in its metadata or the matching CLI flag (CLI > env > default). The override
values those thresholds select are in the helpers just below Config.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from enum import Enum, auto
from pathlib import Path

GB = 1024**3
SCRIPT_DIR = Path(__file__).resolve().parent
PROG = Path(__file__).name


# --------------------------------------------------------------------------
# Configuration - the single source of truth for every tunable.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """Every tunable for this script, with its default. This is the only place
    to change behaviour; there are no magic numbers elsewhere (the size tiers in
    SIZE_TIERS read their values from here).

    Runtime/server defaults - ctx-size, KV quant, temp, spec-type - are NOT here;
    they live in the base preset this script extends (the per-host <hostname>.ini,
    default samm-mbp.ini; see the base_ini property). Config controls
    the sync itself and the per-model overrides it writes on top of that base.

    Fields built with field(..., metadata={"env": ...}) can also be overridden by
    that env var or its matching CLI flag; precedence is CLI > env > the default
    shown here. Plain fields (no metadata) are edit-in-place defaults.
    """

    # --- Where things live (env: MODELS_SRC / MODELS_DIR / LLAMA_GGUF) --------
    src: Path = field(
        default=Path.home() / ".lmstudio" / "models",
        metadata={"env": "MODELS_SRC", "conv": Path},
    )
    dest: Path = field(
        default=SCRIPT_DIR / "models", metadata={"env": "MODELS_DIR", "conv": Path}
    )
    llama_gguf: str = field(
        default="llama-gguf", metadata={"env": "LLAMA_GGUF", "conv": str}
    )

    # --- MTP / architecture detection ----------------------------------------
    head_max_blocks: int = (
        8  # nextn tensors + <= this many blocks == head stub, not a model
    )
    probe_timeout: int = 120  # seconds per llama-gguf probe (it reads headers only)
    diffusion_arch: tuple[str, ...] = (
        "diffusion",
        "dream",
        "llada",
    )  # only llama-diffusion-cli runs these
    cache_version: int = 4  # probe-cache schema; bump to invalidate old caches

    # --- Size-tier thresholds, in GB (env: SIZE_SMALL_KV_GB / SIZE_CKPT_GB) ---
    # They stack: a model can match more than one tier. See SIZE_TIERS for wiring.
    small_kv_gb: int = field(
        default=40, metadata={"env": "SIZE_SMALL_KV_GB", "conv": int}
    )  # under this: upgrade KV cache
    ckpt_gb: int = field(
        default=80, metadata={"env": "SIZE_CKPT_GB", "conv": int}
    )  # over this: cap ctx + checkpoints

    # --- What each size tier writes (the override values themselves) ----------
    small_kv_qwen: str = (
        "bf16"  # KV cache type for Qwen under small_kv_gb (Qwen handles bf16 well)
    )
    small_kv_other: str = "f16"  # KV cache type for every other model under small_kv_gb
    ckpt_checkpoints: str = "32"  # ctx-checkpoints for models over ckpt_gb
    ckpt_ctx_size: int = 131072  # ctx-size for models over ckpt_gb (128K)

    # --- Chat template fix for Qwen 3.5+ (env: QWEN_CHAT_TEMPLATE) ------------
    # The templates shipped inside Qwen 3.5+ GGUFs are broken (tool calls and
    # thinking blocks); point those models at the corrected jinja instead.
    qwen_template: Path = field(
        default=Path.home()
        / "git"
        / "Qwen-Fixed-Chat-Templates"
        / "chat_template.jinja",
        metadata={"env": "QWEN_CHAT_TEMPLATE", "conv": Path},
    )
    qwen_template_min: tuple[int, int] = (3, 5)  # lowest Qwen version that gets it
    # PR #28092's --cache-disk. Written per model rather than once in the launcher,
    # because the router renders its own CLI args into every child (server_models
    # builds base_preset from argc/argv) and a child deletes every entry in the
    # directory whose cache key is not its own. One shared path means loading a
    # second model wipes the first model's cache. Set empty to disable.
    cache_disk_dir: Path = field(
        default=Path.home() / ".cache" / "llama.cpp" / "prompt-cache",
        metadata={"env": "LLAMA_CACHE_DISK", "conv": Path},
    )
    cache_disk_max_mib: int = 32768  # ~900 MiB per 33k-token prompt

    @property
    def small_kv_bytes(self) -> int:
        return self.small_kv_gb * GB

    @property
    def ckpt_bytes(self) -> int:
        return self.ckpt_gb * GB

    @property
    def gen_dir(self) -> Path:
        return SCRIPT_DIR / ".generated"

    @property
    def base_ini(self) -> Path:
        """Base preset to extend: LLAMA_BASE_INI, else <hostname>.ini, else the
        M5 Max preset. Lets one repo serve hosts with different RAM budgets."""
        env = os.environ.get("LLAMA_BASE_INI")
        if env:
            return Path(env)
        host_ini = SCRIPT_DIR / f"{socket.gethostname().split('.')[0]}.ini"
        return host_ini if host_ini.exists() else SCRIPT_DIR / "samm-mbp.ini"

    @property
    def router_ini(self) -> Path:
        return self.gen_dir / "router.ini"

    @property
    def diffusion_tsv(self) -> Path:
        return self.gen_dir / "diffusion-models.tsv"

    @property
    def cache_path(self) -> Path:
        return self.gen_dir / f"probe-cache-v{self.cache_version}.tsv"

    def __post_init__(self) -> None:
        if not self.small_kv_gb < self.ckpt_gb:
            raise ValueError("size tiers must be ordered: small_kv < ckpt")
        # os.symlink stores its target verbatim and the kernel resolves a
        # relative one against the link's directory, not our cwd - a relative
        # --src would build a farm of dangling links. abspath, not resolve, so a
        # symlinked source tree keeps the path the user recognises.
        for name in ("src", "dest", "qwen_template"):
            object.__setattr__(self, name, Path(os.path.abspath(getattr(self, name))))

    @classmethod
    def load(cls, **cli_overrides) -> Config:
        """Build a Config with precedence CLI arg > env var > field default."""
        values: dict = {}
        for f in fields(cls):
            if cli_overrides.get(f.name) is not None:
                values[f.name] = cli_overrides[f.name]
                continue
            env = f.metadata.get("env")
            if env and env in os.environ:
                conv = f.metadata.get("conv", str)
                values[f.name] = conv(os.environ[env])
        return cls(**values)


# --------------------------------------------------------------------------
# MTP classification -> the spec-type each kind needs. Lives here with the rest
# of the per-model policy, not in the writer, which only formats what it is
# given. EMBEDDED_MTP is absent on purpose: it inherits [*] spec-type.
# --------------------------------------------------------------------------
class ModelKind(Enum):
    EMBEDDED_MTP = auto()  # nextn in the model (Qwen): self-drafts, inherits [*]
    PAIRED = auto()  # no nextn, separate head exists (Gemma 4): spec-draft-model
    NGRAM = auto()  # no MTP at all: model-free ngram speculation fallback
    NO_SPEC = auto()  # embedding model; speculation is meaningless


KIND_SPEC: dict[ModelKind, str] = {
    # Re-stated (not inherited) so the pairing is readable in one section.
    ModelKind.PAIRED: "draft-mtp",
    # Replaces the inherited draft-mtp, which would abort the load. No draft
    # model needed - it drafts from recent-context n-grams.
    ModelKind.NGRAM: "ngram-mod",
    # Speculation is meaningless for an embedding model.
    ModelKind.NO_SPEC: "none",
}


# --------------------------------------------------------------------------
# Size tiers as data: predicate -> ini overrides. Later tiers win on key clash.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SizeTier:
    name: str
    applies: Callable[[int, Config], bool]  # (weight_bytes, cfg) -> in this tier?
    make: Callable[[str, Config], dict[str, str]]  # (model_name, cfg) -> ini keys


def _kv_override(name: str, cfg: Config) -> dict[str, str]:
    kv = cfg.small_kv_qwen if "qwen" in name.lower() else cfg.small_kv_other
    return {"cache-type-k": kv, "cache-type-v": kv}


SIZE_TIERS: list[SizeTier] = [
    SizeTier("small-kv", lambda b, c: b < c.small_kv_bytes, _kv_override),
    SizeTier(
        "ckpt",
        lambda b, c: b > c.ckpt_bytes,
        lambda _n, c: {
            "ctx-checkpoints": c.ckpt_checkpoints,
            "ctx-size": str(c.ckpt_ctx_size),
        },
    ),
]


def cache_disk_overrides(name: str, cfg: Config) -> dict[str, str]:
    """Give each model its own prompt-cache directory - see Config.cache_disk_dir."""
    if not str(cfg.cache_disk_dir):
        return {}
    return {
        "cache-disk": str(cfg.cache_disk_dir / name),
        "cache-disk-max": str(cfg.cache_disk_max_mib),
    }


def size_overrides(name: str, weight_bytes: int, cfg: Config) -> dict[str, str]:
    out: dict[str, str] = {}
    for tier in SIZE_TIERS:
        if tier.applies(weight_bytes, cfg):
            out.update(tier.make(name, cfg))
    return out


# --------------------------------------------------------------------------
# Chat-template override: Qwen 3.5 and newer ship broken in-GGUF templates.
# --------------------------------------------------------------------------
# Matches Qwen3.5, qwen-3.6, Qwen_Qwen3.8, ThinkingCap-Qwen3.6-27B, ... The
# version is optional-minor so a bare "Qwen4" reads as 4.0 (still newer).
QWEN_VERSION_RE = re.compile(r"(?i)qwen[._-]?(\d+)(?:\.(\d+))?")


def qwen_version(name: str) -> tuple[int, int] | None:
    """Highest Qwen version in a model name, or None if it names no Qwen.

    Highest, not first: "Qwen2.5-coder-distill-Qwen3.6" is a 3.6 model, and a
    publisher prefix ("Qwen_Qwen3.6-27B") repeats the vendor token harmlessly.
    """
    versions = [
        (int(maj), int(minor or 0)) for maj, minor in QWEN_VERSION_RE.findall(name)
    ]
    return max(versions) if versions else None


# Lowest Qwen version that gets the measured 3.8 draft/sampler tuning below.
QWEN_TUNED_MIN = (3, 8)


def template_overrides(name: str, cfg: Config) -> dict[str, str]:
    ver = qwen_version(name)
    if ver is None or ver < cfg.qwen_template_min:
        return {}
    return {"chat-template-file": str(cfg.qwen_template)}


def mtp_draft_overrides(name: str) -> dict[str, str]:
    """Draft tuning for embedded-MTP Qwen 3.8 models (caller checks the kind).
    Measured on Qwen3.8-27B UD-Q6_K_L (M5 Max, ~450-token prompt): n-max 4 with
    GPU-side draft sampling 37.9 t/s vs 36.7 at the inherited n-max 3; 5 and 6
    were slower (rejected deep drafts waste the verify batch). Plain decode is
    ~19-20 t/s; a DFlash2 draft measured 34.1 t/s on the same model, so
    embedded MTP stays the default. The Flash-Next graft is unaffected: it is
    not farm-scanned and keeps its own section (single-layer head, best at 6).

    Gated >= 3.8, like the chat-template fix: an exact == would drop the tuning
    silently the day a 3.9 lands. Re-measure when one does."""
    ver = qwen_version(name)
    if ver is None or ver < QWEN_TUNED_MIN:
        return {}
    return {"spec-draft-n-max": "4", "spec-draft-backend-sampling": "1"}


def sampler_overrides(name: str) -> dict[str, str]:
    """Qwen 3.8 recommended sampling: temp 1.0, top-p 0.95, top-k 20, min-p 0.0,
    presence/repeat penalties off. Only the values that differ from llama-server
    defaults (temp 0.8, top-k 40, min-p 0.05) or the base [*] (temp 0.6) are
    written; top-p 0.95 and zero penalties are already the server defaults.
    Gated >= 3.8 for the reason in mtp_draft_overrides."""
    ver = qwen_version(name)
    if ver is None or ver < QWEN_TUNED_MIN:
        return {}
    return {"temp": "1.0", "top-k": "20", "min-p": "0.0"}


# --------------------------------------------------------------------------
# GGUF probing (cached by path|size:mtime, TSV-compatible with the bash cache).
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ProbeResult:
    nextn: bool  # has nextn tensors (embedded MTP or a head stub)
    head: bool  # nextn tensors + few blocks -> a standalone MTP head, not a model
    diffusion: bool  # arch only runs under llama-diffusion-cli
    embed: bool  # carries <arch>.pooling_type -> an embedding model
    dflash: bool  # arch = dflash -> a speculative draft, not a runnable model
    ok: bool = True  # False = never probed; every flag above is "unknown", not "no"


class Prober:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.have_gguf = shutil.which(cfg.llama_gguf) is not None
        if not self.have_gguf:
            print(
                f"warning: '{cfg.llama_gguf}' not found; MTP detection disabled.",
                file=sys.stderr,
            )
        self.cache: dict[str, str] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if self.cfg.cache_path.exists():
            for line in self.cfg.cache_path.read_text().splitlines():
                if "\t" in line:
                    k, v = line.split("\t", 1)
                    self.cache[k] = v

    def probe(self, f: Path) -> ProbeResult:
        st = f.stat()
        key = f"{f}|{st.st_size}:{int(st.st_mtime)}"
        val = self.cache.get(key)
        if val is None:
            val = self._probe_raw(f)
            if val is None:
                # Never cache an unprobed result: the key is path|size:mtime, so
                # installing llama-gguf or clearing a transient failure does not
                # change it - the misclassification would outlive its cause.
                return ProbeResult(
                    nextn=False,
                    head=False,
                    diffusion=False,
                    embed=False,
                    dflash=False,
                    ok=False,
                )
            self.cache[key] = val
        cnt, blk, dif, emb, dfl = self._parse(val)
        nextn = cnt > 0
        return ProbeResult(
            nextn=nextn,
            head=nextn and blk <= self.cfg.head_max_blocks,
            diffusion=dif > 0,
            embed=emb > 0,
            dflash=dfl > 0,
        )

    def _probe_raw(self, f: Path) -> str | None:
        """Classification counters for f, or None if it could not be probed."""
        if not self.have_gguf:
            return None
        out = self._run_gguf(f)
        if out is None:
            return None
        cnt = sum(1 for ln in out.splitlines() if re.search(r"name = .*nextn", ln))
        blocks = [int(m) for m in re.findall(r"blk\.(\d+)\.", out)]
        blk = max(blocks) if blocks else 0
        am = re.search(r"[a-z0-9_-]+\.block_count", out)
        arch = am.group(0)[: -len(".block_count")] if am else ""
        dif = 1 if arch.startswith(self.cfg.diffusion_arch) else 0
        dfl = 1 if arch == "dflash" else 0
        emb = 1 if any(".pooling_type" in ln for ln in out.splitlines()) else 0
        return f"{cnt}:{blk}:{dif}:{emb}:{dfl}"

    def _run_gguf(self, f: Path) -> str | None:
        """llama-gguf's dump for f, or None if it did not run cleanly.

        The returncode matters: llama-gguf aborts (134) on a malformed file
        after printing part of its output, and that partial dump parses as a
        plausible model - no nextn, no arch - so a head would look servable."""
        try:
            r = subprocess.run(
                [self.cfg.llama_gguf, str(f), "r", "n"],
                capture_output=True,
                text=True,
                timeout=self.cfg.probe_timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"warning: probe of '{f}' failed: {e}", file=sys.stderr)
            return None
        if r.returncode != 0:
            print(
                f"warning: probe of '{f}' failed: {self.cfg.llama_gguf} exited "
                f"{r.returncode}; treating it as unclassified.",
                file=sys.stderr,
            )
            return None
        return r.stdout or ""

    @staticmethod
    def _parse(val: str) -> tuple[int, int, int, int, int]:
        # Pad older short cache entries with zeros; treat any garbage as 0
        # (a corrupt cache degrades to "unknown", it never crashes the run).
        def to_int(x: str) -> int:
            try:
                return int(x)
            except ValueError:
                return 0

        p = (val.split(":") + ["0"] * 5)[:5]
        return (to_int(p[0]), to_int(p[1]), to_int(p[2]), to_int(p[3]), to_int(p[4]))

    def save_cache(self) -> None:
        lines = [f"{k}\t{v}" for k, v in sorted(self.cache.items())]
        self.cfg.cache_path.write_text("".join(f"{ln}\n" for ln in lines))


# --------------------------------------------------------------------------
# Model records. spec-draft-model only exists for PAIRED, so it hangs off the
# record, not a side table.
# --------------------------------------------------------------------------
@dataclass
class Model:
    name: str
    kind: ModelKind
    tags: list[str] = field(default_factory=list)
    draft_head: Path | None = None
    overrides: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Pure filename helpers.
# --------------------------------------------------------------------------
def extract_quant(basename: str) -> str:
    """Quant label (Q6_K, UD-Q5_K_XL, IQ4_NL, BF16) or "" if the name encodes none.

    GGUF's general.file_type would be more authoritative, but llama-gguf prints
    key names only, not their values.
    """
    b = basename[:-5] if basename.endswith(".gguf") else basename
    matches = re.findall(r"(?i)(?:UD-)?(?:IQ|Q)[0-9]+(?:_[0-9A-Za-z]+)*", b)
    q = matches[-1] if matches else ""
    if not q:
        matches = re.findall(r"(?i)BF16|F16|F32|MXFP4", b)
        q = matches[-1] if matches else ""
    return q.upper()


def normalise(name: str) -> str:
    """A base-model identity key for pairing heads to models: lowercased, with
    quant / shard / format / variant tokens stripped. E.g. both
    "gemma-4-31B-it-MTP-Q8_0" and "gemma-4-31B-it-qat-UD-Q4_K_XL" -> gemma-4-31b-it.

    The quant pattern covers IQ as well as Q: without the optional i, an
    IQ-quantised file keeps its quant tokens (gemma-4-31b-it-iq4-xs), matches no
    base identity, and silently falls back to ngram instead of pairing.
    """
    s = (name[:-5] if name.endswith(".gguf") else name).lower()
    s = re.sub(r"-[0-9]+-of-[0-9]+", "", s)
    s = re.sub(r"[._-]i?q[0-9]+(_[0-9a-z]+)*", "", s)
    s = re.sub(r"[._-](ud|k_xl|k_m|k_s|k_l|f16|bf16|fp16|f32|mxfp4|i1|imatrix)", "", s)
    s = s.replace("_", "-")
    tokens = [t for t in s.split("-") if t and t not in ("mtp", "qat", "gguf")]
    return "-".join(tokens)


def force_symlink(src: Path, dst: Path) -> None:
    """ln -sfn: replace an existing link/file at dst, never deref a dir link."""
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            raise IsADirectoryError(f"refusing to replace real directory: {dst}")
        dst.unlink()
    os.symlink(src, dst)


def only_symlinks(d: Path) -> bool:
    """True if every entry under d (recursively) is a symlink - i.e. a directory
    we generated. A real file or nested real dir makes it False.
    """
    for root, dirs, files in os.walk(d):  # followlinks=False: don't recurse links
        for entry in (*dirs, *files):
            if not (Path(root) / entry).is_symlink():
                return False
    return True


def iter_ggufs(src: Path):
    """Every real *.gguf under src. followlinks so a symlinked src tree (e.g.
    ~/.lmstudio/models -> ~/.cache/lm-studio/models) or symlinked publisher dirs
    are scanned; the visited dev/inode set stops a symlink cycle looping forever.

    Symlinked *files* are skipped (so not `find -L`): they are the farm's own
    links, or a second route to a file the walk already yields under its real
    path - linking both would give one model two farm names.
    """
    seen: set[tuple[int, int]] = set()
    for root, dirs, files in os.walk(src, followlinks=True):
        try:
            st = os.stat(root)
        except OSError:
            dirs.clear()
            continue
        if (st.st_dev, st.st_ino) in seen:
            dirs.clear()  # already walked this directory via another path
            continue
        seen.add((st.st_dev, st.st_ino))
        for fn in files:
            if fn.endswith(".gguf"):
                p = Path(root) / fn
                if p.is_file() and not p.is_symlink():
                    yield p


# --------------------------------------------------------------------------
# The sync itself.
# --------------------------------------------------------------------------
class Sync:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.prober = Prober(cfg)
        self.head_index: dict[str, Path] = {}
        self.models: list[Model] = []
        self.diffusion: list[tuple[str, Path]] = []
        self.linked = 0
        self.skipped_heads = 0
        self.skipped_dflash = 0
        self.nommproj_twins = 0
        self.templated = 0
        self.size_tiered = 0
        self.warned_template = False

    # -- classification helpers ------------------------------------------------
    def _is_head(self, g: Path) -> bool:
        pr = self.prober.probe(g)
        if pr.ok:
            return pr.head
        # Unprobed: fall back to the mtp- filename convention rather than assume
        # "not a head" - a head admitted to the farm aborts on load.
        return g.name.startswith("mtp-") and g.name.endswith(".gguf")

    def _is_diffusion(self, g: Path) -> bool:
        # No filename convention to fall back on; unprobed reads as "not one".
        return self.prober.probe(g).diffusion

    def _is_dflash(self, g: Path) -> bool:
        pr = self.prober.probe(g)
        if pr.ok:
            return pr.dflash
        # Fall back to the -DFlash filename convention (see _is_head).
        return "dflash" in g.name.lower()

    def _template_overrides(self, name: str) -> dict[str, str]:
        """Qwen 3.5+ chat-template fix, dropped if the jinja file is absent -
        pointing chat-template-file at a missing path aborts the model load."""
        out = template_overrides(name, self.cfg)
        if "chat-template-file" in out and not self.cfg.qwen_template.is_file():
            if not self.warned_template:
                print(
                    f"warning: chat template '{self.cfg.qwen_template}' not found; "
                    "Qwen 3.5+ models keep their built-in template.",
                    file=sys.stderr,
                )
                self.warned_template = True
            out.pop("chat-template-file")
        return out

    def _unique_name(self, desired: str, suffix: str, prefix: str) -> str:
        """A free farm name: the bare name, else <prefix>_<desired>, else a
        numbered <prefix>_<desired>_N. The first to claim a bare name keeps it,
        so callers must iterate in a stable order (see _build_farm) or a rename
        would orphan a web-UI favourite. The numbered tail is there because the
        prefixed name can collide too; returning it unchecked let force_symlink
        silently overwrite the earlier model's link.
        """

        def free(n: str) -> bool:
            t = self.cfg.dest / f"{n}{suffix}"
            return not (t.exists() or t.is_symlink())

        if free(desired):
            return desired
        prefixed = f"{prefix}_{desired}"
        if free(prefixed):
            return prefixed
        n = 2
        while not free(f"{prefixed}_{n}"):
            n += 1
        return f"{prefixed}_{n}"

    # -- model finalisation ----------------------------------------------------
    def _finalise(
        self, name: str, match: str, files: list[Path], twin: bool = False
    ) -> Model:
        """Probe the model's file(s), record size-tier keys off the summed weight
        bytes (mmproj is never passed in), then classify it for MTP.

        twin marks a -no-mmproj twin - a second view of a model already counted,
        so it is left out of the summary counters.
        """
        tags: list[str] = []
        quant = extract_quant(match)  # quant first, so a badge reads "Q6_K MTP vision"
        if quant:
            tags.append(quant)

        embedded = is_embed = False
        weight_bytes = 0
        for f in files:
            pr = self.prober.probe(f)
            embedded |= pr.nextn and not pr.head
            is_embed |= pr.embed
            weight_bytes += f.stat().st_size

        overrides = size_overrides(name, weight_bytes, self.cfg)
        sized = bool(overrides)
        overrides.update(self._template_overrides(name))
        overrides.update(sampler_overrides(name))
        overrides.update(cache_disk_overrides(name, self.cfg))
        if not twin:
            self.size_tiered += sized
            self.templated += "chat-template-file" in overrides

        if embedded:
            kind, tags = ModelKind.EMBEDDED_MTP, tags + ["MTP"]
            head = None
            overrides.update(mtp_draft_overrides(name))
        else:
            head = self.head_index.get(normalise(match))
            if head is not None:
                kind, tags = ModelKind.PAIRED, tags + ["MTP"]
            elif is_embed:
                kind = ModelKind.NO_SPEC
            else:
                kind, tags = ModelKind.NGRAM, tags + ["ngram"]

        model = Model(
            name=name, kind=kind, tags=tags, draft_head=head, overrides=overrides
        )
        self.models.append(model)
        return model

    def _link_twin(self, base: str, files: list[Path]) -> None:
        """A text-only twin (weights only, mmproj excluded) so a multimodal model
        can also be served without the projector. It flows through _finalise, so
        it inherits the same spec/size keys; only the -no-mmproj name and the
        absent projector distinguish it (no --no-mmproj flag needed).
        """
        twin = f"{base}-no-mmproj"
        tdir = self.cfg.dest / twin
        tdir.mkdir(parents=True, exist_ok=True)
        for g in files:
            force_symlink(g, tdir / g.name)
        self.linked += 1
        self.nommproj_twins += 1
        self._finalise(twin, files[0].name, files, twin=True)

    # -- passes ----------------------------------------------------------------
    def _update_template_repo(self) -> None:
        """Pull the latest Qwen chat-template fixes before wiring models to them.
        Non-fatal: offline, a dirty checkout, or a plain non-git dir just means
        the template already on disk is used."""
        repo = self.cfg.qwen_template.parent
        if not (repo / ".git").exists():
            return
        try:
            r = subprocess.run(
                ["git", "-C", str(repo), "pull", "--ff-only", "--quiet"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if r.returncode != 0:
                msg = r.stderr.strip() or r.stdout.strip() or f"exit {r.returncode}"
                print(f"warning: git pull in '{repo}' failed: {msg}", file=sys.stderr)
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"warning: git pull in '{repo}' failed: {e}", file=sys.stderr)

    def _check_src(self) -> None:
        if not self.cfg.src.is_dir():
            sys.exit(f"error: source models directory '{self.cfg.src}' not found.")
        # Checked up front: at read time the farm has already been rebuilt.
        if not self.cfg.base_ini.is_file():
            sys.exit(
                f"error: base preset '{self.cfg.base_ini}' not found "
                "(set LLAMA_BASE_INI or add a <hostname>.ini)."
            )
        self.cfg.dest.mkdir(parents=True, exist_ok=True)
        self.cfg.gen_dir.mkdir(parents=True, exist_ok=True)

    def _clean_dest(self) -> None:
        """Remove what we generated: top-level symlinks, plus per-quant/twin dirs
        (those whose contents are entirely symlinks). Real files and dirs placed
        here by hand survive; a hand-placed *symlink* does not, being
        indistinguishable from ours - place a real one, as the MTP graft is.
        """
        dest = self.cfg.dest
        for p in list(dest.iterdir()):
            if p.is_symlink():
                p.unlink()
        for p in list(dest.iterdir()):
            if p.is_dir() and not p.is_symlink() and only_symlinks(p):
                shutil.rmtree(p)

    def _index_heads(self) -> None:
        """Pass 1: index standalone MTP heads by their normalised base name.

        More than one head can normalise to the same base (e.g. a quantised head
        beside the weights AND a purpose-named mtp- stub). Pick deterministically:
        prefer a file named like a standalone head (mtp-*), then the first by
        name - never by directory-walk order, which is unspecified.
        """
        candidates: dict[str, list[Path]] = {}
        for g in iter_ggufs(self.cfg.src):
            b = g.name
            if "mmproj" in b or "-of-" in b:  # mmproj / shards are never heads
                continue
            if self._is_head(g):
                candidates.setdefault(normalise(b), []).append(g)
        for key, heads in candidates.items():
            self.head_index[key] = min(
                heads, key=lambda p: (not p.name.startswith("mtp-"), p.name)
            )

    def _build_farm(self) -> None:
        """Pass 2: build the model farm and classify each model for MTP.

        Sort the dirs: directory-walk order is unspecified, and _unique_name's
        collision fallback (first one wins the bare name) depends on a stable
        order - unsorted, adding or removing a model could silently rename
        another, and the web UI keys favourites on the model name.

        Each directory is handed the files iter_ggufs found for it rather than
        re-globbing, which would also pick up the symlinked GGUFs iter_ggufs
        skips and link one file twice under two names.
        """
        by_dir: dict[Path, list[Path]] = {}
        for g in sorted(iter_ggufs(self.cfg.src)):
            by_dir.setdefault(g.parent, []).append(g)
        for d in sorted(by_dir):
            if d == self.cfg.src:
                self._process_toplevel(by_dir[d])
            else:
                self._process_dir(d, by_dir[d])

    def _process_toplevel(self, ggufs: list[Path]) -> None:
        """Loose *.gguf at the top of the source tree: each is its own model."""
        for g in ggufs:
            if self._is_head(g):
                self.skipped_heads += 1
                continue
            if self._is_dflash(g):
                self.skipped_dflash += 1
                continue
            if self._is_diffusion(g):
                self.diffusion.append((g.stem, g))
                continue
            name = self._unique_name(g.stem, ".gguf", "root")
            force_symlink(g, self.cfg.dest / f"{name}.gguf")
            self.linked += 1
            self._finalise(name, g.name, [g])

    def _process_dir(self, d: Path, ggufs: list[Path]) -> None:
        if not ggufs:
            return

        mmprojs: list[Path] = []
        first_shard: Path | None = None
        shards: list[Path] = []
        quants: list[Path] = []
        for g in ggufs:
            base = g.name
            if "mmproj" in base:
                mmprojs.append(g)
            elif "-of-" in base:
                shards.append(g)
                if "-00001-of-" in base:
                    first_shard = g
            elif self._is_head(g):
                self.skipped_heads += 1
            elif self._is_dflash(g):
                self.skipped_dflash += 1
            else:
                quants.append(g)

        parent = d.parent.name

        # A repo can ship two projectors (mmproj-F16 beside mmproj-F32); first
        # sorted, so walk order cannot silently change which one a model gets.
        mmproj = mmprojs[0] if mmprojs else None
        if len(mmprojs) > 1:
            print(
                f"warning: '{d}' has {len(mmprojs)} mmproj files; using {mmproj.name}.",  # type: ignore[union-attr]
                file=sys.stderr,
            )

        if shards and first_shard is None:
            # Unloadable without shard 1, and the quant loop skips shards, so
            # this would otherwise vanish silently.
            print(
                f"warning: '{d}' has {len(shards)} shard(s) but no -00001-of- "
                "shard; skipping the sharded model (incomplete download?).",
                file=sys.stderr,
            )
        elif first_shard is not None:
            # Sharded model == one model. Link the dir; probe every shard.
            if self._is_diffusion(first_shard):
                self.diffusion.append((d.name, first_shard))
            else:
                name = self._unique_name(d.name, "", parent)
                force_symlink(d, self.cfg.dest / name)
                self.linked += 1
                model = self._finalise(name, first_shard.name, shards)
                if mmproj is not None:
                    model.tags.append("vision")
                    self._link_twin(name, shards)

        # Reached even when the dir also holds a sharded model: a repo commonly
        # ships a sharded big quant beside single-file small ones.
        # Every quant is its own model, named after its GGUF file - never after
        # the containing dir, even when it is the only quant (see module docs).
        # With an mmproj we build a per-quant directory (model + mmproj) so
        # multimodal pairing survives.
        for q in quants:
            qname = q.stem
            if self._is_diffusion(q):
                self.diffusion.append((qname, q))
                continue
            if mmproj is not None:
                name = self._unique_name(qname, "", parent)
                (self.cfg.dest / name).mkdir(parents=True, exist_ok=True)
                force_symlink(q, self.cfg.dest / name / f"{qname}.gguf")
                force_symlink(mmproj, self.cfg.dest / name / mmproj.name)
            else:
                name = self._unique_name(qname, ".gguf", parent)
                force_symlink(q, self.cfg.dest / f"{name}.gguf")
            self.linked += 1
            model = self._finalise(name, q.name, [q])
            if mmproj is not None:
                model.tags.append("vision")
                self._link_twin(name, [q])

    # -- output ----------------------------------------------------------------
    def _write_diffusion_tsv(self) -> None:
        self.cfg.diffusion_tsv.write_text(
            "".join(f"{n}\t{p}\n" for n, p in self.diffusion)
        )

    def _write_router_ini(self) -> None:
        c = self.cfg
        mtp = [m for m in self.models if m.kind is ModelKind.EMBEDDED_MTP]
        paired = [m for m in self.models if m.kind is ModelKind.PAIRED]
        ngram = [m for m in self.models if m.kind is ModelKind.NGRAM]
        nospec = [m for m in self.models if m.kind is ModelKind.NO_SPEC]

        def emit(m: Model) -> str:
            """Key order is deliberate and must be kept: llama-server applies a
            section top-down, so spec-type comes first (it decides whether the
            draft keys mean anything), then spec-draft-model, then tags, then
            the size/template/sampler overrides."""
            out = ""
            spec = KIND_SPEC.get(m.kind)
            if spec is not None:
                out += f"{'spec-type'.ljust(16)} = {spec}\n"
            if m.draft_head is not None:
                out += f"{'spec-draft-model'.ljust(16)} = {m.draft_head}\n"
            if m.tags:
                out += f"tags = {','.join(m.tags)}\n"
            out += "".join(f"{k.ljust(16)} = {v}\n" for k, v in m.overrides.items())
            return out

        parts: list[str] = [c.base_ini.read_text(), "\n"]
        parts.append(
            "; ====================================================================\n"
            f"; AUTO-GENERATED by {PROG} - do not edit; re-run to refresh.\n"
            f"; Why each key is here is documented in {PROG} itself (module docstring\n"
            "; + the override helpers); read it there rather than a copy that drifts.\n"
            "; Host-specific values in effect for this run:\n"
            f";   KV cache upgrade below {c.small_kv_gb} GB: {c.small_kv_qwen} (Qwen) / {c.small_kv_other} (others)\n"
            f";   above {c.ckpt_gb} GB: ctx-checkpoints = {c.ckpt_checkpoints}, ctx-size = {c.ckpt_ctx_size}\n"
            f";   Qwen {c.qwen_template_min[0]}.{c.qwen_template_min[1]}+ chat-template-file = {c.qwen_template}\n"
            "; Not listed here: diffusion-arch models (router cannot load them - see\n"
            "; .generated/diffusion-models.tsv, run under llama-diffusion-cli) and DFlash\n"
            "; drafts (wire manually in the base ini via spec-type = draft-dflash).\n"
            "; ===================================================================="
        )

        if mtp:
            parts.append(
                "\n\n; --- embedded MTP (self-draft; inherits global spec-type) ---"
            )
            for m in mtp:
                parts.append(f"\n\n[{m.name}]\n{emit(m)}".rstrip("\n"))

        if paired:
            parts.append("\n\n; --- base models paired with a separate MTP head ---")
            for m in paired:
                parts.append(f"\n\n[{m.name}]\n{emit(m)}".rstrip("\n"))

        if ngram:
            parts.append(
                "\n\n; --- no MTP: model-free ngram speculation fallback ---\n"
                "; spec-type = ngram-mod replaces the inherited draft-mtp\n"
                "; (which would abort the load); no draft model needed, drafts from\n"
                "; recent-context n-grams."
            )
            for m in ngram:
                parts.append(f"\n\n[{m.name}]\n{emit(m)}".rstrip("\n"))

        if nospec:
            parts.append("\n\n; --- no speculation (embedding models) ---")
            for m in nospec:
                parts.append(f"\n\n[{m.name}]\n{emit(m)}".rstrip("\n"))

        c.router_ini.write_text("".join(parts) + "\n")

    def _summary(self) -> None:
        mtp = sum(1 for m in self.models if m.kind is ModelKind.EMBEDDED_MTP)
        paired = sum(1 for m in self.models if m.kind is ModelKind.PAIRED)
        ngram = sum(1 for m in self.models if m.kind is ModelKind.NGRAM)
        nospec = sum(1 for m in self.models if m.kind is ModelKind.NO_SPEC)
        size_tiered = self.size_tiered
        print(
            f"linked {self.linked} model(s): {mtp} embedded-MTP, {paired} base+head paired, "
            f"{ngram} ngram fallback, {nospec} no-spec (embeddings), {size_tiered} size-tiered, "
            f"{self.templated} Qwen chat-template fix, "
            f"{self.nommproj_twins} no-mmproj twin(s); skipped {self.skipped_heads} MTP head(s), "
            f"{self.skipped_dflash} DFlash draft(s), "
            f"{len(self.diffusion)} diffusion model(s) (run via llama-diffusion-cli)."
        )
        print(f"router preset: {self.cfg.router_ini}")

    def run(self) -> None:
        self._update_template_repo()
        self._check_src()
        self._clean_dest()
        self._index_heads()
        self._build_farm()
        self.prober.save_cache()
        self._write_diffusion_tsv()
        self._write_router_ini()
        self._summary()


def parse_args() -> argparse.Namespace:
    summary = (__doc__ or "").splitlines()[0]
    ap = argparse.ArgumentParser(
        description=summary, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--src", type=Path, help="source model tree to scan (env MODELS_SRC)"
    )
    ap.add_argument("--dest", type=Path, help="flat staging dir (env MODELS_DIR)")
    ap.add_argument(
        "--llama-gguf", dest="llama_gguf", help="llama-gguf binary (env LLAMA_GGUF)"
    )
    ap.add_argument(
        "--small-kv-gb",
        dest="small_kv_gb",
        type=int,
        help="KV-upgrade threshold, GB (env SIZE_SMALL_KV_GB)",
    )
    ap.add_argument(
        "--ckpt-gb",
        dest="ckpt_gb",
        type=int,
        help="ctx-checkpoint threshold, GB (env SIZE_CKPT_GB)",
    )
    ap.add_argument(
        "--qwen-template",
        dest="qwen_template",
        type=Path,
        help="chat template jinja for Qwen 3.5+ (env QWEN_CHAT_TEMPLATE)",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.load(**{k: v for k, v in vars(args).items() if v is not None})
    Sync(cfg).run()


if __name__ == "__main__":
    main()
