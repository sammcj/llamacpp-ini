#!/usr/bin/env python3
"""Prepare ~/.lmstudio/models for `llama-server --models-dir` + wire up MTP.

Problems solved:

1. llama-server only scans the TOP level of --models-dir (loose *.gguf = a
   single-file model, each immediate sub-dir = one multi-file model). It does
   not recurse, so LM Studio's nested <publisher>/<repo>/<file>.gguf layout is
   invisible. We flatten it into a symlink farm. Every quant is exposed as its
   own model, named after its GGUF file (sharded models, one model split across
   files, are named after their dir). A model name never depends on how many
   siblings share its folder: the web UI keys favourites on the name, so
   folder-derived names would silently rename - and unfavourite - a model as
   soon as a second quant landed beside it.

2. The local scanner does NO MTP discovery. We probe every GGUF with llama-gguf
   and split models three ways:
     - embedded MTP  (nextn tensors in a full model, e.g. Qwen): the server
       self-drafts from the model itself -> inherits [*] spec-type=draft-mtp.
     - base + head   (no nextn in the model, but a separate MTP head exists,
       e.g. Gemma 4): we pair them via `spec-draft-model = <head>`.
     - no MTP        (no nextn, no matching head): generative models fall back to
       ngram speculation; embedding models get spec-type=none, since a global
       draft-mtp would abort the load.
   A standalone MTP head (a stub GGUF: nextn tensors, only a handful of
   transformer blocks) is NOT a runnable model and is never linked as one.
   Results are written to .generated/router.ini, the preset the server loads.

3. Diffusion-arch GGUFs (diffusion-gemma, dream, llada, ...) cannot load in
   llama-server at all. We keep them out of the farm and list them in
   .generated/diffusion-models.tsv for use with llama-diffusion-cli.

Nothing is copied; only symlinks. Re-run whenever models change.

Configuration lives in one place: the Config dataclass below. Every field can
be overridden by its env var (shown in the field metadata) or a matching CLI
flag; precedence is CLI > env > default. The size tiers are data too - see
SIZE_TIERS - so changing what a tier does is a one-line edit.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field, fields
from enum import Enum, auto
from pathlib import Path
from typing import Callable

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
    src: Path = field(default=Path.home() / ".lmstudio" / "models", metadata={"env": "MODELS_SRC", "conv": Path})
    dest: Path = field(default=SCRIPT_DIR / "models", metadata={"env": "MODELS_DIR", "conv": Path})
    llama_gguf: str = field(default="llama-gguf", metadata={"env": "LLAMA_GGUF", "conv": str})

    # --- MTP / architecture detection ----------------------------------------
    head_max_blocks: int = 8  # nextn tensors + <= this many blocks == head stub, not a model
    diffusion_arch: tuple[str, ...] = ("diffusion", "dream", "llada")  # only llama-diffusion-cli runs these
    cache_version: int = 3  # probe-cache schema; bump to invalidate old caches

    # --- Size-tier thresholds, in GB (env: SIZE_SMALL_KV_GB / SIZE_CKPT_GB) ---
    # They stack: a model can match more than one tier. See SIZE_TIERS for wiring.
    small_kv_gb: int = field(default=40, metadata={"env": "SIZE_SMALL_KV_GB", "conv": int})  # under this: upgrade KV cache
    ckpt_gb: int = field(default=80, metadata={"env": "SIZE_CKPT_GB", "conv": int})  # over this: cap ctx + checkpoints

    # --- What each size tier writes (the override values themselves) ----------
    small_kv_qwen: str = "bf16"  # KV cache type for Qwen under small_kv_gb (Qwen handles bf16 well)
    small_kv_other: str = "f16"  # KV cache type for every other model under small_kv_gb
    ckpt_checkpoints: str = "32"  # ctx-checkpoints for models over ckpt_gb
    ckpt_ctx_size: int = 65000  # ctx-size for models over ckpt_gb

    # --- Chat template fix for Qwen 3.5+ (env: QWEN_CHAT_TEMPLATE) ------------
    # The templates shipped inside Qwen 3.5+ GGUFs are broken (tool calls and
    # thinking blocks); point those models at the corrected jinja instead.
    qwen_template: Path = field(
        default=Path.home() / "git" / "Qwen-Fixed-Chat-Templates" / "chat_template.jinja",
        metadata={"env": "QWEN_CHAT_TEMPLATE", "conv": Path},
    )
    qwen_template_min: tuple[int, int] = (3, 5)  # lowest Qwen version that gets it

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

    @classmethod
    def load(cls, **cli_overrides) -> "Config":
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
        lambda _n, c: {"ctx-checkpoints": c.ckpt_checkpoints, "ctx-size": str(c.ckpt_ctx_size)},
    ),
]


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
    versions = [(int(maj), int(minor or 0)) for maj, minor in QWEN_VERSION_RE.findall(name)]
    return max(versions) if versions else None


def template_overrides(name: str, cfg: Config) -> dict[str, str]:
    ver = qwen_version(name)
    if ver is None or ver < cfg.qwen_template_min:
        return {}
    return {"chat-template-file": str(cfg.qwen_template)}


# --------------------------------------------------------------------------
# GGUF probing (cached by path|size:mtime, TSV-compatible with the bash cache).
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ProbeResult:
    nextn: bool  # has nextn tensors (embedded MTP or a head stub)
    head: bool  # nextn tensors + few blocks -> a standalone MTP head, not a model
    diffusion: bool  # arch only runs under llama-diffusion-cli
    embed: bool  # carries <arch>.pooling_type -> an embedding model


class Prober:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.have_gguf = shutil.which(cfg.llama_gguf) is not None
        if not self.have_gguf:
            print(f"warning: '{cfg.llama_gguf}' not found; MTP detection disabled.", file=sys.stderr)
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
            self.cache[key] = val
        cnt, blk, dif, emb = self._parse(val)
        nextn = cnt > 0
        return ProbeResult(
            nextn=nextn,
            head=nextn and blk <= self.cfg.head_max_blocks,
            diffusion=dif > 0,
            embed=emb > 0,
        )

    def _probe_raw(self, f: Path) -> str:
        cnt = blk = dif = emb = 0
        if self.have_gguf:
            out = self._run_gguf(f)
            cnt = sum(1 for ln in out.splitlines() if re.search(r"name = .*nextn", ln))
            blocks = [int(m) for m in re.findall(r"blk\.(\d+)\.", out)]
            blk = max(blocks) if blocks else 0
            am = re.search(r"[a-z0-9_-]+\.block_count", out)
            arch = am.group(0)[: -len(".block_count")] if am else ""
            if arch.startswith(self.cfg.diffusion_arch):
                dif = 1
            if any(".pooling_type" in ln for ln in out.splitlines()):
                emb = 1
        return f"{cnt}:{blk}:{dif}:{emb}"

    def _run_gguf(self, f: Path) -> str:
        try:
            r = subprocess.run(
                [self.cfg.llama_gguf, str(f), "r", "n"],
                capture_output=True,
                text=True,
            )
            return r.stdout or ""
        except OSError:
            return ""

    @staticmethod
    def _parse(val: str) -> tuple[int, int, int, int]:
        # Pad older 2/3-field cache entries with zeros; treat any garbage as 0
        # (a corrupt cache degrades to "unknown", it never crashes the run).
        def to_int(x: str) -> int:
            try:
                return int(x)
            except ValueError:
                return 0

        p = (val.split(":") + ["0", "0", "0", "0"])[:4]
        return (to_int(p[0]), to_int(p[1]), to_int(p[2]), to_int(p[3]))

    def save_cache(self) -> None:
        lines = [f"{k}\t{v}" for k, v in sorted(self.cache.items())]
        self.cfg.cache_path.write_text("".join(f"{ln}\n" for ln in lines))


# --------------------------------------------------------------------------
# Model records. The five-way classification is the enum; spec-draft-model only
# exists for PAIRED, so it hangs off the record, not a side table.
# --------------------------------------------------------------------------
class ModelKind(Enum):
    EMBEDDED_MTP = auto()  # self-drafts; inherits the global spec-type
    PAIRED = auto()  # base wired to a separate MTP head via spec-draft-model
    NGRAM = auto()  # no MTP: model-free ngram speculation fallback
    NO_SPEC = auto()  # embedding model; speculation is meaningless


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
    """
    s = (name[:-5] if name.endswith(".gguf") else name).lower()
    s = re.sub(r"-[0-9]+-of-[0-9]+", "", s)
    s = re.sub(r"[._-]q[0-9]+(_[0-9a-z]+)*", "", s)
    s = re.sub(r"[._-](ud|k_xl|k_m|k_s|k_l|f16|bf16|fp16|f32|i1|imatrix)", "", s)
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
    """Every real *.gguf under src (mirrors `find -L -type f`). followlinks so a
    symlinked src tree (e.g. ~/.lmstudio/models -> ~/.cache/lm-studio/models) or
    symlinked publisher dirs are scanned; the farm's own symlinks are excluded by
    the is_symlink() guard below."""
    for root, _, files in os.walk(src, followlinks=True):
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
        self.nommproj_twins = 0
        self.templated = 0
        self.warned_template = False

    # -- classification helpers ------------------------------------------------
    def _is_head(self, g: Path) -> bool:
        if self.prober.have_gguf:
            return self.prober.probe(g).head
        # Fall back to the mtp- filename convention when llama-gguf is absent.
        return g.name.startswith("mtp-") and g.name.endswith(".gguf")

    def _is_diffusion(self, g: Path) -> bool:
        return self.prober.have_gguf and self.prober.probe(g).diffusion

    def _template_overrides(self, name: str) -> dict[str, str]:
        """Qwen 3.5+ chat-template fix, dropped if the jinja file is absent -
        pointing chat-template-file at a missing path aborts the model load."""
        out = template_overrides(name, self.cfg)
        if out and not self.cfg.qwen_template.is_file():
            if not self.warned_template:
                print(
                    f"warning: chat template '{self.cfg.qwen_template}' not found; "
                    "Qwen 3.5+ models keep their built-in template.",
                    file=sys.stderr,
                )
                self.warned_template = True
            return {}
        if out:
            self.templated += 1
        return out

    def _unique_name(self, desired: str, suffix: str, prefix: str) -> str:
        """A free farm name; fall back to <prefix>_<desired> on collision. The
        first to claim a bare name keeps it, so callers must iterate in a stable
        order (see build_farm) or a rename would orphan a web-UI favourite.
        """
        target = self.cfg.dest / f"{desired}{suffix}"
        if target.exists() or target.is_symlink():
            return f"{prefix}_{desired}"
        return desired

    # -- model finalisation ----------------------------------------------------
    def _finalise(self, name: str, match: str, files: list[Path]) -> Model:
        """Probe the model's file(s), record size-tier keys off the summed weight
        bytes (mmproj is never passed in), then classify it for MTP.
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
        overrides.update(self._template_overrides(name))

        if embedded:
            kind, tags = ModelKind.EMBEDDED_MTP, tags + ["MTP"]
            head = None
        else:
            head = self.head_index.get(normalise(match))
            if head is not None:
                kind, tags = ModelKind.PAIRED, tags + ["MTP"]
            elif is_embed:
                kind = ModelKind.NO_SPEC
            else:
                kind, tags = ModelKind.NGRAM, tags + ["ngram"]

        model = Model(name=name, kind=kind, tags=tags, draft_head=head, overrides=overrides)
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
        self._finalise(twin, files[0].name, files)

    # -- passes ----------------------------------------------------------------
    def _check_src(self) -> None:
        if not self.cfg.src.is_dir():
            sys.exit(f"error: source models directory '{self.cfg.src}' not found.")
        self.cfg.dest.mkdir(parents=True, exist_ok=True)
        self.cfg.gen_dir.mkdir(parents=True, exist_ok=True)

    def _clean_dest(self) -> None:
        """Remove what we generated: top-level symlinks, plus per-quant/twin dirs
        (those whose contents are entirely symlinks). Never touch real files or
        directories the user placed here.
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
            self.head_index[key] = min(heads, key=lambda p: (not p.name.startswith("mtp-"), p.name))

    def _build_farm(self) -> None:
        """Pass 2: build the model farm and classify each model for MTP.

        Sort the dirs: directory-walk order is unspecified, and _unique_name's
        collision fallback (first one wins the bare name) depends on a stable
        order - unsorted, adding or removing a model could silently rename
        another, and the web UI keys favourites on the model name.
        """
        ggufs = sorted(iter_ggufs(self.cfg.src))
        for d in sorted({p.parent for p in ggufs}):
            if d == self.cfg.src:
                self._process_toplevel(d)
            else:
                self._process_dir(d)

    def _process_toplevel(self, d: Path) -> None:
        """Loose *.gguf at the top of the source tree: each is its own model."""
        for g in sorted(d.glob("*.gguf")):
            if self._is_head(g):
                self.skipped_heads += 1
                continue
            if self._is_diffusion(g):
                self.diffusion.append((g.stem, g))
                continue
            name = self._unique_name(g.stem, ".gguf", "root")
            force_symlink(g, self.cfg.dest / f"{name}.gguf")
            self.linked += 1
            self._finalise(name, g.name, [g])

    def _process_dir(self, d: Path) -> None:
        ggufs = sorted(d.glob("*.gguf"))
        if not ggufs:
            return

        mmproj: Path | None = None
        first_shard: Path | None = None
        shards: list[Path] = []
        quants: list[Path] = []
        for g in ggufs:
            base = g.name
            if "mmproj" in base:
                mmproj = g
            elif "-of-" in base:
                shards.append(g)
                if "-00001-of-" in base:
                    first_shard = g
            elif self._is_head(g):
                self.skipped_heads += 1
            else:
                quants.append(g)

        parent = d.parent.name

        if first_shard is not None:
            # Sharded model == one model. Link the dir; probe every shard.
            if self._is_diffusion(first_shard):
                self.diffusion.append((d.name, first_shard))
                return
            name = self._unique_name(d.name, "", parent)
            force_symlink(d, self.cfg.dest / name)
            self.linked += 1
            model = self._finalise(name, first_shard.name, shards)
            if mmproj is not None:
                model.tags.append("vision")
                self._link_twin(name, shards)
            return

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
        self.cfg.diffusion_tsv.write_text("".join(f"{n}\t{p}\n" for n, p in self.diffusion))

    def _write_router_ini(self) -> None:
        c = self.cfg
        mtp = [m for m in self.models if m.kind is ModelKind.EMBEDDED_MTP]
        paired = [m for m in self.models if m.kind is ModelKind.PAIRED]
        ngram = [m for m in self.models if m.kind is ModelKind.NGRAM]
        nospec = [m for m in self.models if m.kind is ModelKind.NO_SPEC]

        def emit(m: Model) -> str:
            out = ""
            if m.tags:
                out += f"tags = {','.join(m.tags)}\n"
            out += "".join(f"{k.ljust(16)} = {v}\n" for k, v in m.overrides.items())
            return out

        parts: list[str] = [c.base_ini.read_text(), "\n"]
        parts.append(
            "; ====================================================================\n"
            f"; AUTO-GENERATED by {PROG} - do not edit; re-run to refresh.\n"
            "; MTP is ON by default ([*] spec-type = draft-mtp). MTP-active models also\n"
            "; get tags = MTP so the web UI badges them (the picker merges server tags\n"
            "; with name-parsed tokens). Embedded-MTP (e.g. Qwen) need only the tag;\n"
            "; base+head models (e.g. Gemma) also need spec-draft-model. Generative models\n"
            "; with no MTP fall back to ngram speculation (spec-default, tags = ngram);\n"
            "; embedding models get spec-type = none (speculation is meaningless there).\n"
            f"; Size tiers, merged into the model's own section: under {c.small_kv_gb} GB -> KV cache\n"
            f"; {c.small_kv_qwen} (Qwen) or {c.small_kv_other} (others); over {c.ckpt_gb} GB ->\n"
            f"; ctx-checkpoints = {c.ckpt_checkpoints} and ctx-size = {c.ckpt_ctx_size}.\n"
            f"; Qwen {c.qwen_template_min[0]}.{c.qwen_template_min[1]} and newer (version parsed from the model name) get\n"
            f"; chat-template-file = {c.qwen_template}, replacing the broken template\n"
            "; baked into those GGUFs.\n"
            "; Each multimodal model (one with an mmproj) is tagged vision and also gets a\n"
            "; -no-mmproj twin: a weights-only farm entry that loads text-only (no vision\n"
            "; tag), with the same spec/size keys. Every model is also tagged with its quant\n"
            "; (Q6_K, UD-Q5_K_XL, ...) parsed from the GGUF filename, so the picker shows it.\n"
            "; Tags are comma-joined (e.g. Q6_K,MTP,vision).\n"
            "; Diffusion-arch models are excluded here (the router cannot load them);\n"
            "; see .generated/diffusion-models.tsv, run them with llama-diffusion-cli.\n"
            "; ===================================================================="
        )

        if mtp:
            parts.append("\n\n; --- embedded MTP (self-draft; inherits global spec-type) ---")
            for m in mtp:
                parts.append(f"\n\n[{m.name}]\n{emit(m)}".rstrip("\n"))

        if paired:
            parts.append("\n\n; --- base models paired with a separate MTP head ---")
            for m in paired:
                body = f"spec-type        = draft-mtp\nspec-draft-model = {m.draft_head}\n{emit(m)}"
                parts.append(f"\n\n[{m.name}]\n{body}".rstrip("\n"))

        if ngram:
            parts.append(
                "\n\n; --- no MTP: model-free ngram speculation fallback (--spec-default) ---\n"
                "; spec-type = none clears the inherited draft-mtp; spec-default adds the\n"
                "; ngram-mod config (no draft model needed, drafts from prompt n-grams)."
            )
            for m in ngram:
                body = f"spec-type    = none\nspec-default = true\n{emit(m)}"
                parts.append(f"\n\n[{m.name}]\n{body}".rstrip("\n"))

        if nospec:
            parts.append("\n\n; --- no speculation (embedding models) ---")
            for m in nospec:
                body = f"spec-type = none\n{emit(m)}"
                parts.append(f"\n\n[{m.name}]\n{body}".rstrip("\n"))

        c.router_ini.write_text("".join(parts) + "\n")

    def _summary(self) -> None:
        mtp = sum(1 for m in self.models if m.kind is ModelKind.EMBEDDED_MTP)
        paired = sum(1 for m in self.models if m.kind is ModelKind.PAIRED)
        ngram = sum(1 for m in self.models if m.kind is ModelKind.NGRAM)
        nospec = sum(1 for m in self.models if m.kind is ModelKind.NO_SPEC)
        size_tiered = sum(1 for m in self.models if m.overrides)
        print(
            f"linked {self.linked} model(s): {mtp} embedded-MTP, {paired} base+head paired, "
            f"{ngram} ngram fallback, {nospec} no-spec (embeddings), {size_tiered} size-tiered, "
            f"{self.templated} Qwen chat-template fix, "
            f"{self.nommproj_twins} no-mmproj twin(s); skipped {self.skipped_heads} MTP head(s), "
            f"{len(self.diffusion)} diffusion model(s) (run via llama-diffusion-cli)."
        )
        print(f"router preset: {self.cfg.router_ini}")

    def run(self) -> None:
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
    ap = argparse.ArgumentParser(description=summary, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, help="source model tree to scan (env MODELS_SRC)")
    ap.add_argument("--dest", type=Path, help="flat staging dir (env MODELS_DIR)")
    ap.add_argument("--llama-gguf", dest="llama_gguf", help="llama-gguf binary (env LLAMA_GGUF)")
    ap.add_argument("--small-kv-gb", dest="small_kv_gb", type=int, help="KV-upgrade threshold, GB (env SIZE_SMALL_KV_GB)")
    ap.add_argument("--ckpt-gb", dest="ckpt_gb", type=int, help="ctx-checkpoint threshold, GB (env SIZE_CKPT_GB)")
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
