# infra.py
"""
infra.py

Embedder-agnostisk infrastruktur för:
- PROJECT_ROOT / RUNS_ROOT / RUN_DIR
- RunPaths (standardiserade subfolders & filnamn per run)
- I/O helpers (CSV/PKL/JSON)
- Lätt provenance (checksums + log)

Viktigt:
- Embeddings-/encoder-logik och cache-format ligger INTE här.
  (Det ligger i embeddings.py + cache.py.)

Run-kontext:
- RUN_DIR / RP uppdateras explicit via set_run_tag(...), init_embeddings_run(...),
  eller activate_last_run().
- Ingen implicit "auto-activate" vid import (för att undvika spök-runs).

Revidering 3:
- init_embeddings_run(...) och load_embeddings_run(...) (widget-fritt alternativ)
- embeddings_run_widget(...) använder encoders-API (inte embeddings-API)
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping, Union

import pandas as pd


# ─────────────────────────────────────────────────────────────
# Logging helper
# ─────────────────────────────────────────────────────────────

def log(*msg: object) -> None:
    print("•", *msg)


# ─────────────────────────────────────────────────────────────
# .env loading (early)
# ─────────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    """
    Load .env early so PROJECT_ROOT and API keys exist before infra resolves paths.
    No-op if python-dotenv is not installed.
    """
    try:
        from dotenv import find_dotenv, load_dotenv  # type: ignore
    except Exception:
        return

    fp = find_dotenv(".env", usecwd=True)
    if fp:
        load_dotenv(fp, override=False)
        return

    fp = find_dotenv(".env", usecwd=False)
    if fp:
        load_dotenv(fp, override=False)

_load_dotenv()


# ─────────────────────────────────────────────────────────────
# Project roots (strict; no auto-detection)
# ─────────────────────────────────────────────────────────────

def _require_env(name: str) -> str:
    v = (os.environ.get(name) or "").strip()
    if not v:
        raise RuntimeError(
            f"Missing required env var: {name}. "
            f"Set it in .env or in the environment before importing infra."
        )
    return v


def _resolve_existing_dir(p: str, *, var_name: str) -> Path:
    pp = Path(p).expanduser().resolve()
    if not pp.exists():
        raise RuntimeError(f"{var_name} points to non-existent path: {pp}")
    if not pp.is_dir():
        raise RuntimeError(f"{var_name} is not a directory: {pp}")
    return pp


# Resolve PROJECT_ROOT from env (.env loaded above)
PROJECT_ROOT: Path = _resolve_existing_dir(_require_env("PROJECT_ROOT"), var_name="PROJECT_ROOT")
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

# Everything goes under PROJECT_ROOT/out by default
OUT_ROOT: Path = Path(os.environ.get("OUT_ROOT", str(PROJECT_ROOT / "out"))).expanduser().resolve()
OUT_ROOT.mkdir(parents=True, exist_ok=True)

RUNS_ROOT: Path = Path(os.environ.get("RUNS_ROOT", str(OUT_ROOT / "runs"))).expanduser().resolve()
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

GLOBAL_CACHE_ROOT: Path = Path(
    os.environ.get("PROJECT_GLOBAL_CACHE_ROOT", str(OUT_ROOT / "_cache"))
).expanduser().resolve()
GLOBAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# RunPaths
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunPaths:
    """
    Notebook-facing locator.

    Struktur:
      <PROJECT_ROOT>/
        out/
          runs/<run_tag>/
            exports/
            figures/
            logs/
            cache/        (run-local cache root)
        _cache/           (global cache root)
    """
    run_tag: str

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def runs_root(self) -> Path:
        return RUNS_ROOT

    @property
    def global_cache_root(self) -> Path:
        return GLOBAL_CACHE_ROOT

    @property
    def run_dir(self) -> Path:
        return (self.runs_root / self.run_tag).resolve()

    @property
    def exports(self) -> Path:
        return self.run_dir / "exports"

    @property
    def figures(self) -> Path:
        return self.run_dir / "figures"

    @property
    def models(self) -> Path:
        return self.run_dir / "models"

    @property
    def logs(self) -> Path:
        return self.run_dir / "logs"

    @property
    def cache(self) -> Path:
        return self.run_dir / "cache"

    def ensure_dirs(self) -> None:
        for d in [self.run_dir, self.exports, self.figures, self.models, self.logs, self.cache]:
            d.mkdir(parents=True, exist_ok=True)

    def cache_fp(self, name: str) -> Path:
        self.cache.mkdir(parents=True, exist_ok=True)
        return self.cache / name
    
    def export_fp(self, name: str) -> Path:
        self.exports.mkdir(parents=True, exist_ok=True)
        return self.exports / name

    def figure_fp(self, name: str) -> Path:
        self.figures.mkdir(parents=True, exist_ok=True)
        return self.figures / name

    def model_fp(self, name: str) -> Path:
        self.models.mkdir(parents=True, exist_ok=True)
        return self.models / name

    def log_fp(self, name: str) -> Path:
        self.logs.mkdir(parents=True, exist_ok=True)
        return self.logs / name

    def mkpath(self, *parts: str) -> Path:
        """
        Create/write path inside run_dir. Ensures parent dirs exist.
        Example: RP.mkpath("exports", "tasks.csv")
        """
        p = self.run_dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


# ─────────────────────────────────────────────────────────────
# Run tag helpers
# ─────────────────────────────────────────────────────────────

_TAG_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_TAG_MULTI_DUNDER_RE = re.compile(r"__+")


def _clean_tag_component(s: str) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    if not t:
        return ""
    t = re.sub(r"\s+", "_", t)                 # whitespace → _
    t = _TAG_SAFE_RE.sub("_", t)               # illegal → _
    t = re.sub(r"_{3,}", "__", t)              # 3+ underscores → "__"
    t = t.strip("._- ")                        # trim edges
    return t


def make_run_tag(*parts: str, prefix: str | None = None) -> str:
    items = [_clean_tag_component(p) for p in parts]
    items = [p for p in items if p]
    if prefix:
        items.insert(0, _clean_tag_component(prefix))
    tag = "__".join(items)
    tag = _TAG_MULTI_DUNDER_RE.sub("__", tag).strip("_")
    return tag


# ─────────────────────────────────────────────────────────────
# Persistent last selected run (optional)
# ─────────────────────────────────────────────────────────────

_RUN_STATE_FP = OUT_ROOT / "last_run.json"


def load_last_run_tag() -> str | None:
    try:
        if not _RUN_STATE_FP.exists():
            return None
        obj = json.loads(_RUN_STATE_FP.read_text(encoding="utf-8"))
        tag = _clean_tag_component((obj.get("run_tag") or "").strip())
        return tag or None
    except Exception:
        return None


def save_last_run_tag(run_tag: str | None) -> None:
    _RUN_STATE_FP.parent.mkdir(parents=True, exist_ok=True)
    tag = _clean_tag_component(run_tag or "")
    payload = {
        "run_tag": (tag or None),
        "saved_local": datetime.now().isoformat(timespec="seconds"),
    }
    _RUN_STATE_FP.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────
# Run context (explicit activation only)
# ─────────────────────────────────────────────────────────────

RUN_TAG: str | None = None
RUN_DIR: Path | None = None
RP: RunPaths | None = None


def _require_active_run() -> tuple[str, Path, RunPaths]:
    if RUN_TAG is None or RUN_DIR is None or RP is None:
        raise RuntimeError("No active run. Call init_embeddings_run(...), set_run_tag(...), or activate_last_run().")
    return RUN_TAG, RUN_DIR, RP


def _resolve_run_dir_for_tag(tag: str) -> Path:
    """
    Resolve/create a run directory for a *non-empty, sanitized* tag.
    """
    t = _clean_tag_component(tag)
    if not t:
        raise ValueError("run_tag became empty after sanitizing")
    p = RUNS_ROOT / t
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_run_tag(
    run_tag: str,
    *,
    echo: bool = True,
    persist: bool = True,
    ensure_dirs: bool = True,
) -> RunPaths:
    """
    Activate a run tag (creates run dir if needed).
    """
    global RUN_TAG, RUN_DIR, RP

    tag = _clean_tag_component(run_tag)
    if not tag:
        raise ValueError("set_run_tag: empty tag after sanitizing")

    RUN_TAG = tag
    RUN_DIR = _resolve_run_dir_for_tag(tag)

    RP = RunPaths(run_tag=RUN_DIR.name)
    if ensure_dirs:
        RP.ensure_dirs()

    os.environ["RUN_TAG"] = RUN_TAG
    if persist:
        save_last_run_tag(RUN_TAG)

    if echo:
        print(f"PROJECT_ROOT = {PROJECT_ROOT}")
        print(f"RUNS_ROOT    = {RUNS_ROOT}")
        print(f"RUN_TAG      = {RUN_TAG}")
        print(f"RUN_DIR      = {RUN_DIR}")

    return RP


def activate_last_run(*, echo: bool = False) -> RunPaths:
    """
    Activate last run explicitly.

    Resolution order:
      1) env RUN_TAG
      2) last_run.json

    Strict:
      - If no tag found -> error
      - If tag points to missing/non-dir -> error
    """
    tag_env = (os.getenv("RUN_TAG", "").strip() or None)
    tag_last = load_last_run_tag()
    tag = tag_env or tag_last
    if not tag:
        raise RuntimeError("No last run available: neither RUN_TAG env nor last_run.json is set.")

    tag = _clean_tag_component(tag)
    if not tag:
        raise RuntimeError("Last run tag became empty after sanitizing.")

    run_dir = RUNS_ROOT / tag
    if not run_dir.exists() or not run_dir.is_dir():
        raise RuntimeError(f"Last run tag points to missing directory: {run_dir}")

    return set_run_tag(tag, echo=echo, persist=False)


# ─────────────────────────────────────────────────────────────
# Run listing helpers
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunInfo:
    tag: str
    path: Path
    mtime: float


def list_runs(*, prefix: str | None = None) -> list[RunInfo]:
    runs: list[RunInfo] = []
    if not RUNS_ROOT.exists():
        return runs
    for p in RUNS_ROOT.iterdir():
        if not p.is_dir():
            continue
        tag = p.name
        if prefix and not tag.startswith(prefix):
            continue
        try:
            mtime = p.stat().st_mtime
        except Exception:
            mtime = 0.0
        runs.append(RunInfo(tag=tag, path=p, mtime=mtime))
    runs.sort(key=lambda r: r.mtime, reverse=True)
    return runs


# ─────────────────────────────────────────────────────────────
# Per-run config (run_config.json)
# ─────────────────────────────────────────────────────────────

def _run_config_path(run_dir: Path) -> Path:
    return run_dir / "run_config.json"


def run_config_fp() -> Path:
    _, run_dir, _ = _require_active_run()
    return _run_config_path(run_dir)


def _read_run_config_file(fp: Path) -> dict[str, Any]:
    obj = json.loads(fp.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("run_config.json is not a dict")
    return obj


def read_run_config(*, default: dict[str, Any] | None = None) -> dict[str, Any]:
    fp = run_config_fp()
    if not fp.exists():
        return dict(default or {})
    try:
        return _read_run_config_file(fp)
    except Exception:
        return dict(default or {})


def write_run_config(cfg: Mapping[str, Any], *, overwrite: bool = True) -> Path:
    fp = run_config_fp()
    if fp.exists() and not overwrite:
        raise FileExistsError(str(fp))
    payload = dict(cfg)
    payload.setdefault("saved_local", datetime.now().isoformat(timespec="seconds"))
    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return fp


def _is_run_dirty(run_dir: Path) -> bool:
    """
    'Dirty' = run innehåller artefakter utöver en ev run_config + tomma standardmappar.
    Konservativ check: om exports/figures/logs/cache innehåller något, eller om
    andra filer än run_config.json finns i run_dir-root.
    """
    if not run_dir.exists():
        return False

    for p in run_dir.iterdir():
        if p.name == "run_config.json":
            continue

        if p.is_dir() and p.name in {"exports", "figures", "models", "logs", "cache"}:
            if any(p.rglob("*")):
                return True
            continue

        return True

    return False


def _cfg_diff(existing: dict[str, Any], expected: dict[str, Any], keys: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for k in keys:
        ev = existing.get(k, None)
        xv = expected.get(k, None)
        if str(ev) != str(xv):
            out[k] = {"existing": ev, "expected": xv}
    return out


# ─────────────────────────────────────────────────────────────
# Run init status + errors
# ─────────────────────────────────────────────────────────────

RunConfigState = Literal["missing", "invalid", "ok"]
RunMatchState = Literal["match", "mismatch", "unknown"]  # unknown if config missing/invalid
RunAction = Literal[
    "reuse",                 # reused existing run (match)
    "create_new",            # created new run
    "repair_config",         # wrote config because missing/invalid and run is empty
    "use_existing_config",   # used existing run + its config as truth
    "error",                 # raised (or would raise)
]


@dataclass
class RunStatus:
    requested_tag: str
    resolved_tag: str
    run_dir: Path

    exists: bool
    config_state: RunConfigState
    match_state: RunMatchState
    dirty: bool

    expected_cfg: dict[str, Any] = field(default_factory=dict)
    existing_cfg: dict[str, Any] = field(default_factory=dict)
    diff: dict[str, dict[str, Any]] = field(default_factory=dict)

    action: RunAction = "reuse"
    note: str = ""


class RunInitError(RuntimeError):
    def __init__(self, message: str, status: RunStatus):
        super().__init__(message)
        self.status = status


# ─────────────────────────────────────────────────────────────
# Embeddings-run helpers (widget-fritt API)
# ─────────────────────────────────────────────────────────────

def _expected_embeddings_run_config(
    *,
    run_tag: str,
    prefix: str,
    year: int,
    onet_version: str,
    encoder_name: str,
    encoder_fragment: str,
    note: str,
) -> dict[str, Any]:
    return {
        "kind": "embeddings_run",
        "prefix": prefix,
        "run_tag": run_tag,
        "year": int(year),
        "onet_version": str(onet_version),
        "encoder_name": str(encoder_name),
        "encoder_fragment": str(encoder_fragment),
        "note": (note or "").strip(),
    }


def init_embeddings_run(
    *,
    year: int,
    onet_version: str,
    encoder_name: str,
    prefix: str = "embeddings",
    note: str = "",
    append_timestamp_on_new: bool = True,
    persist: bool = True,
    ensure_dirs: bool = True,
    return_status: bool = False,
):
    """
    Ensure-or-create embeddings run.

    Default-semantik (för notebook-simplicitet):
    - Bygg deterministisk base_tag (utan timestamp)
    - Om base_tag saknas -> skapa base_tag + skriv run_config.json
    - Om base_tag finns och run_config matchar -> reuse (rör inte config)
    - Om base_tag finns men run_config saknas/invalid -> error
    - Om base_tag finns men mismatch -> skapa ny run med timestamp (om append_timestamp_on_new)

    Notebooken behöver inget policyobjekt. Om du vill ändra beteende i framtiden,
    gör det här (centraliserat), inte i notebooks.
    """
    import encoders as _enc

    spec = _enc.get_encoder(encoder_name)
    note2 = (note or "").strip()
    v_norm = str(onet_version).replace(".", "_")

    base_tag = make_run_tag(
        *([note2] if note2 else []),
        spec.run_tag_fragment(),
        f"year-{int(year)}",
        f"v{v_norm}",
        prefix=prefix,
    )
    base_dir = RUNS_ROOT / base_tag
    exists = base_dir.exists()

    expected_cfg = _expected_embeddings_run_config(
        run_tag=base_tag,
        prefix=prefix,
        year=year,
        onet_version=str(onet_version),
        encoder_name=str(encoder_name),
        encoder_fragment=spec.run_tag_fragment(),
        note=note2,
    )

    cfg_path = _run_config_path(base_dir)
    existing_cfg: dict[str, Any] = {}
    config_state: RunConfigState = "missing"

    if exists and cfg_path.exists():
        try:
            existing_cfg = _read_run_config_file(cfg_path)
            config_state = "ok"
        except Exception:
            config_state = "invalid"

    dirty = _is_run_dirty(base_dir) if exists else False

    match_keys = ["kind", "prefix", "year", "onet_version", "encoder_name", "encoder_fragment", "note"]
    diff = _cfg_diff(existing_cfg, expected_cfg, match_keys) if config_state == "ok" else {}
    match_state: RunMatchState = (
        "match" if (config_state == "ok" and not diff)
        else ("unknown" if config_state != "ok" else "mismatch")
    )

    status = RunStatus(
        requested_tag=base_tag,
        resolved_tag=base_tag,
        run_dir=base_dir,
        exists=exists,
        config_state=config_state,
        match_state=match_state,
        dirty=dirty,
        expected_cfg=expected_cfg,
        existing_cfg=existing_cfg,
        diff=diff,
        action="reuse",
        note="",
    )

    def _return(rp: RunPaths, cfg: dict[str, Any], spec_):
        if return_status:
            return rp, cfg, spec_, status
        return rp, cfg, spec_

    # 1) base saknas -> skapa base
    if not exists:
        rp = set_run_tag(base_tag, echo=True, persist=persist, ensure_dirs=ensure_dirs)
        write_run_config(expected_cfg, overwrite=True)
        status.action = "create_new"
        status.note = "created base run"
        cfg = read_run_config(default=expected_cfg)
        return _return(rp, cfg, spec)

    # 2) base finns och matchar -> reuse
    if match_state == "match":
        rp = set_run_tag(base_tag, echo=False, persist=persist, ensure_dirs=ensure_dirs)
        status.action = "reuse"
        status.note = "reused matching base run"
        cfg = read_run_config(default=expected_cfg)
        return _return(rp, cfg, spec)

    # 3) base finns men config saknas/invalid -> error
    if config_state in {"missing", "invalid"}:
        status.action = "error"
        raise RunInitError(f"Run exists but run_config is {config_state}: {cfg_path}", status)

    # 4) mismatch -> create new
    if not append_timestamp_on_new:
        # konservativt: mismatch utan timestamp är nästan alltid en kollision
        append_timestamp_on_new = True

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_tag = f"{base_tag}__{ts}"

    rp = set_run_tag(new_tag, echo=True, persist=persist, ensure_dirs=ensure_dirs)

    cfg_new = _expected_embeddings_run_config(
        run_tag=new_tag,
        prefix=prefix,
        year=year,
        onet_version=str(onet_version),
        encoder_name=str(encoder_name),
        encoder_fragment=spec.run_tag_fragment(),
        note=note2,
    )
    write_run_config(cfg_new, overwrite=True)

    status.action = "create_new"
    status.resolved_tag = new_tag
    status.run_dir = RUNS_ROOT / new_tag
    status.note = "created new run due to config mismatch"
    warnings.warn(f"Existing run '{base_tag}' did not match; created new run '{new_tag}'.", RuntimeWarning)

    cfg = read_run_config(default=cfg_new)
    return _return(rp, cfg, spec)


def load_embeddings_run(*, strict: bool = True, return_status: bool = False):
    """
    Load active run (RUN_TAG/RP) and resolve encoder from run_config.

    strict=True:
      - kräver att run_config finns, är läsbar och har encoder_name
      - validerar kind=="embeddings_run"
    """
    import encoders as _enc

    tag, run_dir, rp = _require_active_run()

    cfg_path = _run_config_path(run_dir)
    exists = cfg_path.exists()
    dirty = _is_run_dirty(run_dir)

    existing: dict[str, Any] = {}
    config_state: RunConfigState = "missing"

    if exists:
        try:
            existing = _read_run_config_file(cfg_path)
            config_state = "ok"
        except Exception:
            config_state = "invalid"

    status = RunStatus(
        requested_tag=tag,
        resolved_tag=tag,
        run_dir=run_dir,
        exists=True,
        config_state=config_state,
        match_state="unknown",
        dirty=dirty,
        expected_cfg={},
        existing_cfg=existing,
        diff={},
        action="reuse",
        note="loaded active run",
    )

    if strict and config_state != "ok":
        status.action = "error"
        raise RunInitError(f"Active run_config is {config_state}: {cfg_path}", status)

    cfg = existing if config_state == "ok" else read_run_config(default={})

    if strict:
        if cfg.get("kind") != "embeddings_run":
            raise RuntimeError(f"run_config kind mismatch: {cfg.get('kind')}")
        enc_name = str(cfg.get("encoder_name", "")).strip()
        if not enc_name:
            raise RuntimeError("run_config.json saknar encoder_name")
        spec = _enc.get_encoder(enc_name)
    else:
        enc_name = str(cfg.get("encoder_name", "")).strip()
        spec = _enc.get_encoder(enc_name) if enc_name else None

    if return_status:
        return rp, cfg, spec, status
    return rp, cfg, spec


# Add to infra.py, after load_embeddings_run()

def find_embeddings_run(
    encoder_name: str,
    *,
    year: int | None = None,
    onet_version: str | None = None,
    prefix: str = "embeddings",
    strict: bool = True,
) -> RunPaths | None:
    """
    Find a single embeddings run matching encoder_name (and optionally year/onet_version).
    
    Reads run_config.json of each candidate; does not activate the run.
    Returns RunPaths pointing into the matching run, or None if not found
    (or raises if strict=True and not found).
    
    If multiple runs match, returns the most recent (by mtime).
    """
    v_norm = str(onet_version).replace(".", "_") if onet_version is not None else None
    candidates = list_runs(prefix=prefix)
    
    matches: list[tuple[float, RunPaths]] = []
    for info in candidates:
        cfg_fp = _run_config_path(info.path)
        if not cfg_fp.exists():
            continue
        try:
            cfg = _read_run_config_file(cfg_fp)
        except Exception:
            continue
        if cfg.get("kind") != "embeddings_run":
            continue
        if str(cfg.get("encoder_name", "")).strip() != str(encoder_name).strip():
            continue
        if year is not None and int(cfg.get("year", -1)) != int(year):
            continue
        if v_norm is not None and str(cfg.get("onet_version", "")).replace(".", "_") != v_norm:
            continue
        rp = RunPaths(run_tag=info.tag)
        matches.append((info.mtime, rp))
    
    if not matches:
        if strict:
            raise FileNotFoundError(
                f"No embeddings run found for encoder_name={encoder_name!r}"
                + (f", year={year}" if year is not None else "")
                + (f", onet_version={onet_version!r}" if onet_version is not None else "")
            )
        return None
    
    matches.sort(key=lambda t: t[0], reverse=True)
    return matches[0][1]


def iter_embeddings_runs(
    *,
    year: int | None = None,
    onet_version: str | None = None,
    prefix: str = "embeddings",
) -> list[tuple[str, RunPaths, dict[str, Any]]]:
    """
    Yield (encoder_name, RunPaths, run_config) for every embeddings run on disk
    that matches the given filters. Most recent first when multiple runs share
    an encoder_name.
    """
    v_norm = str(onet_version).replace(".", "_") if onet_version is not None else None
    out: list[tuple[float, str, RunPaths, dict[str, Any]]] = []
    for info in list_runs(prefix=prefix):
        cfg_fp = _run_config_path(info.path)
        if not cfg_fp.exists():
            continue
        try:
            cfg = _read_run_config_file(cfg_fp)
        except Exception:
            continue
        if cfg.get("kind") != "embeddings_run":
            continue
        if year is not None and int(cfg.get("year", -1)) != int(year):
            continue
        if v_norm is not None and str(cfg.get("onet_version", "")).replace(".", "_") != v_norm:
            continue
        enc_name = str(cfg.get("encoder_name", "")).strip()
        if not enc_name:
            continue
        out.append((info.mtime, enc_name, RunPaths(run_tag=info.tag), cfg))
    
    out.sort(key=lambda t: t[0], reverse=True)
    return [(name, rp, cfg) for _, name, rp, cfg in out]
    

# ─────────────────────────────────────────────────────────────
# Generic IO helpers
# ─────────────────────────────────────────────────────────────

def read_json(fp: Union[str, Path]) -> Any:
    return json.loads(Path(fp).read_text(encoding="utf-8"))


def write_json(fp: Union[str, Path], obj: Any) -> None:
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_pkl(fp: Union[str, Path]) -> Any:
    with open(fp, "rb") as f:
        return pickle.load(f)


def write_pkl(fp: Union[str, Path], obj: Any) -> None:
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_csv_any(fp: Union[str, Path], **kwargs) -> pd.DataFrame:
    return pd.read_csv(fp, **kwargs)


def write_csv(fp: Union[str, Path], df: pd.DataFrame, *, index: bool = False, **kwargs) -> None:
    p = Path(fp)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=index, **kwargs)


def sha256_file(fp: Union[str, Path], *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


__all__ = [
    "PROJECT_ROOT",
    "OUT_ROOT",
    "RUNS_ROOT",
    "GLOBAL_CACHE_ROOT",
    "RUN_TAG",
    "RUN_DIR",
    "RP",
    "RunPaths",
    "RunInfo",
    "RunStatus",
    "RunInitError",
    "activate_last_run",
    "set_run_tag",
    "make_run_tag",
    "list_runs",
    "run_config_fp",
    "read_run_config",
    "write_run_config",
    "init_embeddings_run",
    "load_embeddings_run",
    "find_embeddings_run",
    "iter_embeddings_runs",
    "read_json",
    "write_json",
    "read_pkl",
    "write_pkl",
    "read_csv_any",
    "write_csv",
    "sha256_file",
    "log",
]

