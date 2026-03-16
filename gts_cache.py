"""
gts_cache.py
------------
Tunn wrapper runt cache.py for att cacha GTS-scenarioresultat (dicts).

Befintlig cache.NpyJsonStore hanterar numpy-arrays. Scenarioresultat ar
dicts, sa vi anvander pickle men ateranvander cache.stable_key() och
cache.atomic_write_json() for nyckling och metadata.

Cachenyckeln ar SHA256 av:
  - Scenario-parametrar   (JSON, sorterat)
  - Kallkod: gts_core.py  (inspect.getsource)
  - Kallkod: gts_plot.py  (inspect.getsource)

Andringar i parametrar ELLER kod ger cache miss -> kors om automatiskt.

Filformat per entry:
  <cache_dir>/<k[:2]>/<k[2:4]>/<key>/result.pkl
  <cache_dir>/<k[:2]>/<k[2:4]>/<key>/meta.json

Anvandning
----------
    import gts_cache
    gts_cache.init(RP.exports / "scenario_cache")

    res1 = gts_cache.run(CASE1, gts.run_scenario)
    res2 = gts_cache.run(CASE2, gts.run_scenario, force=True)

    gts_cache.status()
    gts_cache.clear()
    gts_cache.clear("Case 1: Automation (gamma=0)")
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cache  # projektets befintliga cache-modul

# ---------------------------------------------------------------------------
_cache_dir: Path = Path("scenario_cache")


def init(path) -> None:
    """Satt cache-katalog. Anropa en gang fran notebooken."""
    global _cache_dir
    _cache_dir = Path(path)
    _cache_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------

def _source_hash(*module_names: str) -> str:
    parts = []
    for name in module_names:
        mod = importlib.import_module(name)
        try:
            parts.append(inspect.getsource(mod))
        except (OSError, TypeError):
            parts.append(f"<no source: {name}>")
    return cache.stable_key("".join(parts).encode("utf-8"))


def _make_key(scenario: dict) -> str:
    params = {k: v for k, v in scenario.items()
              if k not in ("plot", "save_path")}
    param_hash = cache.stable_key(params)
    code_hash  = _source_hash("gts_core", "gts_plot")
    return cache.stable_key({"p": param_hash, "c": code_hash})


def _entry_paths(key: str):
    d = _cache_dir / key[:2] / key[2:4] / key
    return d / "result.pkl", d / "meta.json"


def _load(key: str) -> Any | None:
    pkl_path, meta_path = _entry_paths(key)
    if not pkl_path.exists() or not meta_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _save(key: str, result: Any, scenario: dict) -> None:
    pkl_path, meta_path = _entry_paths(key)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=pkl_path.name + ".", dir=str(pkl_path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, pkl_path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass
    cache.atomic_write_json(meta_path, {
        "name":       scenario.get("name", ""),
        "scenario":   {k: v for k, v in scenario.items()
                       if k not in ("plot", "save_path")},
        "cached_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "cache_key":  key,
    })


# ---------------------------------------------------------------------------

def run(scenario: dict, fn: Callable, *, force: bool = False) -> Any:
    """
    Returnera cachat resultat om det finns och koden ar oforandrad.
    Annars kor fn(scenario), spara och returnera.
    """
    key  = _make_key(scenario)
    name = scenario.get("name", key[:8])
    if not force:
        result = _load(key)
        if result is not None:
            print(f"  [cache hit]  {name}")
            return result
    print(f"  [running]    {name}")
    result = fn(scenario)
    _save(key, result, scenario)
    return result


def status() -> None:
    """Skriv ut vad som finns i cachen."""
    if not _cache_dir.exists():
        print("Cache ar tom (katalogen finns inte).")
        return
    entries = sorted(_cache_dir.rglob("meta.json"))
    if not entries:
        print("Cache ar tom.")
        return
    print(f"Cache: {_cache_dir}  ({len(entries)} poster)\n")
    print(f"  {'Namn':<50}  {'Cachat (UTC)':>22}  {'Nyckel':>12}")
    for meta_path in entries:
        try:
            meta = json.loads(meta_path.read_text())
            print(f"  {meta.get('name','?'):<50}  "
                  f"{meta.get('cached_utc','?'):>22}  "
                  f"{meta.get('cache_key','?')[:12]:>12}")
        except Exception:
            print(f"  <olesbar meta: {meta_path}>")


def clear(name: str | None = None) -> None:
    """
    Rensa cachen.
    clear()       -> radera allt
    clear("namn") -> radera bara poster vars name matchar exakt
    """
    import shutil
    if not _cache_dir.exists():
        print("Inget att rensa.")
        return
    if name is None:
        shutil.rmtree(_cache_dir, ignore_errors=True)
        _cache_dir.mkdir(parents=True, exist_ok=True)
        print("Cache rensad.")
        return
    removed = 0
    for meta_path in _cache_dir.rglob("meta.json"):
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("name") == name:
                shutil.rmtree(meta_path.parent, ignore_errors=True)
                removed += 1
        except Exception:
            pass
    print(f"Raderade {removed} post(er) med namn '{name}'.")