# cache.py
"""
cache.py

Generisk, content-addressed cache.

- Inga domänbegrepp (ingen "embedding").
- Stabil nyckling via deterministisk JSON.
- Filformat: data.npy + meta.json
- Atomiska writes.
- Skalad kataloglayout via sharding (default).

Layout:
  <root>/<namespace>/<k[:2]>/<k[2:4]>/<key>/data.npy
  <root>/<namespace>/<k[:2]>/<k[2:4]>/<key>/meta.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

import numpy as np


Jsonable = Union[None, bool, int, float, str, list, dict]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonical_json(obj: Any) -> str:
    """Deterministisk JSON-serialisering (sort_keys, fasta separators) för hashing."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def stable_key(payload: Union[str, bytes, Mapping[str, Any], list, tuple, Any], *, algo: str = "sha256") -> str:
    """
    Stabil hex-hash för payload.

    Tillåter:
    - bytes
    - str (UTF-8)
    - dict/list/tuple (JSON)
    - dataclass (asdict -> JSON)

    Övriga typer -> TypeError (ingen str(payload)-fallback).
    """
    if isinstance(payload, bytes):
        b = payload
    elif isinstance(payload, str):
        b = payload.encode("utf-8")
    elif isinstance(payload, (dict, list, tuple)):
        b = _canonical_json(payload).encode("utf-8")
    elif is_dataclass(payload):
        b = _canonical_json(asdict(payload)).encode("utf-8")
    else:
        raise TypeError(f"Unsupported payload type for stable_key: {type(payload)}")

    h = hashlib.new(algo)
    h.update(b)
    return h.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _atomic_write_json(path: Path, obj: Jsonable) -> None:
    data = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write_bytes(path, data)


def _atomic_write_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, np.asarray(arr), allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


# Public wrappers (so other modules don't need to import private helpers)
def atomic_write_json(path: Union[str, Path], obj: Jsonable) -> None:
    _atomic_write_json(Path(path), obj)


def atomic_write_npy(path: Union[str, Path], arr: np.ndarray) -> None:
    _atomic_write_npy(Path(path), arr)


@dataclass(frozen=True)
class CacheHit:
    value: np.ndarray
    meta: dict


class NpyJsonStore:
    """
    Filbaserad store:
      data.npy + meta.json

    Sharding:
      shard_levels=(2,2) innebär:
        <root>/<namespace>/<k[:2]>/<k[2:4]>/<key>/...
    """

    def __init__(
        self,
        *,
        root: Path,
        namespace: str,
        schema_version: str = "1",
        shard_levels: Tuple[int, int] = (2, 2),
    ) -> None:
        self.root = Path(root)
        self.namespace = namespace.strip("/")

        if not self.namespace:
            raise ValueError("namespace must be non-empty")

        self.schema_version = str(schema_version)
        self.shard_levels = shard_levels

    def _entry_dir(self, key: str) -> Path:
        a, b = self.shard_levels
        if a <= 0:
            return self.root / self.namespace / key
        if b <= 0:
            return self.root / self.namespace / key[:a] / key
        return self.root / self.namespace / key[:a] / key[a : a + b] / key

    def path_for(self, key: str) -> Tuple[Path, Path]:
        d = self._entry_dir(key)
        return d / "data.npy", d / "meta.json"

    def exists(self, key: str) -> bool:
        npy_fp, meta_fp = self.path_for(key)
        return npy_fp.exists() and meta_fp.exists()

    def get(
        self,
        key: str,
        *,
        strict: bool = False,
        log: bool = True,
        mmap_mode: Optional[str] = None,
    ) -> Optional[CacheHit]:
        npy_fp, meta_fp = self.path_for(key)
        if not (npy_fp.exists() and meta_fp.exists()):
            return None

        try:
            meta = json.loads(meta_fp.read_text(encoding="utf-8"))
            arr = np.load(npy_fp, allow_pickle=False, mmap_mode=mmap_mode)

            # Sanity-check mot meta om möjligt
            if isinstance(meta, dict):
                shp = meta.get("shape")
                dt = meta.get("dtype")
                if shp is not None and list(arr.shape) != list(shp):
                    raise ValueError(f"shape mismatch: file={arr.shape}, meta={shp}")
                if dt is not None and str(arr.dtype) != str(dt):
                    raise ValueError(f"dtype mismatch: file={arr.dtype}, meta={dt}")

            return CacheHit(value=arr, meta=meta)

        except (json.JSONDecodeError, ValueError, OSError) as e:
            # Förväntade fel (korrupt post etc.) kan behandlas som cache-miss i non-strict.
            if log:
                logging.warning("Cache get failed for key=%s (%s): %s", key, type(e).__name__, e)
            if strict:
                raise
            return None

        except Exception as e:
            if log:
                logging.warning("Cache get unexpected error for key=%s (%s): %s", key, type(e).__name__, e)
            if strict:
                raise
            return None

    def put(self, key: str, value: np.ndarray, meta: Optional[dict] = None) -> None:
        value = np.asarray(value)
        npy_fp, meta_fp = self.path_for(key)

        base_meta: dict = {
            "schema_version": self.schema_version,
            "created_utc": utc_now_iso(),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "order": "F" if value.flags["F_CONTIGUOUS"] and not value.flags["C_CONTIGUOUS"] else "C",
        }
        if meta:
            base_meta.update(meta)

        _atomic_write_npy(npy_fp, value)
        _atomic_write_json(meta_fp, base_meta)

    def clear_key(self, key: str) -> None:
        """
        Best-effort borttagning av en entry. Vi städar endast:
        - entry-katalogen (…/<key>/) om den blir tom

        (Vi försöker inte rensa shard-parent-kataloger; det minskar race-risk.)
        """
        npy_fp, meta_fp = self.path_for(key)
        entry_dir = npy_fp.parent

        for fp in (npy_fp, meta_fp):
            try:
                if fp.exists():
                    fp.unlink()
            except Exception:
                pass

        # Rensa endast entry_dir om tom
        try:
            if entry_dir.exists() and not any(entry_dir.iterdir()):
                entry_dir.rmdir()
        except Exception:
            pass

    def clear_namespace(self) -> None:
        ns_dir = self.root / self.namespace
        if not ns_dir.exists():
            return
        import shutil
        shutil.rmtree(ns_dir, ignore_errors=True)
