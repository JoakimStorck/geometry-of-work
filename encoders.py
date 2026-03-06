# encoders.py
from __future__ import annotations

from typing import Dict, List
import importlib

from encoder_definitions import EncoderSpec

_REG: Dict[str, EncoderSpec] = {}
_ALIASES: Dict[str, str] = {}
_LOADED = False


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return

    defs = importlib.import_module("encoder_definitions")
    enc = getattr(defs, "ENCODERS")
    als = getattr(defs, "ALIASES")

    _REG.clear()
    _REG.update({str(k).strip(): v for k, v in enc.items()})

    _ALIASES.clear()
    _ALIASES.update({str(k).strip(): str(v).strip() for k, v in als.items()})

    _LOADED = True


def list_encoders() -> List[str]:
    _ensure_loaded()
    return sorted(_REG.keys())


def resolve_encoder_name(name: str) -> str:
    _ensure_loaded()
    key = str(name or "").strip()
    return _ALIASES.get(key, key)


def get_encoder(name: str) -> EncoderSpec:
    _ensure_loaded()
    key = resolve_encoder_name(name)
    if key not in _REG:
        raise KeyError(f"Unknown encoder: {name}. Available: {', '.join(list_encoders())}")
    return _REG[key]
