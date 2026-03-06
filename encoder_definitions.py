# encoder_definitions.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional
import re

EncoderType = Literal["openai", "sentence-transformers", "mistral"]

_TAG_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_WS_RE = re.compile(r"\s+")


def _slugify(s: str) -> str:
    s = (s or "").strip()
    s = _WS_RE.sub("_", s)
    s = _TAG_SAFE_RE.sub("_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


@dataclass(frozen=True)
class EncoderSpec:
    encoder_type: EncoderType

    # Common
    dimensions: Optional[int] = None
    normalize: bool = True
    prefix: str = "onet_task"
    extra_id: Optional[str] = None

    # OpenAI
    openai_model: Optional[str] = None
    openai_base_url: Optional[str] = None
    openai_api_key_env: str = "OPENAI_API_KEY"

    # SentenceTransformers
    st_model: Optional[str] = None
    st_device: Optional[str] = None
    st_trust_remote_code: bool = False

    # Mistral
    mistral_model: Optional[str] = None
    mistral_api_key_env: str = "MISTRAL_API_KEY"

    # Retry knobs (applies to network providers)
    retry_retries: int = 3
    retry_backoff: float = 1.5
    retry_initial_delay: float = 1.0
    retry_jitter: bool = True

    def embedder_id(self) -> str:
        if self.encoder_type == "openai":
            if not self.openai_model:
                raise ValueError("openai_model saknas för encoder_type='openai'")
            dim = f"?dim={int(self.dimensions)}" if self.dimensions else ""
            base = f"openai:{self.openai_model}{dim}"

        elif self.encoder_type == "sentence-transformers":
            if not self.st_model:
                raise ValueError("st_model saknas för encoder_type='sentence-transformers'")
            base = f"st:{self.st_model}"

        elif self.encoder_type == "mistral":
            if not self.mistral_model:
                raise ValueError("mistral_model saknas för encoder_type='mistral'")
            base = f"mistral:{self.mistral_model}"

        else:
            raise ValueError(f"Okänd encoder_type: {self.encoder_type}")

        if self.extra_id:
            base = f"{base}#{self.extra_id}"
        return base

    def embedder_slug(self) -> str:
        return _slugify(self.embedder_id())

    def run_tag_fragment(self) -> str:
        if self.encoder_type == "openai":
            model = self.openai_model or "unknown-model"
            dim = f"d{int(self.dimensions)}" if self.dimensions else "d?"
            return "__".join(["openai", _slugify(model), dim])

        if self.encoder_type == "sentence-transformers":
            model = self.st_model or "unknown-model"
            dim = f"d{int(self.dimensions)}" if self.dimensions else "d?"
            return "__".join(["st", _slugify(model), dim])

        if self.encoder_type == "mistral":
            model = self.mistral_model or "unknown-model"
            dim = f"d{int(self.dimensions)}" if self.dimensions else "d?"
            return "__".join(["mistral", _slugify(model), dim])

        return _slugify(self.embedder_id())


# Canonical encoders
ENCODERS: Dict[str, EncoderSpec] = {
    # OpenAI – text-embedding-3-large (3072d)
    # Keep "openai-3-large" as canonical to match existing notebooks.
    "openai-3-large": EncoderSpec(
        encoder_type="openai",
        openai_model="text-embedding-3-large",
        dimensions=3072,
        normalize=True,
        prefix="onet_task",
    ),

    # SentenceTransformers
    "st-gtr-t5-large": EncoderSpec(
        encoder_type="sentence-transformers",
        st_model="sentence-transformers/gtr-t5-large",
        normalize=True,
        prefix="onet_task",
    ),

    "st-minilm-l6-v2": EncoderSpec(
        encoder_type="sentence-transformers",
        st_model="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        st_device="cpu",
        normalize=True,
        prefix="onet_task",
    ),

    # Mistral
    "mistral-embed-norm": EncoderSpec(
        encoder_type="mistral",
        mistral_model="mistral-embed",
        normalize=True,
        prefix="onet_task",
    ),
}

# Aliases (back-compat / convenience)
ALIASES: Dict[str, str] = {
    # convenience alias you may prefer in notebooks/scripts
    "openai-3-large-3072": "openai-3-large",

    # older default name used in some notebooks/widgets
    # NOTE: alias points to the same spec (still normalize=True)
    "openai-3-large-3072-nonorm": "openai-3-large",
}
