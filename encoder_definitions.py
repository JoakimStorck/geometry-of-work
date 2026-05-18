# encoder_definitions.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple
import re

EncoderType = Literal["openai", "sentence-transformers", "mistral", "tfidf"]

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
    # Instruction-tuned ST models (Qwen3, e5, instructor) need a prefix on input.
    # Applied as: f"{st_instruction_prefix}{text}" before encoding.
    # Empty string = no prefix (back-compat with existing ST encoders).
    st_instruction_prefix: str = ""
    st_model_kwargs: Optional[Dict[str, Any]] = None  # NEW: e.g. {"torch_dtype": "bfloat16"}    
    
    # Mistral
    mistral_model: Optional[str] = None
    mistral_api_key_env: str = "MISTRAL_API_KEY"

    # TF-IDF + TruncatedSVD baseline.
    # Note: tfidf is a corpus-fitted model, not a per-text encoder. It bypasses
    # the per-text cache. dimensions field is used for SVD output size.
    tfidf_max_features: Optional[int] = 50000
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.95
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    tfidf_sublinear_tf: bool = True
    tfidf_svd_random_state: int = 42

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
            if self.st_instruction_prefix:
                # Include a short prefix hash so two specs with different prefixes
                # don't collide on the same st_model.
                import hashlib
                ph = hashlib.sha256(self.st_instruction_prefix.encode("utf-8")).hexdigest()[:6]
                base = f"{base}+prefix={ph}"

        elif self.encoder_type == "mistral":
            if not self.mistral_model:
                raise ValueError("mistral_model saknas för encoder_type='mistral'")
            base = f"mistral:{self.mistral_model}"

        elif self.encoder_type == "tfidf":
            dim = f"?svd={int(self.dimensions)}" if self.dimensions else "?svd=?"
            ng = f"ng={self.tfidf_ngram_range[0]}-{self.tfidf_ngram_range[1]}"
            mf = f"mf={self.tfidf_max_features}" if self.tfidf_max_features else "mf=all"
            base = f"tfidf:{ng};{mf}{dim}"

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

        if self.encoder_type == "tfidf":
            dim = f"d{int(self.dimensions)}" if self.dimensions else "d?"
            ng = f"ng{self.tfidf_ngram_range[0]}-{self.tfidf_ngram_range[1]}"
            return "__".join(["tfidf", ng, dim])

        return _slugify(self.embedder_id())


# Canonical encoders
ENCODERS: Dict[str, EncoderSpec] = {
    # OpenAI – text-embedding-3-large (3072d)
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

    # Qwen3-Embedding-4B (open SOTA, instruction-tuned).
    # Requires a task-prefix on input to perform optimally.
    "qwen3-embedding-4b": EncoderSpec(
        encoder_type="sentence-transformers",
        st_model="Qwen/Qwen3-Embedding-4B",
        st_device="cuda",
        st_trust_remote_code=True,
        st_instruction_prefix=(
            "Instruct: Represent this occupational task statement "
            "for similarity comparison with other task statements.\nQuery: "
        ),
        st_model_kwargs={"torch_dtype": "bfloat16"},
        dimensions=2560,
        normalize=True,
        prefix="onet_task",
    ),
    
    # BGE-M3 (open, broadly adopted retrieval encoder).
    "bge-m3": EncoderSpec(
        encoder_type="sentence-transformers",
        st_model="BAAI/bge-m3",
        st_device="cuda",
        dimensions=1024,
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

    # TF-IDF + TruncatedSVD baseline. Dense 768-dim output, fed to the same
    # PCA pipeline as the transformer encoders.
    "tfidf-svd-768": EncoderSpec(
        encoder_type="tfidf",
        dimensions=768,
        normalize=True,
        prefix="onet_task",
        tfidf_max_features=50000,
        tfidf_min_df=2,
        tfidf_max_df=0.95,
        tfidf_ngram_range=(1, 2),
        tfidf_sublinear_tf=True,
    ),
}

# Aliases (back-compat / convenience)
ALIASES: Dict[str, str] = {
    "openai-3-large-3072": "openai-3-large",
    "openai-3-large-3072-nonorm": "openai-3-large",
}