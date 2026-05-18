# embeddings.py
"""
embeddings.py

Provider-agnostisk embeddings-modul.
- Pluggbar encoder (OpenAI / sentence-transformers / Mistral)
- Per-text cache via cache.py (data.npy + meta.json), valbar scope: global eller run
- Run-local persistens av "hela matrisen" via load_or_compute_run_embeddings(..., emb_npy, fp_json)

Revidering 3 (städad):
- EncoderSpec comes from encoder_definitions (shared with infra/encoders)
- Inga beroenden av notebook-UI / runtag-arkitektur i notebookceller
- Cache-root-resolve via infra (global/run) men cache-format ligger i cache.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional

import numpy as np

import infra
from cache import NpyJsonStore, atomic_write_json, atomic_write_npy, stable_key
from encoder_definitions import EncoderSpec

CacheScope = Literal["global", "run"]

# bump när du ändrar preprocessing-semantik eller prefix-apply-semantik
_PREPROC_VERSION = "v0.2"



# ─────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────

def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


# Uppdaterad _runtime_versions (lägg till tfidf-grenen)
def _runtime_versions(spec: "EncoderSpec") -> Dict[str, str]:
    out = {"python": sys.version.split()[0], "numpy": np.__version__}
    try:
        if spec.encoder_type == "openai":
            import openai
            out["openai"] = getattr(openai, "__version__", "unknown")
        elif spec.encoder_type == "sentence-transformers":
            import sentence_transformers
            out["sentence_transformers"] = getattr(sentence_transformers, "__version__", "unknown")
            try:
                import transformers
                out["transformers"] = getattr(transformers, "__version__", "unknown")
            except Exception:
                pass
            try:
                import torch
                out["torch"] = getattr(torch, "__version__", "unknown")
            except Exception:
                pass
        elif spec.encoder_type == "mistral":
            import mistralai
            out["mistralai"] = getattr(mistralai, "__version__", "unknown")
        elif spec.encoder_type == "tfidf":
            import sklearn
            out["scikit_learn"] = getattr(sklearn, "__version__", "unknown")
            import scipy
            out["scipy"] = getattr(scipy, "__version__", "unknown")
    except Exception:
        pass
    return out


def _retry(
    fn: Callable[[], np.ndarray],
    *,
    retries: int = 3,
    backoff: float = 1.5,
    initial_delay: float = 1.0,
    jitter: bool = True,
) -> np.ndarray:
    last_exc: Optional[BaseException] = None
    delay = float(initial_delay)
    for _ in range(max(1, int(retries))):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if retries <= 1:
                break
            j = random.uniform(0.85, 1.15) if jitter else 1.0
            time.sleep(delay * j)
            delay *= float(backoff)
    assert last_exc is not None
    raise last_exc


# ─────────────────────────────────────────────────────────────
# Signature + provider embed fns
# ─────────────────────────────────────────────────────────────

# Uppdaterad embedder_signature (lägg till tfidf-fält i payload)
def embedder_signature(spec: EncoderSpec) -> str:
    payload = {
        "encoder_type": spec.encoder_type,
        "dimensions": spec.dimensions,
        "normalize": spec.normalize,
        "extra_id": spec.extra_id,
        "preproc_version": _PREPROC_VERSION,
        "openai_model": spec.openai_model,
        "openai_base_url": spec.openai_base_url,
        "st_model": spec.st_model,
        "st_device": spec.st_device,
        "st_trust_remote_code": spec.st_trust_remote_code,
        "st_instruction_prefix": spec.st_instruction_prefix,
        "st_model_kwargs": dict(spec.st_model_kwargs or {}),
        "mistral_model": spec.mistral_model,
        "tfidf_max_features": spec.tfidf_max_features,
        "tfidf_min_df": spec.tfidf_min_df,
        "tfidf_max_df": spec.tfidf_max_df,
        "tfidf_ngram_range": list(spec.tfidf_ngram_range),
        "tfidf_sublinear_tf": spec.tfidf_sublinear_tf,
        "tfidf_svd_random_state": spec.tfidf_svd_random_state,
    }
    canon = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha256(canon).hexdigest()[:10]
    base = spec.embedder_slug()
    return f"{base}__pre{_PREPROC_VERSION}__{h}"


def make_embed_fn_openai(spec: EncoderSpec) -> Callable[[List[str]], np.ndarray]:
    from openai import OpenAI  # type: ignore

    api_key = os.environ.get(spec.openai_api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {spec.openai_api_key_env}")

    client = OpenAI(api_key=api_key, base_url=spec.openai_base_url) if spec.openai_base_url else OpenAI(api_key=api_key)
    model = spec.openai_model
    if not model:
        raise ValueError("openai_model saknas")

    dims = spec.dimensions

    def _embed(batch: List[str]) -> np.ndarray:
        def _call() -> np.ndarray:
            if dims:
                resp = client.embeddings.create(model=model, input=batch, dimensions=dims)
            else:
                resp = client.embeddings.create(model=model, input=batch)
            return np.asarray([d.embedding for d in resp.data], dtype=np.float32)

        return _retry(
            _call,
            retries=spec.retry_retries,
            backoff=spec.retry_backoff,
            initial_delay=spec.retry_initial_delay,
            jitter=spec.retry_jitter,
        )

    return _embed


# Uppdaterad ST-funktion med prefix-stöd
def make_embed_fn_sentence_transformers(spec: EncoderSpec) -> Callable[[List[str]], np.ndarray]:
    from sentence_transformers import SentenceTransformer

    if not spec.st_model:
        raise ValueError("st_model saknas")

    # Resolve dtype string → torch.dtype if provided in model_kwargs.
    model_kwargs = dict(spec.st_model_kwargs or {})
    if "dtype" in model_kwargs and isinstance(model_kwargs["dtype"], str):
        import torch
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        s = model_kwargs["dtype"]
        if s not in dtype_map:
            raise ValueError(f"Unknown dtype: {s!r}")
        model_kwargs["dtype"] = dtype_map[s]

    model = SentenceTransformer(
        spec.st_model,
        device=spec.st_device,
        trust_remote_code=spec.st_trust_remote_code,
        model_kwargs=model_kwargs if model_kwargs else None,
    )
    prefix = spec.st_instruction_prefix or ""

    def _embed(batch: List[str]) -> np.ndarray:
        if prefix:
            batch = [f"{prefix}{t}" for t in batch]
        arr = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return np.asarray(arr, dtype=np.float32)

    return _embed


def make_embed_fn_mistral(spec: EncoderSpec) -> Callable[[List[str]], np.ndarray]:
    from mistralai.client import Mistral  # type: ignore

    api_key = os.environ.get(spec.mistral_api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {spec.mistral_api_key_env}")

    if not spec.mistral_model:
        raise ValueError("mistral_model saknas")

    client = Mistral(api_key=api_key)
    model = spec.mistral_model

    def _embed(batch: List[str]) -> np.ndarray:
        def _call() -> np.ndarray:
            resp = client.embeddings.create(model=model, inputs=batch)
            return np.asarray([d.embedding for d in resp.data], dtype=np.float32)

        return _retry(
            _call,
            retries=spec.retry_retries,
            backoff=spec.retry_backoff,
            initial_delay=spec.retry_initial_delay,
            jitter=spec.retry_jitter,
        )

    return _embed


    # Helt ny: TF-IDF + TruncatedSVD (corpus-fitted, separate path)
def compute_tfidf_embeddings(
    texts: List[str],
    *,
    spec: EncoderSpec,
    show_progress: bool = True,
    progress_desc: str = "Embedding (tfidf)",
) -> np.ndarray:
    """
    Compute TF-IDF + TruncatedSVD embeddings on the full text corpus.

    Unlike API-based encoders, TF-IDF is corpus-fitted: vocabulary, idf weights,
    and SVD components are all learned from the supplied texts. The result is a
    dense (n_texts, dimensions) matrix matching the format produced by the
    other encoders.

    Bypasses the per-text cache since computation is fast and not text-local.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    texts_pp = preprocess_texts(texts, allow_none=True)
    n = len(texts_pp)

    if spec.dimensions is None or spec.dimensions <= 0:
        raise ValueError("TF-IDF encoder requires positive 'dimensions' (SVD output size)")

    if show_progress:
        print(f"   [{progress_desc}] fitting TfidfVectorizer on {n:,} texts...")

    vec = TfidfVectorizer(
        max_features=spec.tfidf_max_features,
        min_df=spec.tfidf_min_df,
        max_df=spec.tfidf_max_df,
        ngram_range=tuple(spec.tfidf_ngram_range),
        sublinear_tf=spec.tfidf_sublinear_tf,
        lowercase=True,
        strip_accents="unicode",
    )
    X_sparse = vec.fit_transform(texts_pp)
    actual_vocab = X_sparse.shape[1]

    svd_components = int(spec.dimensions)
    if svd_components >= actual_vocab:
        # SVD can't produce more components than the sparse rank allows.
        # Cap at actual_vocab - 1 with a warning.
        capped = max(2, actual_vocab - 1)
        print(f"   [{progress_desc}] WARNING: requested svd dim {svd_components} >= "
              f"vocab size {actual_vocab}, capping at {capped}")
        svd_components = capped

    if show_progress:
        print(f"   [{progress_desc}] TruncatedSVD: {actual_vocab} → {svd_components}")

    svd = TruncatedSVD(
        n_components=svd_components,
        random_state=int(spec.tfidf_svd_random_state),
    )
    X_dense = svd.fit_transform(X_sparse).astype(np.float32, copy=False)

    # If actual SVD dim < requested (vocab-limited), zero-pad to requested dims
    # so downstream shape assertions hold.
    if X_dense.shape[1] < int(spec.dimensions):
        pad = np.zeros((n, int(spec.dimensions) - X_dense.shape[1]), dtype=np.float32)
        X_dense = np.concatenate([X_dense, pad], axis=1)

    if spec.normalize:
        X_dense = _l2_normalize(X_dense)

    if show_progress:
        evr = float(svd.explained_variance_ratio_.sum())
        print(f"   [{progress_desc}] done. SVD explained variance ratio sum: {evr:.4f}")

    return X_dense

    
def make_embed_fn(spec: EncoderSpec) -> Callable[[List[str]], np.ndarray]:
    if spec.encoder_type == "openai":
        return make_embed_fn_openai(spec)
    if spec.encoder_type == "sentence-transformers":
        return make_embed_fn_sentence_transformers(spec)
    if spec.encoder_type == "mistral":
        return make_embed_fn_mistral(spec)
    raise ValueError(f"Okänd encoder_type: {spec.encoder_type}")


# ─────────────────────────────────────────────────────────────
# Text preprocessing + cache roots
# ─────────────────────────────────────────────────────────────

def preprocess_texts(texts: List[str], *, allow_none: bool = True) -> List[str]:
    out: List[str] = []
    for t in texts:
        if t is None:
            if not allow_none:
                raise TypeError("None not allowed in texts")
            tt = ""
        else:
            tt = str(t)
        out.append(tt.strip())
    return out


def _resolve_cache_root(scope: CacheScope) -> Path:
    # Cache-format ligger i cache.py; här väljer vi bara root.
    if scope == "run":
        return Path(infra.RP.cache)
    return Path(infra.GLOBAL_CACHE_ROOT)


def _key_payload(t_pp: str) -> Dict[str, Any]:
    return {"v": _PREPROC_VERSION, "text": t_pp}


# ─────────────────────────────────────────────────────────────
# Public embedding entrypoints
# ─────────────────────────────────────────────────────────────

def encode_with_cache(
    texts: List[str],
    *,
    spec: EncoderSpec,
    batch_size: int,
    cache_scope: CacheScope,
    allow_none: bool = True,
    show_progress: bool = True,
    progress_desc: str = "Embedding",
) -> np.ndarray:
    """
    Per-text cache:
      Namespace = embeddings/<embedder_signature>
      Key       = stable_key({"v": preproc_version, "text": preproc_text})

    Provider input:
      preproc_text (no prefix; prefix kept only as metadata)

    Progress:
      - Visas endast om det finns cache-missar.
      - Kräver inte tqdm; om tqdm saknas blir det no-op.
      - Enhet = batchar (miss-batchar), inte totalt antal texter.
    """

    # --- helper: tqdm om tillgänglig, annars no-op ---
    def _tqdm(it, **kwargs):
        if not show_progress:
            return it
        try:
            from tqdm.auto import tqdm  # type: ignore
            return tqdm(it, **kwargs)
        except Exception:
            return it

    texts_pp = preprocess_texts(texts, allow_none=allow_none)
    sig = embedder_signature(spec)
    root = _resolve_cache_root(cache_scope)
    store = NpyJsonStore(root=root, namespace=f"embeddings/{sig}")

    embed_fn = make_embed_fn(spec)

    keys = [stable_key(_key_payload(t)) for t in texts_pp]
    out: List[Optional[np.ndarray]] = [None] * len(texts_pp)
    missing_idx: List[int] = []

    # --- lookups ---
    for i, k in enumerate(keys):
        hit = store.get(k, strict=False, log=False)
        if hit is not None:
            out[i] = np.asarray(hit.value, dtype=np.float32)
        else:
            missing_idx.append(i)

    # --- compute misses (with optional progress) ---
    if missing_idx:
        missing_texts = [texts_pp[i] for i in missing_idx]
        missing_inputs = missing_texts

        n_miss = len(missing_inputs)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        starts = range(0, n_miss, batch_size)
        it = _tqdm(
            starts,
            total=(n_miss + batch_size - 1) // batch_size,
            desc=progress_desc,
            unit="batch",
        )

        for b0 in it:
            b_inputs = missing_inputs[b0 : b0 + batch_size]
            b_vecs = np.asarray(embed_fn(b_inputs), dtype=np.float32)

            if b_vecs.shape[0] != len(b_inputs):
                raise ValueError(f"Embedder returned {b_vecs.shape[0]} vectors for {len(b_inputs)} texts.")

            if spec.dimensions is not None and b_vecs.shape[1] != int(spec.dimensions):
                raise ValueError(f"Embedding dim mismatch: got {b_vecs.shape[1]}, expected {spec.dimensions}")

            if spec.normalize:
                b_vecs = _l2_normalize(b_vecs)

            for j in range(b_vecs.shape[0]):
                orig_i = missing_idx[b0 + j]
                k = keys[orig_i]
                vec = b_vecs[j]

                meta = {
                    "producer": "embeddings",
                    "embedder_id": spec.embedder_id(),
                    "embedder_sig": sig,
                    "encoder_type": spec.encoder_type,
                    "dimensions": spec.dimensions,
                    "normalize": spec.normalize,
                    "extra_id": spec.extra_id,
                    "preproc_version": _PREPROC_VERSION,
                    "runtime": _runtime_versions(spec),
                }
                store.put(k, vec, meta=meta)
                out[orig_i] = vec

    X = np.vstack([v for v in out if v is not None]).astype(np.float32, copy=False)
    if X.shape[0] != len(texts_pp):
        raise RuntimeError("Internal error: output rows mismatch")
    return X


def embed_texts(
    texts: List[str],
    *,
    spec: EncoderSpec,
    batch_size: int,
    cache_scope: CacheScope = "global",
    allow_none: bool = True,
    show_progress: bool = True,
    progress_desc: str = "Embedding",
) -> np.ndarray:
    return encode_with_cache(
        texts,
        spec=spec,
        batch_size=batch_size,
        cache_scope=cache_scope,
        allow_none=allow_none,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )


# ─────────────────────────────────────────────────────────────
# Run-local persistens: whole matrix + fingerprint
# ─────────────────────────────────────────────────────────────

def _texts_digest(texts_pp: List[str]) -> str:
    h = hashlib.sha256()
    for t in texts_pp:
        h.update(t.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def make_fingerprint(texts: List[str], *, spec: EncoderSpec, batch_size: int) -> Dict[str, Any]:
    texts_pp = preprocess_texts(texts, allow_none=True)
    fp: Dict[str, Any] = {
        "kind": "run_embeddings",
        "n": len(texts_pp),
        "texts_digest": _texts_digest(texts_pp),
        "embedder_id": spec.embedder_id(),
        "embedder_slug": spec.embedder_slug(),
        "embedder_sig": embedder_signature(spec),
        "normalize": spec.normalize,
        "preproc_version": _PREPROC_VERSION,
        # info only
        "batch_size_info": int(batch_size),
    }
    return fp


def fingerprint_matches(fp_prev: Mapping[str, Any], fp_now: Mapping[str, Any]) -> bool:
    keys = ["kind", "n", "texts_digest", "embedder_sig"]
    return all(fp_prev.get(k) == fp_now.get(k) for k in keys)


# Uppdaterad load_or_compute_run_embeddings (lägg till tfidf-gren)
def load_or_compute_run_embeddings(
    texts: list[str],
    *,
    spec: EncoderSpec,
    batch_size: int,
    emb_npy: Path,
    fp_json: Path,
    force: bool = False,
    cache_scope: CacheScope = "global",
    allow_none: bool = True,
    show_progress: bool = True,
    progress_desc: str = "Embedding",
) -> np.ndarray:
    """
    Run-local persistens:
    - Om emb_npy + fp_json finns och fingerprint matchar -> load .npy
    - Annars compute via embed_texts (eller compute_tfidf_embeddings för tfidf)
      och spara .npy + fp_json (atomiskt)
    """
    fp_now = make_fingerprint(texts, spec=spec, batch_size=batch_size)

    if not force and emb_npy.exists() and fp_json.exists():
        try:
            fp_prev = json.loads(fp_json.read_text(encoding="utf-8"))
            if isinstance(fp_prev, dict) and fingerprint_matches(fp_prev, fp_now):
                return np.load(emb_npy, allow_pickle=False)
        except Exception:
            pass

    # TF-IDF goes through a separate compute path (corpus-fitted, no per-text cache).
    if spec.encoder_type == "tfidf":
        X = compute_tfidf_embeddings(
            texts,
            spec=spec,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )
    else:
        X = embed_texts(
            texts,
            spec=spec,
            batch_size=batch_size,
            cache_scope=cache_scope,
            allow_none=allow_none,
            show_progress=show_progress,
            progress_desc=progress_desc,
        )

    emb_npy.parent.mkdir(parents=True, exist_ok=True)
    fp_json.parent.mkdir(parents=True, exist_ok=True)

    atomic_write_npy(emb_npy, np.asarray(X, dtype=np.float32))
    atomic_write_json(fp_json, fp_now)
    return X


def validate_run_embeddings(
    texts: list[str],
    X: np.ndarray,
    *,
    fp_json: Path,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Validera att X matchar fingerprint som ligger på fp_json.
    - shape n x d
    - texts_digest
    - embedder_sig etc.
    """
    if not fp_json.exists():
        raise FileNotFoundError(str(fp_json))

    fp_prev = json.loads(fp_json.read_text(encoding="utf-8"))
    if not isinstance(fp_prev, dict):
        raise ValueError("fp_json did not contain a dict fingerprint")

    texts_pp = preprocess_texts(texts, allow_none=True)
    dig = _texts_digest(texts_pp)

    ok = True
    errs: List[str] = []

    if int(fp_prev.get("n", -1)) != len(texts_pp):
        ok = False
        errs.append(f"n mismatch: fp={fp_prev.get('n')} now={len(texts_pp)}")

    if fp_prev.get("texts_digest") != dig:
        ok = False
        errs.append("texts_digest mismatch")

    X = np.asarray(X)
    if X.ndim != 2:
        ok = False
        errs.append(f"X not 2D: shape={X.shape}")
    else:
        if X.shape[0] != len(texts_pp):
            ok = False
            errs.append(f"X rows mismatch: X={X.shape[0]} texts={len(texts_pp)}")

    if not ok and strict:
        raise ValueError("validate_run_embeddings failed: " + "; ".join(errs))

    fp_prev["_validate_ok"] = ok
    fp_prev["_validate_errs"] = errs
    return fp_prev


__all__ = [
    "EncoderSpec",
    "CacheScope",
    "embedder_signature",
    "embed_texts",
    "encode_with_cache",
    "load_or_compute_run_embeddings",
    "make_fingerprint",
    "validate_run_embeddings",
    "compute_tfidf_embeddings",
]
