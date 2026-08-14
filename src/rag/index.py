"""Phase 5 Part A: build a FAISS index over the Phase 4 embeddings.

Run from the repo root (after src.rag.embed):

    python -m src.rag.index

No LLM API calls.

Why IndexFlatIP and not an approximate index
--------------------------------------------
IndexFlatIP is an exhaustive inner-product index - it compares the query against
every stored vector. Three reasons that is the right choice here:

1. **Inner product is exactly cosine.** The Phase 4 vectors are L2-normalised, so
   the dot product of two of them *is* their cosine similarity. No separate
   normalisation step at query time, and scores are directly interpretable
   against the Phase 4 sanity-check numbers.

2. **5,751 vectors is far below the scale where approximation pays.** A flat scan
   over 5,751 x 384 float32 is about 8.4 MB of arithmetic - sub-millisecond. IVF
   and HNSW buy speed by searching only part of the space, and that trade only
   becomes worthwhile when a linear scan is genuinely too slow. At this size an
   approximate index would add tuning parameters (nlist/nprobe, efSearch), a
   training step, and recall below 100%, in exchange for saving no measurable
   time.

3. **Exact search is reproducible.** A flat index returns precisely what a
   brute-force numpy scan returns, every run, with no dependence on training data
   order or probe settings. That property is what lets tests/test_retrieval.py
   assert FAISS results equal a numpy scan exactly - which in turn means any
   retrieval oddity found later is a problem with the embeddings or the query,
   never with the index. Recall@k of an approximate index is itself a variable
   that would have to be measured and defended.

**At millions of vectors this choice would flip.** Beyond roughly 10^6 embeddings
a flat scan stops being free and an approximate index - IVFFlat with a tuned
nprobe, or HNSW where memory allows - becomes the correct choice, accepting ~95-99%
recall for orders-of-magnitude faster search. Nothing in this codebase depends on
the index being flat except the exactness test, so that swap stays cheap.

Stale-index protection
----------------------
The sidecar JSON records a SHA256 of the embeddings file the index was built
from. An index built from superseded embeddings is exactly the silent-failure
class hit in Phase 4 (chunks truncated with no error): shapes stay plausible,
searches keep returning results, and the results are quietly attributed to the
wrong text. src/rag/retrieve.py re-checks this fingerprint at load time.
"""

from __future__ import annotations

import hashlib
import json
import time

import faiss
import numpy as np
import pandas as pd

from src import config

# A unit vector's norm should be 1.0; allow only float32 rounding.
NORM_TOLERANCE = 1e-4


def _rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_vectors() -> tuple[np.ndarray, pd.DataFrame]:
    vector_path = config.EMBEDDINGS_DIR / "embeddings.npy"
    metadata_path = config.EMBEDDINGS_DIR / "metadata.parquet"

    for path in (vector_path, metadata_path):
        if not path.exists():
            raise SystemExit(
                f"Missing {path}. Run: python -m src.rag.chunk && "
                "python -m src.rag.embed"
            )

    vectors = np.load(vector_path)
    metadata = pd.read_parquet(metadata_path)

    if len(vectors) != len(metadata):
        raise SystemExit(
            f"{len(vectors)} vectors but {len(metadata)} metadata rows. "
            "Re-run: python -m src.rag.embed"
        )
    return vectors, metadata


def assert_unit_norm(vectors: np.ndarray) -> np.ndarray:
    """Correctness precondition, not a nicety.

    IndexFlatIP ranks by raw inner product. That equals cosine similarity only
    when both operands are unit vectors. If the stored vectors are not
    normalised, every score this system reports - and every ranking derived from
    them, including the dedupe threshold - is silently wrong, because longer
    vectors win on magnitude rather than on direction.
    """
    norms = np.linalg.norm(vectors, axis=1)
    worst = float(np.abs(norms - 1.0).max())

    if worst > NORM_TOLERANCE:
        offenders = int((np.abs(norms - 1.0) > NORM_TOLERANCE).sum())
        raise SystemExit(
            "ABORT: embeddings are not L2-normalised.\n"
            f"  {offenders:,} of {len(norms):,} vectors deviate from unit norm; "
            f"worst deviation {worst:.2e} (tolerance {NORM_TOLERANCE:.0e}).\n"
            f"  norm range: {norms.min():.6f} - {norms.max():.6f}\n"
            "  IndexFlatIP ranks by inner product, which equals cosine ONLY for\n"
            "  unit vectors. Indexing these would produce silently wrong scores.\n"
            "  Fix: re-run src.rag.embed (it encodes with normalize_embeddings=True)."
        )

    print(f"unit-norm check: PASS "
          f"(max deviation {worst:.2e}, tolerance {NORM_TOLERANCE:.0e})")
    return norms


def build_index(vectors: np.ndarray) -> tuple[faiss.Index, float]:
    """Exhaustive inner-product index over contiguous float32 vectors."""
    prepared = np.ascontiguousarray(vectors, dtype=np.float32)

    started = time.perf_counter()
    index = faiss.IndexFlatIP(prepared.shape[1])
    index.add(prepared)
    elapsed = time.perf_counter() - started

    return index, elapsed


def main() -> None:
    _rule("BUILD FAISS INDEX")

    vectors, metadata = load_vectors()
    print(f"vectors  : {vectors.shape[0]:,} x {vectors.shape[1]} ({vectors.dtype})")
    print(f"metadata : {len(metadata):,} rows")

    assert_unit_norm(vectors)

    index, elapsed = build_index(vectors)
    print(f"\nindex type : {type(index).__name__}")
    print(f"is_trained : {index.is_trained} (flat indexes need no training)")
    print(f"ntotal     : {index.ntotal:,}")
    print(f"dimension  : {index.d}")
    print(f"build time : {elapsed * 1000:.1f} ms")

    # Confirm the index reproduces a brute-force scan before trusting it.
    probe = np.ascontiguousarray(vectors[:1], dtype=np.float32)
    scores, ids = index.search(probe, 3)
    brute = vectors @ vectors[0]
    brute_top = np.argsort(brute)[::-1][:3]
    exact = list(ids[0]) == list(brute_top)
    print(f"self-search: top-3 ids {list(ids[0])}, "
          f"matches brute force: {exact}")
    if not exact:
        raise SystemExit("Index disagrees with brute-force scan - aborting.")

    _rule("SAVING")
    config.FAISS_DIR.mkdir(parents=True, exist_ok=True)
    index_path = config.FAISS_DIR / "index.faiss"
    faiss.write_index(index, str(index_path))

    embeddings_path = config.EMBEDDINGS_DIR / "embeddings.npy"
    fingerprint = file_sha256(embeddings_path)

    info = {
        "phase": "5-faiss-index",
        "index_type": type(index).__name__,
        "metric": "inner_product (== cosine, vectors are L2-normalised)",
        "vector_count": int(index.ntotal),
        "dimension": int(index.d),
        "is_trained": bool(index.is_trained),
        "build_seconds": round(elapsed, 6),
        "approximate": False,
        "index_file_bytes": index_path.stat().st_size,
        "embeddings_source": {
            "path": str(embeddings_path.relative_to(config.PROJECT_ROOT)),
            "sha256": fingerprint,
            "bytes": embeddings_path.stat().st_size,
        },
        "why_flat": (
            "5,751 vectors is far below the scale at which approximate search "
            "pays for itself; exact search keeps results reproducible and "
            "identical to a brute-force scan. Revisit at ~1e6 vectors."
        ),
    }
    info_path = config.FAISS_DIR / "index_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"{index_path}  ({index_path.stat().st_size / 1_048_576:.2f} MB)")
    print(f"{info_path}")
    print(f"\nembeddings fingerprint (sha256): {fingerprint[:16]}...")
    print("Retrieval re-checks this at load time to catch a stale index.")


if __name__ == "__main__":
    main()
