"""Phase 4 Part C: embed the chunks with a local sentence-transformers model.

Run from the repo root (after src.rag.chunk):

    python -m src.rag.embed

Encodes every chunk from data/processed/chunks.json with config.EMBEDDING_MODEL
and writes the vectors plus aligned metadata to data/processed/embeddings/.

No LLM API calls - the embedding model runs locally, which is why the corpus can
be re-embedded freely while tuning chunk size.

Alignment contract
------------------
Row i of embeddings.npy corresponds to row i of metadata.parquet. Both are
written in the same pass from the same list and the lengths are asserted equal
before saving, so the mapping cannot silently drift.
"""

from __future__ import annotations

import json
import random
import sys
import time

import numpy as np
import pandas as pd

from src import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Probes for the sanity check: each should retrieve topically matching text.
PROBE_QUERIES = [
    "risks related to supply chain disruption and manufacturing",
    "revenue recognition accounting policy",
    "cybersecurity threats and data breaches",
    "credit losses and loan loss provisions",
]

# Deliberately unrelated to any of the probes above; used as a contrast anchor.
CONTRAST_QUERY = "recipes for baking sourdough bread at home"


def _rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def load_chunks() -> list[dict]:
    if not config.CHUNKS_FILE.exists():
        raise SystemExit(
            f"No chunks at {config.CHUNKS_FILE}. Run: python -m src.rag.chunk"
        )
    return json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))


def sanity_check(model, vectors: np.ndarray, frame: pd.DataFrame) -> dict:
    """Verify the embeddings carry semantic signal before Phase 5 depends on them.

    Vectors are L2-normalised, so a dot product is the cosine similarity.

    Three comparisons:
      1. random chunk pairs        - the noise floor
      2. adjacent chunks, same doc - should sit clearly above the floor
      3. topical probe queries     - should retrieve on-topic text well above it

    If topically related text does not score higher than random text, the
    embeddings are wrong and Phase 5 retrieval would fail in ways that are hard
    to diagnose later.
    """
    _rule("SANITY CHECK: are these embeddings meaningful?")
    rng = random.Random(config.RANDOM_SEED)

    pairs = [
        (rng.randrange(len(vectors)), rng.randrange(len(vectors)))
        for _ in range(2000)
    ]
    random_sims = np.array([float(vectors[a] @ vectors[b]) for a, b in pairs])
    random_mean = float(random_sims.mean())

    adjacent = [
        (i, i + 1)
        for i in range(len(frame) - 1)
        if frame.at[i, "source_filename"] == frame.at[i + 1, "source_filename"]
        and frame.at[i, "chunk_index"] + 1 == frame.at[i + 1, "chunk_index"]
    ]
    sampled = rng.sample(adjacent, min(2000, len(adjacent)))
    adjacent_sims = np.array([float(vectors[a] @ vectors[b]) for a, b in sampled])
    adjacent_mean = float(adjacent_sims.mean())

    print(f"1. random chunk pairs        (n={len(pairs):,}): "
          f"mean cosine {random_mean:.4f}   <- noise floor")
    print(f"2. adjacent chunks, same doc (n={len(sampled):,}): "
          f"mean cosine {adjacent_mean:.4f}   "
          f"({adjacent_mean - random_mean:+.4f} vs floor)")

    print("\n3. topical probe queries -> best matching chunks:")
    probe_vectors = model.encode(
        PROBE_QUERIES + [CONTRAST_QUERY], normalize_embeddings=True
    )
    probe_top = []
    for query, vector in zip(PROBE_QUERIES, probe_vectors[:-1]):
        scores = vectors @ vector
        best = int(np.argmax(scores))
        probe_top.append(float(scores[best]))
        row = frame.iloc[best]
        preview = " ".join(row["text"].split())[:150]
        print(f"\n   query : {query}")
        print(f"   top   : {scores[best]:.4f}  [{row['ticker']} {row['form_type']} "
              f"{row['fiscal_year']} chunk {row['chunk_index']}]")
        print(f"   text  : {preview}...")

    contrast_scores = vectors @ probe_vectors[-1]
    contrast_best = float(contrast_scores.max())
    print(f"\n   contrast query ('{CONTRAST_QUERY}')")
    print(f"   best match anywhere in the corpus: {contrast_best:.4f}")

    probe_mean = float(np.mean(probe_top))
    checks = {
        "adjacent_above_random": adjacent_mean > random_mean,
        "probes_above_random": probe_mean > random_mean + 0.15,
        "probes_above_contrast": probe_mean > contrast_best,
    }
    passed = all(checks.values())

    print("\n   " + "-" * 60)
    print(f"   mean top-1 score, on-topic probes : {probe_mean:.4f}")
    print(f"   best score, off-topic contrast    : {contrast_best:.4f}")
    print(f"   random-pair noise floor           : {random_mean:.4f}")
    for name, ok in checks.items():
        print(f"   [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n   VERDICT: {'PASS' if passed else 'FAIL'} - embeddings "
          f"{'carry semantic signal' if passed else 'DO NOT behave as expected'}")
    if not passed:
        print("   Do not proceed to Phase 5 until this passes.")

    return {
        "random_pair_mean_cosine": round(random_mean, 4),
        "adjacent_chunk_mean_cosine": round(adjacent_mean, 4),
        "probe_top1_mean_cosine": round(probe_mean, 4),
        "offtopic_contrast_best_cosine": round(contrast_best, 4),
        "checks": checks,
        "passed": passed,
    }


def main() -> None:
    _rule("EMBEDDING CHUNKS")
    chunks = load_chunks()
    texts = [chunk["text"] for chunk in chunks]
    print(f"chunks     : {len(chunks):,}")
    print(f"model      : {config.EMBEDDING_MODEL} (local, no API calls)")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    max_tokens = model.max_seq_length
    print(f"max_seq_len: {max_tokens} word pieces")

    over = sum(1 for chunk in chunks if chunk.get("token_count", 0) > max_tokens)
    if over:
        print(
            f"\n!! WARNING: {over:,} of {len(chunks):,} chunks "
            f"({over / len(chunks) * 100:.1f}%) are longer than {max_tokens} tokens.\n"
            f"   Everything past token {max_tokens} in those chunks is dropped before\n"
            f"   encoding and will be unretrievable in Phase 5. Lower CHUNK_SIZE in\n"
            f"   src/config.py (currently {config.CHUNK_SIZE}) and re-run chunk + embed."
        )

    print()
    started = time.perf_counter()
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit vectors: dot product == cosine
    ).astype(np.float32)
    elapsed = time.perf_counter() - started

    frame = pd.DataFrame(chunks)
    assert len(frame) == len(vectors) == len(chunks), "metadata/vector length mismatch"

    _rule("EMBEDDING SUMMARY")
    size_bytes = vectors.nbytes
    print(f"vectors        : {vectors.shape[0]:,}")
    print(f"dimension      : {vectors.shape[1]}")
    print(f"dtype          : {vectors.dtype}")
    print(f"array size     : {size_bytes / 1_048_576:.2f} MB "
          f"({size_bytes:,} bytes)")
    print(f"encoding time  : {elapsed:.2f} s "
          f"({len(texts) / elapsed:.1f} chunks/s, CPU)")
    norms = np.linalg.norm(vectors, axis=1)
    print(f"L2 norms       : min {norms.min():.6f}, max {norms.max():.6f} "
          f"(normalised, so cosine == dot product)")

    checks = sanity_check(model, vectors, frame)

    _rule("SAVING")
    config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    vector_path = config.EMBEDDINGS_DIR / "embeddings.npy"
    metadata_path = config.EMBEDDINGS_DIR / "metadata.parquet"

    np.save(vector_path, vectors)
    frame.to_parquet(metadata_path, index=False)
    print(f"{vector_path}  {vectors.shape}")
    print(f"{metadata_path}  {frame.shape}")

    info = {
        "phase": "4-embeddings",
        "model": config.EMBEDDING_MODEL,
        "local_no_api": True,
        "vector_count": int(vectors.shape[0]),
        "dimension": int(vectors.shape[1]),
        "dtype": str(vectors.dtype),
        "normalized": True,
        "array_bytes": int(size_bytes),
        "encoding_seconds": round(elapsed, 2),
        "chunks_per_second": round(len(texts) / elapsed, 1),
        "max_seq_length": int(max_tokens),
        "chunks_truncated_at_embed_time": int(over),
        "alignment": "row i of embeddings.npy == row i of metadata.parquet",
        "sanity_check": checks,
    }
    info_path = config.EMBEDDINGS_DIR / "embedding_info.json"
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"{info_path}")

    # Fold the embedding summary into the committed chunking stats.
    stats_path = config.RESULTS_DIR / "chunking_stats.json"
    if stats_path.exists():
        summary = json.loads(stats_path.read_text(encoding="utf-8"))
        summary["embeddings"] = info
        stats_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"{stats_path}  (embedding summary merged in)")

    if not checks["passed"]:
        raise SystemExit("Sanity check FAILED - see above before continuing to Phase 5.")


if __name__ == "__main__":
    main()
