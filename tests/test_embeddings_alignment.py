"""Guards the embeddings <-> metadata row alignment invariant.

Row i of embeddings.npy must correspond to row i of metadata.parquet. Phase 5
retrieval looks a vector up by index and reads its citation fields from the same
index in the metadata; if the two ever drift, retrieval returns real text
attributed to the wrong filing, year and character span. Nothing raises - the
answers just become quietly, confidently wrong.

The realistic way this breaks is re-running src.rag.chunk without re-running
src.rag.embed: chunks.json changes, embeddings.npy does not, and the arrays
still have plausible shapes.

Equal lengths alone do not prove alignment, so these tests re-embed the stored
text of selected rows and require the result to match the stored vector for that
same index. Vectors are L2-normalised, so the dot product is the cosine.

    pytest tests/test_embeddings_alignment.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config

# Cosine below this means the stored vector is not an encoding of the stored
# text at that index. The tolerance covers float32 rounding only.
COSINE_TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def artifacts():
    vector_path = config.EMBEDDINGS_DIR / "embeddings.npy"
    metadata_path = config.EMBEDDINGS_DIR / "metadata.parquet"

    if not vector_path.exists() or not metadata_path.exists():
        pytest.skip(
            "No embeddings on disk. Run: python -m src.rag.chunk && "
            "python -m src.rag.embed"
        )

    return np.load(vector_path), pd.read_parquet(metadata_path)


@pytest.fixture(scope="module")
def model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


@pytest.fixture(scope="module")
def probe_indices(artifacts):
    """First, last, and evenly spaced interior rows."""
    vectors, _ = artifacts
    count = len(vectors)
    return sorted({0, count // 4, count // 2, (3 * count) // 4, count - 1})


def test_counts_match(artifacts):
    vectors, metadata = artifacts
    assert len(vectors) == len(metadata), (
        f"{len(vectors)} vectors but {len(metadata)} metadata rows - "
        "re-run src.rag.embed"
    )


def test_dimension_and_dtype(artifacts):
    vectors, _ = artifacts
    assert vectors.shape[1] == config.EMBEDDING_DIM
    assert vectors.dtype == np.float32


def test_vectors_are_normalised(artifacts):
    """Cosine == dot product only holds for unit vectors."""
    vectors, _ = artifacts
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), (
        f"norms range {norms.min():.6f}-{norms.max():.6f}, expected 1.0"
    )


def test_metadata_has_citation_fields(artifacts):
    """Phase 6 citations need every one of these."""
    _, metadata = artifacts
    required = {
        "text", "source_filename", "ticker", "company", "form_type",
        "fiscal_year", "accession_number", "source_url", "chunk_index",
        "char_start", "char_end",
    }
    assert required <= set(metadata.columns), required - set(metadata.columns)


def test_stored_vector_matches_reembedded_text(artifacts, model, probe_indices):
    """The invariant: re-embedding row i's text reproduces vector i."""
    vectors, metadata = artifacts

    texts = [metadata.iloc[i]["text"] for i in probe_indices]
    fresh = model.encode(texts, normalize_embeddings=True)

    failures = []
    for position, index in enumerate(probe_indices):
        cosine = float(fresh[position] @ vectors[index])
        if abs(cosine - 1.0) > COSINE_TOLERANCE:
            failures.append(f"row {index}: cosine {cosine:.6f}")

    assert not failures, (
        "Stored vectors do not match their metadata text at: "
        + "; ".join(failures)
        + ". Embeddings are stale or misaligned - re-run src.rag.embed."
    )


def test_misalignment_would_be_detected(artifacts, model, probe_indices):
    """Negative control.

    Confirms the check above can actually fail. If neighbouring rows embedded
    almost identically, a shifted array would still pass and the test would be
    worthless. Compare row i's text against row i+1's vector and require a
    clearly lower score.
    """
    vectors, metadata = artifacts
    candidates = [i for i in probe_indices if i + 1 < len(vectors)]

    texts = [metadata.iloc[i]["text"] for i in candidates]
    fresh = model.encode(texts, normalize_embeddings=True)

    shifted = [float(fresh[p] @ vectors[i + 1]) for p, i in enumerate(candidates)]
    assert max(shifted) < 1.0 - COSINE_TOLERANCE, (
        "An off-by-one shift scores as high as a correct match, so the "
        f"alignment test cannot detect drift. Max shifted cosine: {max(shifted):.6f}"
    )
