"""Phase 5 Part B: retrieval over the FAISS index.

Run from the repo root for an ad-hoc query:

    python -m src.rag.retrieve "cybersecurity risks" --k 5 --ticker KO

No LLM API calls - this module only ranks stored chunks. Generation is Phase 6.

Filtering strategy
------------------
Metadata filters are applied POST-HOC to over-fetched FAISS results rather than
through a faiss.IDSelector, for two reasons:

  * With IndexFlatIP the scan is exhaustive and sub-millisecond, so an ID
    selector saves no meaningful time - it only avoids scoring vectors that would
    be discarded anyway.
  * Post-hoc filtering keeps the FAISS call identical whether or not filters are
    present, so filtered and unfiltered results are guaranteed to come from the
    same code path and be directly comparable. An ID selector introduces a second
    path that could diverge.

The cost is that a narrow filter can under-fill k: if the requested slice is rare,
the over-fetch window may contain too few matching rows. This is handled by
escalating the fetch size (k -> 4k -> 16k -> ... -> whole corpus) until k is
satisfied or every vector has been considered, and by reporting `filled=False`
when the corpus genuinely cannot supply k results. At millions of vectors, where
a full scan is no longer cheap, an IDSelector would become the better choice.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

import faiss
import numpy as np
import pandas as pd

from src import config
from src.rag.index import file_sha256

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CITATION_FIELDS = [
    "source_filename", "ticker", "company", "form_type", "fiscal_year",
    "accession_number", "source_url", "chunk_index", "char_start", "char_end",
    "token_count",
]


class StaleIndexError(RuntimeError):
    """The index and the embeddings on disk no longer correspond."""


@dataclass
class SearchResult:
    """One retrieved chunk: its score, its text, and everything needed to cite it."""

    rank: int
    score: float
    row: int
    text: str
    metadata: dict

    def citation(self) -> str:
        m = self.metadata
        return (
            f"{m['company']} {m['form_type']} FY{m['fiscal_year']}, "
            f"chunk {m['chunk_index']} (chars {m['char_start']}-{m['char_end']}), "
            f"accession {m['accession_number']}"
        )


@dataclass
class SearchReport:
    """Results plus what happened on the way to them."""

    query: str
    k: int
    results: list[SearchResult]
    filled: bool
    fetched: int
    dropped_duplicates: list[dict] = field(default_factory=list)

    @property
    def dropped_count(self) -> int:
        return len(self.dropped_duplicates)


class Retriever:
    """Loads the FAISS index and metadata once and searches over them."""

    def __init__(self, verify: bool = True, verbose: bool = False):
        index_path = config.FAISS_DIR / "index.faiss"
        info_path = config.FAISS_DIR / "index_info.json"
        metadata_path = config.EMBEDDINGS_DIR / "metadata.parquet"
        vector_path = config.EMBEDDINGS_DIR / "embeddings.npy"

        for path in (index_path, info_path, metadata_path, vector_path):
            if not path.exists():
                raise SystemExit(
                    f"Missing {path}.\n"
                    "Run: python -m src.rag.chunk && python -m src.rag.embed "
                    "&& python -m src.rag.index"
                )

        self.index = faiss.read_index(str(index_path))
        self.metadata = pd.read_parquet(metadata_path)
        self.info = json.loads(info_path.read_text(encoding="utf-8"))
        self._vector_path = vector_path
        self._vectors: np.ndarray | None = None
        self._model = None

        if verify:
            self._verify()
        if verbose:
            print(
                f"Retriever ready: {self.index.ntotal:,} vectors, "
                f"{len(self.metadata):,} metadata rows, "
                f"{type(self.index).__name__}"
            )

    # -- integrity ---------------------------------------------------------

    def _verify(self) -> None:
        """Fail loudly if the index, metadata and embeddings have drifted apart."""
        if self.index.ntotal != len(self.metadata):
            raise StaleIndexError(
                f"Index holds {self.index.ntotal:,} vectors but metadata has "
                f"{len(self.metadata):,} rows.\n"
                "Likely cause: src.rag.chunk was re-run without re-running "
                "src.rag.embed and src.rag.index.\n"
                "Fix: python -m src.rag.embed && python -m src.rag.index"
            )

        recorded = self.info.get("embeddings_source", {}).get("sha256")
        current = file_sha256(self._vector_path)
        if recorded and recorded != current:
            raise StaleIndexError(
                "The FAISS index was built from a different embeddings file.\n"
                f"  index built from sha256 {recorded[:16]}...\n"
                f"  embeddings.npy is now  sha256 {current[:16]}...\n"
                "Likely cause: src.rag.embed was re-run without rebuilding the "
                "index.\n"
                "Every search would return text attributed to the wrong chunk.\n"
                "Fix: python -m src.rag.index"
            )

    # -- lazy resources ----------------------------------------------------

    @property
    def vectors(self) -> np.ndarray:
        """Stored vectors, used for chunk-to-chunk dedupe similarity.

        Dedupe compares candidates using the vectors already on disk rather than
        re-embedding their text: re-embedding would be slower and, more
        importantly, could disagree with what is actually in the index.
        """
        if self._vectors is None:
            self._vectors = np.load(self._vector_path)
        return self._vectors

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(config.EMBEDDING_MODEL)
        return self._model

    def encode_query(self, query: str) -> np.ndarray:
        vector = self.model.encode([query], normalize_embeddings=True)
        return np.ascontiguousarray(vector, dtype=np.float32)

    # -- filtering ---------------------------------------------------------

    def _matches(self, row: int, filters: dict) -> bool:
        record = self.metadata.iloc[row]
        for key, wanted in filters.items():
            if key not in self.metadata.columns:
                raise KeyError(
                    f"Unknown filter field {key!r}. "
                    f"Available: {sorted(self.metadata.columns)}"
                )
            allowed = wanted if isinstance(wanted, (list, tuple, set)) else [wanted]
            if record[key] not in allowed:
                return False
        return True

    # -- search ------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int = config.TOP_K,
        filters: dict | None = None,
        dedupe: bool = True,
        dedupe_threshold: float = config.DEDUPE_THRESHOLD,
    ) -> SearchReport:
        """Return up to k chunks for a query, optionally filtered and deduped."""
        query_vector = self.encode_query(query)
        total = self.index.ntotal

        fetch = min(total, max(k * config.OVERFETCH_MULTIPLIER, k))
        while True:
            scores, ids = self.index.search(query_vector, fetch)
            candidates = [
                (int(row), float(score))
                for row, score in zip(ids[0], scores[0])
                if row != -1
            ]
            if filters:
                candidates = [c for c in candidates if self._matches(c[0], filters)]

            kept, dropped = self._collapse(candidates, k, dedupe, dedupe_threshold)

            # Enough results, or the corpus is exhausted: stop escalating.
            if len(kept) >= k or fetch >= total:
                break
            fetch = min(total, fetch * config.OVERFETCH_MULTIPLIER)

        results = [
            SearchResult(
                rank=position + 1,
                score=score,
                row=row,
                text=self.metadata.iloc[row]["text"],
                metadata={
                    key: self._native(self.metadata.iloc[row][key])
                    for key in CITATION_FIELDS
                },
            )
            for position, (row, score) in enumerate(kept[:k])
        ]

        return SearchReport(
            query=query,
            k=k,
            results=results,
            filled=len(results) >= k,
            fetched=fetch,
            dropped_duplicates=dropped,
        )

    def _collapse(
        self, candidates: list[tuple[int, float]], k: int, dedupe: bool, threshold: float
    ) -> tuple[list[tuple[int, float]], list[dict]]:
        """Greedily keep the best-scoring chunks, dropping near-duplicates.

        Candidates arrive in descending score order, so keeping the first
        occurrence keeps the highest-scoring member of each duplicate group.
        """
        if not dedupe:
            return candidates[:k], []

        kept: list[tuple[int, float]] = []
        dropped: list[dict] = []

        for row, score in candidates:
            duplicate_of = None
            for kept_row, _ in kept:
                similarity = float(self.vectors[row] @ self.vectors[kept_row])
                if similarity > threshold:
                    duplicate_of = (kept_row, similarity)
                    break

            if duplicate_of is None:
                kept.append((row, score))
                if len(kept) >= k:
                    break
            else:
                kept_row, similarity = duplicate_of
                record, other = self.metadata.iloc[row], self.metadata.iloc[kept_row]
                dropped.append(
                    {
                        "dropped_row": row,
                        "dropped_source": str(record["source_filename"]),
                        "dropped_chunk_index": int(record["chunk_index"]),
                        "dropped_score": round(score, 4),
                        "duplicate_of_row": kept_row,
                        "duplicate_of_source": str(other["source_filename"]),
                        "duplicate_of_chunk_index": int(other["chunk_index"]),
                        "similarity": round(similarity, 4),
                    }
                )

        return kept, dropped

    @staticmethod
    def _native(value):
        """numpy scalar -> python scalar, so results are JSON-serialisable."""
        return value.item() if hasattr(value, "item") else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the filing index.")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--ticker", nargs="*", help="restrict to these tickers")
    parser.add_argument("--fiscal-year", nargs="*", type=int)
    parser.add_argument("--no-dedupe", action="store_true")
    parser.add_argument("--chars", type=int, default=300, help="preview length")
    arguments = parser.parse_args()

    filters: dict = {}
    if arguments.ticker:
        filters["ticker"] = [t.upper() for t in arguments.ticker]
    if arguments.fiscal_year:
        filters["fiscal_year"] = arguments.fiscal_year

    retriever = Retriever(verbose=True)
    report = retriever.search(
        arguments.query,
        k=arguments.k,
        filters=filters or None,
        dedupe=not arguments.no_dedupe,
    )

    print(f"\nquery   : {report.query!r}")
    print(f"filters : {filters or 'none'}")
    print(f"dedupe  : {'off' if arguments.no_dedupe else 'on'}")
    print(f"fetched : {report.fetched} candidates from FAISS")
    if not report.filled:
        print(
            f"WARNING: only {len(report.results)} of k={report.k} results available "
            "- the filter is narrower than the corpus can satisfy."
        )

    for result in report.results:
        preview = " ".join(result.text.split())[: arguments.chars]
        print(f"\n[{result.rank}] {result.score:.4f}  {result.citation()}")
        print(f"    {preview}...")

    if report.dropped_count:
        print(f"\ndeduped {report.dropped_count} near-duplicate(s):")
        for entry in report.dropped_duplicates:
            print(
                f"  - {entry['dropped_source']} chunk {entry['dropped_chunk_index']} "
                f"(score {entry['dropped_score']}) ~ "
                f"{entry['duplicate_of_source']} chunk "
                f"{entry['duplicate_of_chunk_index']} "
                f"(cosine {entry['similarity']})"
            )


if __name__ == "__main__":
    main()
