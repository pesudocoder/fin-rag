"""Phase 5 retrieval tests.

    pytest tests/test_retrieval.py -v

Covers the invariants that would otherwise fail silently:

  * the index and metadata describe the same number of chunks
  * a query with an obvious correct source retrieves that source
  * metadata filters genuinely restrict the result set
  * deduplication removes the verified cross-year duplicates
  * FAISS IndexFlatIP returns exactly what a brute-force numpy scan returns

The last one is the load-bearing test for the Phase 5 design decision. A flat
index is only worth choosing over an approximate one if it is genuinely exact;
this asserts that it is, which means any future retrieval oddity can be
attributed to the embeddings or the query rather than to the index.

No LLM API calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.rag.retrieve import Retriever

# A query whose correct source is unambiguous: only Coca-Cola's filings discuss
# concentrate operations and independent bottling partners.
KNOWN_QUERY = "risks relating to concentrate operations and independent bottling partners"
KNOWN_TICKER = "KO"

# Verified in Phase 4: this query surfaces the same risk-factor text from both
# fiscal years, and the two chunks embed identically.
DUPLICATE_QUERY = "risks from disruption of our supply chain and single source suppliers"


@pytest.fixture(scope="module")
def retriever():
    if not (config.FAISS_DIR / "index.faiss").exists():
        pytest.skip(
            "No FAISS index. Run: python -m src.rag.chunk && "
            "python -m src.rag.embed && python -m src.rag.index"
        )
    return Retriever()


def test_index_and_metadata_counts_match(retriever):
    assert retriever.index.ntotal == len(retriever.metadata), (
        f"{retriever.index.ntotal} vectors vs {len(retriever.metadata)} metadata "
        "rows - re-run src.rag.embed and src.rag.index"
    )


def test_index_dimension_matches_config(retriever):
    assert retriever.index.d == config.EMBEDDING_DIM


def test_known_query_retrieves_expected_ticker(retriever):
    report = retriever.search(KNOWN_QUERY, k=config.TOP_K)
    tickers = [result.metadata["ticker"] for result in report.results]

    assert report.results, "no results returned"
    assert KNOWN_TICKER in tickers, (
        f"expected {KNOWN_TICKER} in top-{config.TOP_K}, got {tickers}"
    )
    assert tickers[0] == KNOWN_TICKER, (
        f"expected {KNOWN_TICKER} at rank 1, got {tickers[0]}"
    )


def test_results_carry_full_citation_metadata(retriever):
    report = retriever.search(KNOWN_QUERY, k=3)
    required = {
        "source_filename", "ticker", "company", "form_type", "fiscal_year",
        "accession_number", "source_url", "chunk_index", "char_start", "char_end",
    }
    for result in report.results:
        assert required <= set(result.metadata), required - set(result.metadata)
        assert result.citation()


def test_scores_are_descending(retriever):
    report = retriever.search(KNOWN_QUERY, k=config.TOP_K)
    scores = [result.score for result in report.results]
    assert scores == sorted(scores, reverse=True), scores


class TestFiltering:
    def test_ticker_filter_restricts_results(self, retriever):
        report = retriever.search(
            "supply chain disruption", k=config.TOP_K, filters={"ticker": "MSFT"}
        )
        tickers = {result.metadata["ticker"] for result in report.results}

        assert report.results, "filter returned nothing"
        assert tickers == {"MSFT"}, f"filter leaked other tickers: {tickers}"

    def test_filter_actually_changes_the_result_set(self, retriever):
        """A filter that matched everything would pass the test above vacuously."""
        unfiltered = retriever.search("supply chain disruption", k=config.TOP_K)
        filtered = retriever.search(
            "supply chain disruption", k=config.TOP_K, filters={"ticker": "MSFT"}
        )

        unfiltered_tickers = {r.metadata["ticker"] for r in unfiltered.results}
        assert unfiltered_tickers != {"MSFT"}, (
            "unfiltered results are already all MSFT, so this query cannot "
            "demonstrate that filtering works"
        )
        assert [r.row for r in filtered.results] != [r.row for r in unfiltered.results]

    def test_fiscal_year_filter(self, retriever):
        report = retriever.search(
            "cybersecurity risks", k=config.TOP_K, filters={"fiscal_year": 2023}
        )
        years = {result.metadata["fiscal_year"] for result in report.results}
        assert years == {2023}, years

    def test_list_valued_filter(self, retriever):
        report = retriever.search(
            "cybersecurity risks", k=config.TOP_K, filters={"ticker": ["KO", "MSFT"]}
        )
        tickers = {result.metadata["ticker"] for result in report.results}
        assert tickers <= {"KO", "MSFT"}, tickers

    def test_unknown_filter_field_raises(self, retriever):
        with pytest.raises(KeyError):
            retriever.search("anything", filters={"not_a_column": "x"})

    def test_overly_narrow_filter_reports_underfill(self, retriever):
        """A filter narrower than k must report filled=False, not pad or crash."""
        report = retriever.search(
            "supply chain",
            k=5,
            filters={"ticker": "AAPL", "fiscal_year": 2023, "chunk_index": [1, 2]},
        )
        assert len(report.results) < 5
        assert report.filled is False


class TestDeduplication:
    def test_dedupe_drops_known_cross_year_duplicates(self, retriever):
        report = retriever.search(DUPLICATE_QUERY, k=config.TOP_K, dedupe=True)
        assert report.dropped_count > 0, (
            "expected near-duplicates to be dropped for a query whose risk-factor "
            "text is repeated across fiscal years"
        )
        for entry in report.dropped_duplicates:
            assert entry["similarity"] > config.DEDUPE_THRESHOLD

    def test_dedupe_off_keeps_the_duplicates(self, retriever):
        report = retriever.search(DUPLICATE_QUERY, k=config.TOP_K, dedupe=False)
        assert report.dropped_count == 0

    def test_dedupe_reduces_duplicate_pairs_in_results(self, retriever):
        """The point of dedupe: no two returned chunks may be near-identical."""
        without = retriever.search(DUPLICATE_QUERY, k=config.TOP_K, dedupe=False)
        with_dedupe = retriever.search(DUPLICATE_QUERY, k=config.TOP_K, dedupe=True)

        def duplicate_pairs(report):
            rows = [result.row for result in report.results]
            return sum(
                1
                for i, a in enumerate(rows)
                for b in rows[i + 1:]
                if float(retriever.vectors[a] @ retriever.vectors[b])
                > config.DEDUPE_THRESHOLD
            )

        assert duplicate_pairs(without) > 0, (
            "this query no longer surfaces duplicates; pick another for this test"
        )
        assert duplicate_pairs(with_dedupe) == 0

    def test_dedupe_keeps_the_higher_scoring_member(self, retriever):
        """Dropping the better chunk of a duplicate pair would lose information."""
        report = retriever.search(DUPLICATE_QUERY, k=config.TOP_K, dedupe=True)
        kept_scores = {result.row: result.score for result in report.results}
        for entry in report.dropped_duplicates:
            kept = kept_scores.get(entry["duplicate_of_row"])
            if kept is not None:
                assert kept >= entry["dropped_score"] - 1e-6


def test_faiss_matches_brute_force_numpy_scan(retriever):
    """IndexFlatIP must introduce no approximation error whatsoever.

    This is what justifies choosing a flat index over IVF/HNSW: results are
    exactly a brute-force scan, so retrieval is reproducible and the index is
    never a suspect when something looks wrong.

    Exactness is asserted on the SCORE SEQUENCE, not on the id sequence. This
    corpus contains chunks that are byte-identical across fiscal years - the
    same risk-factor paragraph copied from one 10-K to the next - so their
    vectors are identical and they tie exactly. Tie-break order is
    implementation-defined and differs between FAISS and numpy's argsort; that
    is not approximation error. What must hold is that the returned scores match
    to float precision at every rank, and that any positional id difference
    occurs only where the scores are equal.
    """
    queries = [
        KNOWN_QUERY,
        "cybersecurity incidents and unauthorized access",
        "regulatory capital requirements under Basel rules",
    ]
    k = 10
    vectors = retriever.vectors

    for query in queries:
        query_vector = retriever.encode_query(query)

        faiss_scores, faiss_ids = retriever.index.search(query_vector, k)
        brute_scores = vectors @ query_vector[0]
        brute_ids = np.argsort(-brute_scores, kind="stable")[:k]

        # 1. Identical score sequence: no approximation anywhere.
        np.testing.assert_allclose(
            faiss_scores[0],
            brute_scores[brute_ids],
            rtol=0,
            atol=1e-6,
            err_msg=f"score mismatch for {query!r} - the index is not exact",
        )

        # 2. Every id FAISS returned scores what brute force says it scores.
        for returned_id, returned_score in zip(faiss_ids[0], faiss_scores[0]):
            np.testing.assert_allclose(
                returned_score, brute_scores[returned_id], rtol=0, atol=1e-6
            )

        # 3. Any positional disagreement must be an exact tie, never a reorder
        #    of genuinely different scores.
        for position, (faiss_id, brute_id) in enumerate(zip(faiss_ids[0], brute_ids)):
            if faiss_id != brute_id:
                assert abs(
                    float(brute_scores[faiss_id]) - float(brute_scores[brute_id])
                ) < 1e-6, (
                    f"FAISS and brute force disagree at rank {position} for "
                    f"{query!r} on chunks with DIFFERENT scores "
                    f"({brute_scores[faiss_id]:.8f} vs {brute_scores[brute_id]:.8f}) "
                    "- this is real approximation error, not a tie-break."
                )


def test_exact_score_ties_exist_in_this_corpus(retriever):
    """Documents why the test above compares scores rather than ids.

    Verbatim cross-year duplication is a property of this corpus, not a quirk of
    one query. If this ever stops holding, the tie-break allowance above is
    weaker than it needs to be and should be tightened.
    """
    query_vector = retriever.encode_query(KNOWN_QUERY)
    scores = retriever.vectors @ query_vector[0]
    top = np.sort(scores)[::-1][:10]
    ties = sum(1 for a, b in zip(top, top[1:]) if abs(a - b) < 1e-6)
    assert ties > 0, "no exact score ties in the top 10 - duplicates may be gone"


def test_stale_index_detection_is_wired_up(retriever):
    """The fingerprint check must be comparing a real recorded hash."""
    recorded = retriever.info.get("embeddings_source", {}).get("sha256")
    assert recorded and len(recorded) == 64, (
        "index_info.json has no usable embeddings sha256, so a stale index "
        "would not be detected"
    )
