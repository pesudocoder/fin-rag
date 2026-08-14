"""Phase 5 Part C: retrieval evaluation harness.

Run from the repo root:

    python -m src.rag.eval_retrieval

No LLM API calls. This measures RETRIEVAL only.

What this measures, and what it does not
----------------------------------------
Each probe declares which company's filing should supply the answer. The harness
checks whether a chunk from that company appears in the top-k, and at what rank.

That is **source-level checking, not answer correctness.** A probe counts as a
hit when the right company's document is retrieved - it says nothing about
whether the retrieved passage actually answers the question, whether it is the
best passage available, or whether an LLM reading it would produce a truthful
answer. Retrieving the right document is a necessary but not sufficient condition
for a grounded answer. Answer quality and hallucination rate are Phase 7.

Probe design
------------
Per the Phase 4 limitation, probes target qualitative and risk-factor content
only. Table structure is destroyed during HTML extraction, so figures are
decoupled from their row and period labels; a probe asking for a specific number
would be testing a capability the corpus cannot support and would fail for
reasons unrelated to retrieval quality.

The set covers three categories:
  * company-specific risk factors, where one company should dominate
  * cross-company topics (cybersecurity, FX), where several are legitimate
  * out-of-corpus topics, where the correct behaviour is a low top score - these
    exist to check the index does not return confident nonsense for questions it
    has no material on, which is precisely the setup for a Phase 7 hallucination.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src import config
from src.rag.retrieve import Retriever

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# expected_tickers=None marks a probe whose subject is absent from the corpus.
PROBES = [
    # -- company-specific risk factors ---------------------------------------
    {"query": "risks from disruption of our supply chain and single source suppliers",
     "expected_tickers": ["AAPL"], "category": "company-specific"},
    {"query": "dependence on outsourcing partners for manufacturing and assembly",
     "expected_tickers": ["AAPL"], "category": "company-specific"},
    {"query": "risks relating to concentrate operations and independent bottling partners",
     "expected_tickers": ["KO"], "category": "company-specific"},
    {"query": "obesity concerns and changing consumer preferences about sugar sweetened beverages",
     "expected_tickers": ["KO"], "category": "company-specific"},
    {"query": "water scarcity and poor water quality affecting our production",
     "expected_tickers": ["KO"], "category": "company-specific"},
    {"query": "credit risk arising from wholesale and consumer lending portfolios",
     "expected_tickers": ["JPM"], "category": "company-specific"},
    {"query": "regulatory capital and liquidity requirements under Basel rules",
     "expected_tickers": ["JPM"], "category": "company-specific"},
    {"query": "risks of operating broker dealer and market making businesses",
     "expected_tickers": ["JPM"], "category": "company-specific"},
    {"query": "competition in cloud computing and infrastructure services",
     "expected_tickers": ["MSFT"], "category": "company-specific"},
    {"query": "risks related to developing and deploying artificial intelligence",
     "expected_tickers": ["MSFT"], "category": "company-specific"},
    {"query": "our gaming business including Xbox consoles and content",
     "expected_tickers": ["MSFT"], "category": "company-specific"},

    # -- cross-company topics ------------------------------------------------
    {"query": "cybersecurity incidents and unauthorized access to our systems",
     "expected_tickers": ["AAPL", "JPM", "KO", "MSFT"], "category": "cross-company"},
    {"query": "foreign currency exchange rate fluctuations affecting results",
     "expected_tickers": ["AAPL", "JPM", "KO", "MSFT"], "category": "cross-company"},
    {"query": "climate change and environmental sustainability risks",
     "expected_tickers": ["AAPL", "JPM", "KO", "MSFT"], "category": "cross-company"},

    # -- deliberately absent from the corpus ---------------------------------
    {"query": "clinical trial results for our new oncology drug candidate",
     "expected_tickers": None, "category": "out-of-corpus"},
    {"query": "aircraft fleet fuel hedging and aircraft lease obligations",
     "expected_tickers": None, "category": "out-of-corpus"},
]


def _rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def evaluate(retriever: Retriever, dedupe: bool, k: int) -> dict:
    """Run every probe and collect per-probe and aggregate numbers."""
    rows = []

    for probe in PROBES:
        report = retriever.search(probe["query"], k=k, dedupe=dedupe)
        tickers = [result.metadata["ticker"] for result in report.results]

        expected = probe["expected_tickers"]
        rank = None
        if expected:
            for position, ticker in enumerate(tickers, start=1):
                if ticker in expected:
                    rank = position
                    break

        rows.append(
            {
                "query": probe["query"],
                "category": probe["category"],
                "expected_tickers": expected,
                "retrieved": [
                    {
                        "rank": result.rank,
                        "score": round(result.score, 4),
                        "ticker": result.metadata["ticker"],
                        "fiscal_year": result.metadata["fiscal_year"],
                        "chunk_index": result.metadata["chunk_index"],
                        "source": result.metadata["source_filename"],
                    }
                    for result in report.results
                ],
                "tickers_in_topk": tickers,
                "unique_companies_in_topk": len(set(tickers)),
                "first_correct_rank": rank,
                "top_score": round(report.results[0].score, 4) if report.results else None,
                "dropped_duplicates": report.dropped_duplicates,
                "dropped_count": report.dropped_count,
            }
        )

    scored = [row for row in rows if row["expected_tickers"]]
    absent = [row for row in rows if not row["expected_tickers"]]

    def hit_rate(limit: int) -> float:
        hits = sum(
            1 for row in scored
            if row["first_correct_rank"] and row["first_correct_rank"] <= limit
        )
        return round(hits / len(scored), 4)

    reciprocal = [
        1 / row["first_correct_rank"] if row["first_correct_rank"] else 0.0
        for row in scored
    ]

    return {
        "dedupe": dedupe,
        "k": k,
        "probe_count": len(rows),
        "scored_probe_count": len(scored),
        "hit_rate_at_1": hit_rate(1),
        "hit_rate_at_3": hit_rate(3),
        "hit_rate_at_5": hit_rate(5),
        "mrr": round(float(np.mean(reciprocal)), 4),
        "total_duplicates_dropped": sum(row["dropped_count"] for row in rows),
        "out_of_corpus_top_scores": {
            row["query"]: row["top_score"] for row in absent
        },
        "mean_unique_companies_in_topk": round(
            float(np.mean([row["unique_companies_in_topk"] for row in rows])), 3
        ),
        "probes": rows,
    }


def print_run(run: dict) -> None:
    label = "DEDUPE ON" if run["dedupe"] else "DEDUPE OFF"
    _rule(f"PROBE RESULTS - {label} (k={run['k']})")

    for row in run["probes"]:
        expected = ", ".join(row["expected_tickers"]) if row["expected_tickers"] else "NONE (absent)"
        rank = row["first_correct_rank"]
        verdict = f"rank {rank}" if rank else ("n/a" if not row["expected_tickers"] else "MISS")
        print(f"\n[{row['category']}] {row['query']}")
        print(f"  expected: {expected}   -> {verdict}")
        for result in row["retrieved"]:
            mark = "*" if row["expected_tickers"] and result["ticker"] in row["expected_tickers"] else " "
            print(
                f"   {mark} {result['rank']}. {result['score']:.4f}  "
                f"{result['ticker']} FY{result['fiscal_year']} chunk {result['chunk_index']}"
            )
        if row["dropped_count"]:
            for entry in row["dropped_duplicates"]:
                print(
                    f"     [deduped] {entry['dropped_source']} chunk "
                    f"{entry['dropped_chunk_index']} ~ "
                    f"{entry['duplicate_of_source']} chunk "
                    f"{entry['duplicate_of_chunk_index']} (cosine {entry['similarity']})"
                )

    _rule(f"AGGREGATE - {label}")
    print(f"scored probes (with an expected ticker): {run['scored_probe_count']}")
    print(f"hit rate @1 : {run['hit_rate_at_1']:.4f}")
    print(f"hit rate @3 : {run['hit_rate_at_3']:.4f}")
    print(f"hit rate @5 : {run['hit_rate_at_5']:.4f}")
    print(f"MRR         : {run['mrr']:.4f}")
    print(f"duplicates dropped across all probes: {run['total_duplicates_dropped']}")
    print(f"mean unique companies in top-{run['k']}: {run['mean_unique_companies_in_topk']}")

    print("\nout-of-corpus probes (lower is better - nothing relevant exists):")
    for query, score in run["out_of_corpus_top_scores"].items():
        print(f"  {score:.4f}  {query}")


def compare(with_dedupe: dict, without_dedupe: dict) -> dict:
    """What changed when near-duplicates were removed."""
    _rule("EFFECT OF DEDUPLICATION")

    rows = []
    for on, off in zip(with_dedupe["probes"], without_dedupe["probes"]):
        changed = on["tickers_in_topk"] != off["tickers_in_topk"]
        rows.append(
            {
                "query": on["query"],
                "dropped": on["dropped_count"],
                "companies_off": off["unique_companies_in_topk"],
                "companies_on": on["unique_companies_in_topk"],
                "topk_changed": changed,
            }
        )

    frame = pd.DataFrame(rows)
    frame["query"] = frame["query"].str.slice(0, 46)
    print(frame.to_string(index=False))

    summary = {
        "total_duplicates_dropped": int(frame["dropped"].sum()),
        "probes_affected": int((frame["dropped"] > 0).sum()),
        "probes_with_changed_topk": int(frame["topk_changed"].sum()),
        "mean_unique_companies_dedupe_off": round(float(frame["companies_off"].mean()), 3),
        "mean_unique_companies_dedupe_on": round(float(frame["companies_on"].mean()), 3),
        "hit_rate_at_5_off": without_dedupe["hit_rate_at_5"],
        "hit_rate_at_5_on": with_dedupe["hit_rate_at_5"],
        "mrr_off": without_dedupe["mrr"],
        "mrr_on": with_dedupe["mrr"],
    }

    print(f"\nduplicates dropped   : {summary['total_duplicates_dropped']} "
          f"across {summary['probes_affected']} of {len(frame)} probes")
    print(f"top-k composition changed on {summary['probes_with_changed_topk']} probes")
    print(f"mean unique companies in top-k: "
          f"{summary['mean_unique_companies_dedupe_off']} (off) -> "
          f"{summary['mean_unique_companies_dedupe_on']} (on)")
    print(f"hit rate @5: {summary['hit_rate_at_5_off']} (off) -> "
          f"{summary['hit_rate_at_5_on']} (on)")
    print(f"MRR        : {summary['mrr_off']} (off) -> {summary['mrr_on']} (on)")
    return summary


def main() -> None:
    _rule("RETRIEVAL EVALUATION")
    print("SCOPE: this measures whether the right COMPANY'S FILING is retrieved.")
    print("It does NOT measure answer correctness, passage quality, or whether an")
    print("LLM reading these chunks would answer truthfully. Retrieving the right")
    print("document is necessary but not sufficient for a grounded answer.")
    print("Answer quality and hallucination rate are measured in Phase 7.")
    print("\nProbes target qualitative / risk-factor content only: table structure")
    print("is destroyed during extraction (Phase 4 limitation), so figure-level")
    print("questions would fail for reasons unrelated to retrieval.")

    retriever = Retriever(verbose=True)

    with_dedupe = evaluate(retriever, dedupe=True, k=config.TOP_K)
    without_dedupe = evaluate(retriever, dedupe=False, k=config.TOP_K)

    print_run(with_dedupe)
    print_run(without_dedupe)
    dedupe_summary = compare(with_dedupe, without_dedupe)

    _rule("SAVING")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "5-retrieval-eval",
        "scope": (
            "Ticker-level source checking only. Not answer correctness - see "
            "Phase 7 for answer quality and hallucination rate."
        ),
        "index_type": retriever.info["index_type"],
        "vector_count": retriever.info["vector_count"],
        "k": config.TOP_K,
        "dedupe_threshold": config.DEDUPE_THRESHOLD,
        "overfetch_multiplier": config.OVERFETCH_MULTIPLIER,
        "probe_count": len(PROBES),
        "with_dedupe": with_dedupe,
        "without_dedupe": without_dedupe,
        "dedupe_effect": dedupe_summary,
    }
    path = config.RESULTS_DIR / "retrieval_eval.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{path}")


if __name__ == "__main__":
    main()
