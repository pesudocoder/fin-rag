"""Phase 6 Part B: ask the RAG pipeline a question from the command line.

    python -m src.rag.ask "What supply chain risks does Apple disclose?"
    python -m src.rag.ask "How does Coca-Cola describe water risk?" --ticker KO
    python -m src.rag.ask "What are JPMorgan's capital requirements?" --year 2024
    python -m src.rag.ask "..." --no-context     # baseline arm, no retrieval
    python -m src.rag.ask "..." --json           # machine-readable

Calls the LLM.
"""

from __future__ import annotations

import argparse
import json
import sys

from src import config
from src.rag.chain import LLMCallError, RAGChain

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask a grounded question about the indexed SEC filings."
    )
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=config.TOP_K,
                        help=f"chunks to retrieve (default {config.TOP_K})")
    parser.add_argument("--ticker", nargs="*", help="restrict to these tickers")
    parser.add_argument("--year", nargs="*", type=int, dest="fiscal_year",
                        help="restrict to these fiscal years")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="keep near-duplicate chunks")
    parser.add_argument("--no-context", action="store_true",
                        help="baseline arm: ask the model with no retrieved context")
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()

    filters: dict = {}
    if arguments.ticker:
        filters["ticker"] = [t.upper() for t in arguments.ticker]
    if arguments.fiscal_year:
        filters["fiscal_year"] = arguments.fiscal_year

    chain = RAGChain()
    try:
        result = chain.answer(
            arguments.question,
            k=arguments.k,
            filters=filters or None,
            dedupe=not arguments.no_dedupe,
            use_context=not arguments.no_context,
        )
    except LLMCallError as exc:
        # Exit non-zero so a transport failure is never mistaken for a refusal.
        print(f"LLM CALL FAILED\n\n{exc}", file=sys.stderr)
        raise SystemExit(2)

    if arguments.as_json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print(f"\nQUESTION\n  {result.question}")

    mode = "NO CONTEXT (baseline arm)" if not result.used_context else "RAG"
    print(f"\nMODE\n  {mode}")

    print(f"\nANSWER\n{result.answer}")

    if result.used_context:
        if result.citations:
            print("\nCITATIONS")
            for citation in result.citations:
                print(f"  {citation.render()}")
        else:
            print("\nCITATIONS\n  (none - the model cited no excerpt)")

        if result.invalid_citations:
            print(
                f"\n  WARNING: the model cited {result.invalid_citations}, "
                f"but only {len(result.chunks)} excerpts were supplied.\n"
                "  A citation to an excerpt that does not exist is a fabricated "
                "source."
            )

        print("\nRETRIEVED (all supplied excerpts)")
        for chunk in result.chunks:
            preview = " ".join(chunk["text"].split())[:110]
            cited = any(c.number == chunk["number"] and c.valid for c in result.citations)
            print(
                f"  [{chunk['number']}]{'*' if cited else ' '} {chunk['score']:.4f} "
                f"{chunk['ticker']} FY{chunk['fiscal_year']} chunk {chunk['chunk_index']}"
            )
            print(f"       {preview}...")
        print("  (* = cited in the answer)")

    print("\n" + "-" * 70)
    print(f"model          : {result.model}"
          + ("  (FALLBACK)" if result.fallback_used else ""))
    print(f"prompt version : {result.prompt_version}")
    print(f"latency        : {result.latency_seconds:.2f} s")
    print(f"attempts       : {result.attempts}")
    if result.used_context:
        print(f"retrieval      : max {result.max_score:.4f}, "
              f"mean {result.mean_score:.4f} "
              "(logged only - refusal is not gated on score)")
    print(f"refused        : {result.refused}")


if __name__ == "__main__":
    main()
