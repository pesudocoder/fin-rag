"""Phase 6 Part C: qualitative smoke test of the RAG chain.

    python -m src.rag.smoke_test

Runs 6 questions through both arms - with retrieved context and without - and
prints them side by side. 12 LLM calls total.

THIS IS INSPECTION, NOT MEASUREMENT. Six questions, one run, no rubric, no second
rater. Nothing here is a hallucination rate. It exists to see whether the chain
behaves sanely before Phase 7 spends effort scoring it properly, and to surface
obvious failures early. Do not quote any number from this file as a result.

Question design follows two limitations established earlier:

  * All corpus-answerable questions are scoped to a SINGLE company. Phase 5
    measured mean unique companies in the top-5 at 1.625, and cross-company
    probes collapsed onto one filer (the FX probe returned 5 Coca-Cola chunks),
    so a comparative question would fail at retrieval and test the wrong thing.

  * One question deliberately targets a figure from a financial TABLE. Phase 4
    established that table structure is destroyed during HTML extraction, so
    numbers reach the model stripped of their row and column labels. The
    question tests whether the model invents a plausible-looking figure from
    decontextualised digits.
"""

from __future__ import annotations

import json
import sys
import textwrap

from src import config
from src.rag.chain import LLMCallError, RAGChain
from src.rag.judge import RefusalJudge

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class MixedModelRun(RuntimeError):
    """More than one model answered within a single experiment."""

QUESTIONS = [
    {
        "id": "aapl_supply_chain",
        "category": "answerable-single-company",
        "question": "What risks does Apple disclose about relying on outsourcing partners for manufacturing?",
        "expectation": "Should answer from Apple's risk factors with citations.",
    },
    {
        "id": "ko_water",
        "category": "answerable-single-company",
        "question": "How does Coca-Cola describe the risk of water scarcity to its business?",
        "expectation": "Should answer from Coca-Cola's risk factors with citations.",
    },
    {
        "id": "msft_ai",
        "category": "answerable-single-company",
        "question": "What risks does Microsoft disclose about developing and deploying artificial intelligence?",
        "expectation": "Should answer from Microsoft's risk factors with citations.",
    },
    {
        "id": "absent_aircraft",
        "category": "absent-from-corpus",
        "question": "What do these filings disclose about aircraft fleet fuel hedging and aircraft lease obligations?",
        "expectation": (
            "Should refuse. No airline filings are in the corpus. Phase 5 measured "
            "this query's top retrieval score at 0.4630 - high enough to look "
            "plausible, because JPMorgan discusses leases and hedging, just not "
            "aircraft ones. This is the hardest refusal case in the corpus."
        ),
    },
    {
        "id": "adversarial_table_figure",
        "category": "adversarial-table-figure",
        "question": "What was JPMorgan's exact total allowance for lending-related commitments at the end of 2023, in millions of dollars?",
        "expectation": (
            "Table structure is destroyed in extraction, so the retrieved text "
            "contains bare numbers without row or period labels. The model should "
            "say it cannot determine the figure. Producing a confident number "
            "would be a hallucination."
        ),
    },
    {
        "id": "out_of_corpus_company",
        "category": "company-not-in-corpus",
        "question": "What risks does Tesla disclose about battery supply and raw material sourcing?",
        "expectation": (
            "Tesla is not in the corpus, but the model certainly has pretraining "
            "knowledge of Tesla's risk factors. The sharpest test of whether it "
            "stays grounded or falls back on what it already knows."
        ),
    },
]


def _rule(title: str) -> None:
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")


def _wrap(text: str, indent: str = "    ") -> str:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return "\n".join(
        textwrap.fill(p, width=72, initial_indent=indent, subsequent_indent=indent)
        for p in paragraphs
    )


def main() -> None:
    _rule("PHASE 6 SMOKE TEST - QUALITATIVE INSPECTION ONLY")
    print("This is NOT measurement. Six questions, one run, no rubric, no second")
    print("rater, no scoring. It shows whether the chain behaves sanely before")
    print("Phase 7 evaluates it properly. Do not quote anything here as a result;")
    print("in particular this produces NO hallucination rate.")
    print(f"\nmodel: {config.EXPERIMENT_MODEL} (PINNED - no per-call fallback)")
    print("  Pinned so both arms are answered by the same model. Per-call")
    print("  fallback previously gave 3 of 6 pairs different models per arm,")
    print("  turning a context comparison into a partial model comparison.")
    print(f"  NOTE: {config.EXPERIMENT_MODEL} ignores temperature, so output is")
    print("  not deterministic. See config.EXPERIMENT_MODEL for the trade-off.")
    print(f"\nprompt version: {config.RAG_PROMPT_VERSION}")
    print(f"judge: {config.JUDGE_MODEL} (prompt {config.JUDGE_PROMPT_VERSION})")
    print(f"questions: {len(QUESTIONS)}, generation calls: {len(QUESTIONS) * 2}, "
          f"judge calls: {len(QUESTIONS) * 2}")

    chain = RAGChain(pinned_model=config.EXPERIMENT_MODEL)
    judge = RefusalJudge()
    records = []

    for index, item in enumerate(QUESTIONS, start=1):
        _rule(f"{index}/{len(QUESTIONS)}  [{item['category']}]  {item['id']}")
        print(f"Q: {item['question']}")
        print(f"\nexpectation:\n{_wrap(item['expectation'], '  ')}")

        record = {**item}

        for arm, use_context in (("with_context", True), ("no_context", False)):
            try:
                result = chain.answer(item["question"], use_context=use_context)
                record[arm] = result.to_dict()
            except LLMCallError as exc:
                # Recorded as an explicit error, never as an answer string, so a
                # transport failure cannot be counted as a refusal later.
                record[arm] = {"error": str(exc)}
                print(f"\n--- {arm.upper()} ---\n  CALL FAILED: {exc}")
                continue

            assessment = judge.classify(item["question"], result.answer)
            record[arm]["refusal_assessment"] = assessment.to_dict()

            label = "WITH CONTEXT (RAG)" if use_context else "NO CONTEXT (baseline)"
            print(f"\n--- {label} ---")
            print(_wrap(result.answer))

            if use_context:
                valid = [c for c in result.citations if c.valid]
                cited = ", ".join(
                    f"[{c.number}] {c.metadata['ticker']} FY{c.metadata['fiscal_year']}"
                    for c in valid
                ) or "none"
                print(f"\n    cited: {cited}")
                if result.invalid_citations:
                    print(f"    INVALID CITATIONS: {result.invalid_citations} "
                          f"(only {len(result.chunks)} excerpts supplied)")
                print(f"    retrieval: max {result.max_score:.4f}, "
                      f"mean {result.mean_score:.4f}")
                tickers = sorted({c["ticker"] for c in result.chunks})
                print(f"    chunks from: {', '.join(tickers)}")
            flag = "" if assessment.signals_agree else "   <-- SIGNALS DISAGREE"
            print(f"    judge: {assessment.verdict} | keyword_refused: "
                  f"{assessment.keyword_refused}{flag}")
            print(f"      why: {assessment.justification}")
            print(f"    model: {result.model} | {result.latency_seconds:.1f}s")

        records.append(record)

    _rule("SUMMARY")
    header = f"{'id':<28} {'RAG judge':<10} {'base judge':<11} {'signals agree'}"
    print(header)
    print("-" * len(header))
    for record in records:
        rag = record.get("with_context", {}).get("refusal_assessment", {})
        base = record.get("no_context", {}).get("refusal_assessment", {})
        agree = rag.get("signals_agree", True) and base.get("signals_agree", True)
        print(
            f"{record['id']:<28} {rag.get('verdict', 'ERROR'):<10} "
            f"{base.get('verdict', 'ERROR'):<11} {agree}"
        )

    invalid_total = sum(
        len(record.get("with_context", {}).get("invalid_citations", []))
        for record in records
    )

    assessments = [
        record[arm]["refusal_assessment"]
        for record in records
        for arm in ("with_context", "no_context")
        if "refusal_assessment" in record.get(arm, {})
    ]
    disagreements = [a for a in assessments if not a["signals_agree"]]

    print(f"\ninvalid citations across all RAG answers: {invalid_total}")
    print(f"judge vs keyword disagreements: {len(disagreements)} of {len(assessments)}")
    for item in disagreements:
        print(f"  judge={item['verdict']}, keyword_refused={item['keyword_refused']}"
              f"  -- {item['justification'][:90]}")

    # Model parity: a silently mixed run is worse than a failed one, because the
    # mixing is invisible in the output and confounds every comparison drawn
    # from it.
    models_used = sorted({
        record[arm]["model"]
        for record in records
        for arm in ("with_context", "no_context")
        if "model" in record.get(arm, {})
    })
    print(f"\nmodels used across all calls: {models_used}")
    if len(models_used) != 1 or models_used[0] != config.EXPERIMENT_MODEL:
        raise MixedModelRun(
            f"Experiment pinned {config.EXPERIMENT_MODEL} but calls were answered "
            f"by {models_used}. Results are confounded and were NOT saved."
        )
    print(f"model parity check: PASS (all calls answered by {models_used[0]})")

    print("\nReminder: qualitative inspection. Scored evaluation is Phase 7.")

    _rule("SAVING")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase": "6-rag-smoke-test",
        "purpose": (
            "Qualitative inspection only - NOT measurement. No hallucination rate "
            "is produced here. Scored evaluation is Phase 7."
        ),
        "pinned_model": config.EXPERIMENT_MODEL,
        "models_used": models_used,
        "model_parity_verified": True,
        "temperature_note": (
            f"{config.EXPERIMENT_MODEL} ignores temperature (fixed sampling "
            "defaults), so LLM_TEMPERATURE has no effect and output is not "
            "deterministic. Pinned anyway: mixed-model comparison is a systematic "
            "confound, sampling variance is only noise."
        ),
        "judge": {
            "model": config.JUDGE_MODEL,
            "prompt_version": config.JUDGE_PROMPT_VERSION,
            "system_prompt": config.JUDGE_SYSTEM_PROMPT,
            "limitation": (
                "LLM-as-judge is NOT an independent measurement: Gemini grading "
                "Gemini shares training data and failure modes, so errors "
                "correlate with what is being measured. Validate with human "
                "labels (python -m src.rag.judge --template ...) before quoting "
                "any judged number."
            ),
            "keyword_disagreements": len(disagreements),
            "assessments_total": len(assessments),
        },
        "temperature": config.LLM_TEMPERATURE,
        "prompt_version": config.RAG_PROMPT_VERSION,
        "rag_system_prompt": config.RAG_SYSTEM_PROMPT,
        "no_context_system_prompt": config.NO_CONTEXT_SYSTEM_PROMPT,
        "k": config.TOP_K,
        "question_count": len(QUESTIONS),
        "llm_calls": len(QUESTIONS) * 2,
        "invalid_citation_total": invalid_total,
        "results": records,
    }
    path = config.RESULTS_DIR / "rag_smoke_test.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{path}")


if __name__ == "__main__":
    main()
