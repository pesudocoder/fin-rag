"""Phase 7 Part 1: validate the evaluation question set and emit the review doc.

    python -m src.rag.build_eval_set

Reads data/eval/questions.json, checks its structure and counts, runs every
question through the RETRIEVER ONLY, and writes data/eval/questions_review.md.

NO LLM CALLS. Retrieval uses the local sentence-transformers encoder and the
FAISS index; nothing here contacts Gemini. The point is to verify the premises
of the question set - especially that the ABSENT questions really are absent -
before any generation happens.
"""

from __future__ import annotations

import json
import sys
from collections import Counter

from src import config
from src.rag.retrieve import Retriever

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUESTIONS_FILE = config.PROJECT_ROOT / "data" / "eval" / "questions.json"
REVIEW_FILE = config.PROJECT_ROOT / "data" / "eval" / "questions_review.md"

EXPECTED_COUNTS = {"answerable": 12, "absent": 8, "adversarial": 8}

# Phase 5 established these reference points on this corpus.
RANDOM_PAIR_FLOOR = 0.2819        # mean cosine of random chunk pairs
HIGHEST_OUT_OF_CORPUS = 0.4630    # aircraft probe
LOWEST_LEGITIMATE_TOP1 = 0.5750   # KO water scarcity probe


def _rule(title: str) -> None:
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")


def load_questions() -> dict:
    if not QUESTIONS_FILE.exists():
        raise SystemExit(f"Missing {QUESTIONS_FILE}")
    return json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))


def validate(payload: dict) -> list[str]:
    """Structural checks. Returns a list of problems, empty if clean."""
    problems = []
    questions = payload["questions"]

    counts = Counter(q["category"] for q in questions)
    for category, expected in EXPECTED_COUNTS.items():
        if counts[category] != expected:
            problems.append(
                f"{category}: expected {expected}, found {counts[category]}"
            )

    ids = [q["id"] for q in questions]
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    if duplicates:
        problems.append(f"duplicate ids: {duplicates}")

    texts = [q["question"].strip().lower() for q in questions]
    dupe_text = [t for t, n in Counter(texts).items() if n > 1]
    if dupe_text:
        problems.append(f"duplicate question text: {dupe_text}")

    for question in questions:
        for field in ("id", "category", "subcategory", "question",
                      "ground_truth", "expected_behaviour"):
            if field not in question:
                problems.append(f"{question.get('id', '?')}: missing {field}")
        if "target_ticker" not in question:
            problems.append(f"{question['id']}: missing target_ticker")

    # Answerable questions must name a single company.
    for question in questions:
        if question["category"] == "answerable" and not question.get("target_ticker"):
            problems.append(f"{question['id']}: answerable but no target_ticker")

    # Absent questions must justify their premise.
    for question in questions:
        if question["category"] == "absent" and not question.get("absence_rationale"):
            problems.append(f"{question['id']}: absent but no absence_rationale")

    per_company = Counter(
        q["target_ticker"] for q in questions if q["category"] == "answerable"
    )
    for ticker in ("AAPL", "JPM", "KO", "MSFT"):
        if per_company[ticker] != 3:
            problems.append(
                f"answerable/{ticker}: expected 3, found {per_company[ticker]}"
            )

    return problems


def band(score: float) -> str:
    """Where a score sits relative to the Phase 5 reference points."""
    if score < RANDOM_PAIR_FLOOR:
        return "below random floor"
    if score <= HIGHEST_OUT_OF_CORPUS:
        return "low"
    if score < LOWEST_LEGITIMATE_TOP1:
        return "AMBIGUOUS BAND"
    return "high"


def probe(retriever: Retriever, payload: dict) -> list[dict]:
    """Retrieval-only pass over every question. No LLM."""
    rows = []
    for question in payload["questions"]:
        report = retriever.search(question["question"], k=config.TOP_K)
        scores = [r.score for r in report.results]
        tickers = [r.metadata["ticker"] for r in report.results]
        rows.append(
            {
                "id": question["id"],
                "category": question["category"],
                "subcategory": question["subcategory"],
                "target_ticker": question.get("target_ticker"),
                "max_score": round(max(scores), 4) if scores else None,
                "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
                "band": band(max(scores)) if scores else "n/a",
                "tickers": tickers,
                "top_ticker": tickers[0] if tickers else None,
                "target_in_topk": (
                    question.get("target_ticker") in tickers
                    if question.get("target_ticker") else None
                ),
            }
        )
    return rows


def write_review(payload: dict, rows: list[dict]) -> None:
    by_id = {row["id"]: row for row in rows}
    lines: list[str] = []
    add = lines.append

    add("# Evaluation question set - for review")
    add("")
    add("Generated by `python -m src.rag.build_eval_set`. Edit "
        "`data/eval/questions.json`, not this file, then regenerate.")
    add("")
    add("**Fill in `ground_truth` and `expected_behaviour` in the JSON before "
        "running any evaluation.** Both are empty by design: the question set "
        "is written first so that the scoring standard is not fitted to what "
        "the system happens to produce.")
    add("")
    add(f"Corpus: {payload['corpus']}")
    add("")

    add("## Design constraints")
    add("")
    for key, text in payload["design_constraints"].items():
        add(f"- **{key}** - {text}")
    add("")

    add("## Retrieval reference points (Phase 5, this corpus)")
    add("")
    add(f"- random chunk-pair mean cosine (noise floor): **{RANDOM_PAIR_FLOOR}**")
    add(f"- highest out-of-corpus probe score: **{HIGHEST_OUT_OF_CORPUS}**")
    add(f"- lowest legitimate top-1 score: **{LOWEST_LEGITIMATE_TOP1}**")
    add("")
    add(f"Scores between {HIGHEST_OUT_OF_CORPUS} and {LOWEST_LEGITIMATE_TOP1} are "
        "the **ambiguous band** - the range where no score threshold can separate "
        "answerable from unanswerable. Questions landing there are the ones that "
        "make semantic refusal necessary.")
    add("")

    for category, heading, blurb in [
        ("answerable", "Answerable (12)",
         "The control arm. One company each, qualitative risk-factor content, no "
         "table figures."),
        ("absent", "Absent (8)",
         "The answer genuinely is not in the corpus. Each records why, so the "
         "premise can be checked rather than trusted."),
        ("adversarial", "Adversarial (8)",
         "Designed to elicit fabrication."),
    ]:
        add(f"## {heading}")
        add("")
        add(blurb)
        add("")
        for question in payload["questions"]:
            if question["category"] != category:
                continue
            row = by_id[question["id"]]
            add(f"### {question['id']}  ({question['subcategory']})")
            add("")
            add(f"> {question['question']}")
            add("")
            add(f"- **target ticker**: {question.get('target_ticker') or 'none'}")
            if question.get("phrasing"):
                add(f"- **phrasing**: {question['phrasing']}")
            add(f"- **rationale**: {question.get('rationale', '')}")
            if question.get("absence_rationale"):
                add(f"- **why absent**: {question['absence_rationale']}")
            if question.get("adjacent_concept_in_corpus"):
                add(f"- **adjacent concept present in corpus**: "
                    f"{question['adjacent_concept_in_corpus']}")
            if question.get("false_premise"):
                add(f"- **false premise**: {question['false_premise']}")
            add(f"- **retrieval**: max {row['max_score']}, mean {row['mean_score']} "
                f"-> _{row['band']}_")
            add(f"- **top-5 tickers**: {', '.join(row['tickers'])}")
            add("- **ground_truth**: _(to be written)_")
            add("- **expected_behaviour**: _(to be written)_")
            add("")

    REVIEW_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _rule("EVALUATION QUESTION SET - VALIDATION (retrieval only, no LLM)")
    payload = load_questions()

    problems = validate(payload)
    counts = Counter(q["category"] for q in payload["questions"])
    print(f"questions: {len(payload['questions'])}")
    for category, expected in EXPECTED_COUNTS.items():
        print(f"  {category:<12} {counts[category]:>2} (expected {expected})")

    if problems:
        print("\nPROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit("Question set is not valid.")
    print("\nstructure: PASS")

    retriever = Retriever(verbose=True)
    rows = probe(retriever, payload)

    _rule("ABSENT QUESTIONS - PREMISE CHECK")
    print("If any of these scores high, the premise may be wrong and the question")
    print("may in fact be answerable from the corpus. Review before evaluating.\n")
    print(f"reference: random floor {RANDOM_PAIR_FLOOR} | "
          f"highest out-of-corpus {HIGHEST_OUT_OF_CORPUS} | "
          f"lowest legitimate top-1 {LOWEST_LEGITIMATE_TOP1}\n")

    header = f"{'id':<18} {'max':<8} {'mean':<8} {'band':<18} {'top-5 tickers'}"
    print(header)
    print("-" * len(header))
    for row in rows:
        if row["category"] != "absent":
            continue
        print(f"{row['id']:<18} {row['max_score']:<8} {row['mean_score']:<8} "
              f"{row['band']:<18} {' '.join(row['tickers'])}")

    suspicious = [
        row for row in rows
        if row["category"] == "absent" and row["max_score"] >= LOWEST_LEGITIMATE_TOP1
    ]
    if suspicious:
        print(f"\nWARNING: {len(suspicious)} absent question(s) score at or above the")
        print("lowest legitimate top-1 score. Verify the premise by hand:")
        for row in suspicious:
            print(f"  - {row['id']} (max {row['max_score']})")
    else:
        print("\nNo absent question scores above the lowest legitimate top-1 "
              f"({LOWEST_LEGITIMATE_TOP1}).")

    _rule("ANSWERABLE QUESTIONS - RETRIEVAL SANITY")
    print("Does retrieval surface the intended company at all? Retrieval quality,")
    print("not answer quality.\n")
    header = f"{'id':<16} {'target':<8} {'max':<8} {'in top-5':<10} {'top-5 tickers'}"
    print(header)
    print("-" * len(header))
    for row in rows:
        if row["category"] != "answerable":
            continue
        print(f"{row['id']:<16} {row['target_ticker']:<8} {row['max_score']:<8} "
              f"{str(row['target_in_topk']):<10} {' '.join(row['tickers'])}")

    misses = [
        row for row in rows
        if row["category"] == "answerable" and not row["target_in_topk"]
    ]
    if misses:
        print(f"\nWARNING: {len(misses)} answerable question(s) do not retrieve the")
        print("target company at all. These would fail for retrieval reasons:")
        for row in misses:
            print(f"  - {row['id']} (target {row['target_ticker']}, "
                  f"got {' '.join(row['tickers'])})")
    else:
        print("\nAll answerable questions retrieve their target company in the top-5.")

    _rule("ADVERSARIAL QUESTIONS - RETRIEVAL")
    header = f"{'id':<20} {'max':<8} {'band':<18} {'top-5 tickers'}"
    print(header)
    print("-" * len(header))
    for row in rows:
        if row["category"] != "adversarial":
            continue
        print(f"{row['id']:<20} {row['max_score']:<8} {row['band']:<18} "
              f"{' '.join(row['tickers'])}")

    _rule("BAND DISTRIBUTION")
    for category in EXPECTED_COUNTS:
        bands = Counter(r["band"] for r in rows if r["category"] == category)
        print(f"{category:<12} {dict(bands)}")

    write_review(payload, rows)
    _rule("WRITING")
    print(f"{REVIEW_FILE}")
    print("\nNext: fill in ground_truth and expected_behaviour in")
    print(f"{QUESTIONS_FILE}")
    print("No LLM calls were made.")


if __name__ == "__main__":
    main()
