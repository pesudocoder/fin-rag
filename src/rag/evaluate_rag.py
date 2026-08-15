"""Phase 7 Step 3-5: run the RAG evaluation and score it.

    python -m src.rag.evaluate_rag              # run (resumes) then score
    python -m src.rag.evaluate_rag --run-only
    python -m src.rag.evaluate_rag --score-only

RESUMABLE. Generation writes results/eval_run.json incrementally after every
question, and a re-run skips questions already present. The free tier allows 15
requests/minute and this evaluation makes ~160 calls, so a partial failure is a
realistic outcome and losing completed work to it would be avoidable waste.

Both arms are answered by config.EXPERIMENT_MODEL, pinned. Model parity is
asserted before scores are written.

Metric strength - stated plainly because it differs by metric:

  FABRICATION (headline) rests on category-defined expected behaviour, fixed
  before any model output existed. It does not depend on generated ground truth.
  This is the strong part of the evaluation.

  GROUNDING and COVERAGE depend on LLM judges, and coverage additionally depends
  on auto-generated ground-truth points. Both are secondary and weaker.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict

from src import config
from src.rag.chain import LLMCallError, RAGChain
from src.rag.eval_judges import (
    CoverageJudge,
    GroundingJudge,
    InferenceJudge,
    PremiseJudge,
)
from src.rag.judge import RefusalJudge

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUESTIONS_FILE = config.PROJECT_ROOT / "data" / "eval" / "questions.json"
RUN_FILE = config.RESULTS_DIR / "eval_run.json"
SCORES_FILE = config.RESULTS_DIR / "eval_scores.json"
SPOTCHECK_FILE = config.RESULTS_DIR / "eval_spotcheck.csv"

ARMS = ("with_context", "no_context")
SPOTCHECK_ROWS = 10


class MixedModelRun(RuntimeError):
    """More than one model answered within a single experiment."""


def _rule(title: str) -> None:
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")


def load_questions() -> list[dict]:
    payload = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    if any(not q.get("expected_behaviour") for q in payload["questions"]):
        raise SystemExit("expected_behaviour is unpopulated. "
                         "Run: python -m src.rag.eval_populate")
    return payload["questions"]


# ---------------------------------------------------------------- generation


def run_generation(questions: list[dict]) -> dict:
    """Both arms for every question, resuming from any partial run."""
    _rule("STEP 3: GENERATION")

    existing = {}
    if RUN_FILE.exists():
        existing = {
            record["id"]: record
            for record in json.loads(RUN_FILE.read_text(encoding="utf-8"))["results"]
        }
        complete = sum(1 for r in existing.values() if all(a in r for a in ARMS))
        print(f"resuming: {complete} of {len(questions)} questions already complete")

    print(f"model  : {config.EXPERIMENT_MODEL} (PINNED, no per-call fallback)")
    print(f"calls  : {len(questions) * 2} generation")
    print(f"pacing : {config.LLM_MIN_REQUEST_INTERVAL}s between calls "
          f"(free tier allows 15/min)\n")

    chain = RAGChain(pinned_model=config.EXPERIMENT_MODEL)
    records = []

    for index, question in enumerate(questions, start=1):
        record = existing.get(question["id"], {})
        if all(arm in record and "answer" in record[arm] for arm in ARMS):
            records.append(record)
            continue

        record = {
            "id": question["id"],
            "category": question["category"],
            "subcategory": question["subcategory"],
            "question": question["question"],
            "target_ticker": question.get("target_ticker"),
            "expected_behaviour": question["expected_behaviour"],
            **{k: record[k] for k in ARMS if k in record},
        }

        for arm in ARMS:
            if arm in record and "answer" in record[arm]:
                continue
            try:
                result = chain.answer(question["question"], use_context=(arm == "with_context"))
                record[arm] = result.to_dict()
            except LLMCallError as exc:
                # Recorded as an explicit error, never as an answer, so a
                # transport failure can never be scored as a refusal.
                record[arm] = {"error": str(exc)[:400]}
                print(f"  [{index:>2}/{len(questions)}] {question['id']:<18} "
                      f"{arm:<13} CALL FAILED")
                continue

        records.append(record)

        rag = record.get("with_context", {})
        base = record.get("no_context", {})
        print(f"  [{index:>2}/{len(questions)}] {question['id']:<18} "
              f"RAG {len(rag.get('answer', '')):>5}ch | "
              f"base {len(base.get('answer', '')):>5}ch | "
              f"max {rag.get('retrieval', {}).get('max_score', '-')}")

        _write_run(records + [existing[q["id"]] for q in questions[index:]
                              if q["id"] in existing])

    _write_run(records)
    return {"results": records}


def _write_run(records: list[dict]) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    seen, unique = set(), []
    for record in records:
        if record["id"] not in seen:
            seen.add(record["id"])
            unique.append(record)
    RUN_FILE.write_text(json.dumps({
        "phase": "7-eval-run",
        "pinned_model": config.EXPERIMENT_MODEL,
        "prompt_version": config.RAG_PROMPT_VERSION,
        "k": config.TOP_K,
        "results": unique,
    }, indent=2), encoding="utf-8")


# ------------------------------------------------------------------- scoring


def check_model_parity(records: list[dict]) -> list[str]:
    models = sorted({
        record[arm]["model"]
        for record in records for arm in ARMS
        if "model" in record.get(arm, {})
    })
    if len(models) != 1 or models[0] != config.EXPERIMENT_MODEL:
        raise MixedModelRun(
            f"Pinned {config.EXPERIMENT_MODEL} but calls were answered by {models}. "
            "Results are confounded and were NOT scored."
        )
    return models


def score(questions: list[dict], records: list[dict]) -> dict:
    _rule("STEP 4: SCORING")
    by_id = {q["id"]: q for q in questions}

    models = check_model_parity(records)
    print(f"model parity: PASS (all calls answered by {models[0]})\n")

    refusal_judge = RefusalJudge()
    premise_judge = PremiseJudge()
    inference_judge = InferenceJudge()
    grounding_judge = GroundingJudge()
    coverage_judge = CoverageJudge()

    for index, record in enumerate(records, start=1):
        question = by_id[record["id"]]
        for arm in ARMS:
            entry = record.get(arm, {})
            if "answer" not in entry:
                continue
            answer = entry["answer"]

            # (a) refusal - validated judge, primary axis
            assessment = refusal_judge.classify(record["question"], answer)
            entry["refusal_assessment"] = assessment.to_dict()
            refused = assessment.verdict == "REFUSED"
            entry["binary_refused"] = refused    # PARTIAL collapses to ANSWERED

            # (b) fabrication, per category rules
            subcategory = record["subcategory"]
            if subcategory == "false-premise":
                verdict = premise_judge.classify(
                    record["question"], question.get("false_premise", ""), answer
                )
                entry["premise_assessment"] = verdict.to_dict()
                entry["fabricated"] = verdict.verdict == "PLAYS_ALONG"
                entry["fabrication_basis"] = "premise judge"
            elif subcategory == "requires-inference-beyond-text":
                verdict = inference_judge.classify(record["question"], answer)
                entry["inference_assessment"] = verdict.to_dict()
                entry["fabricated"] = verdict.verdict == "CONFIDENT_PROJECTION"
                entry["fabrication_basis"] = "inference judge"
            elif record["category"] in ("absent", "adversarial"):
                entry["fabricated"] = not refused
                entry["fabrication_basis"] = "not a refusal"
            else:
                entry["fabricated"] = None
                entry["fabrication_basis"] = "not applicable (answerable)"

            # (c) grounding + wrong-company citations - RAG arm, answerable only
            if record["category"] == "answerable" and arm == "with_context":
                chunks = {c["number"]: c for c in entry["retrieval"]["chunks"]}
                cited = [
                    chunks[c["number"]] for c in entry["citations"]
                    if c["valid"] and c["number"] in chunks
                ]
                verdict = grounding_judge.classify(record["question"], answer, cited)
                entry["grounding_assessment"] = verdict.to_dict()

                target = record["target_ticker"]
                wrong = [
                    {"number": c["number"], "ticker": c["ticker"]}
                    for c in cited if c["ticker"] != target
                ]
                entry["wrong_company_citations"] = wrong
                entry["cites_wrong_company"] = bool(wrong)

            # (d) coverage - answerable, both arms
            if record["category"] == "answerable":
                truth = question.get("ground_truth") or {}
                points = [p["point"] for p in truth.get("points", [])]
                verdict = coverage_judge.classify(record["question"], answer, points)
                entry["coverage_assessment"] = verdict.to_dict()

        print(f"  [{index:>2}/{len(records)}] scored {record['id']}")

    _write_run(records)
    return aggregate(questions, records)


def aggregate(questions: list[dict], records: list[dict]) -> dict:
    by_arm: dict = {arm: defaultdict(list) for arm in ARMS}

    for record in records:
        for arm in ARMS:
            entry = record.get(arm, {})
            if "answer" not in entry:
                by_arm[arm]["failed"].append(record["id"])
                continue
            by_arm[arm]["all"].append((record, entry))
            by_arm[arm][record["category"]].append((record, entry))

    summary: dict = {"arms": {}}

    for arm in ARMS:
        rows = by_arm[arm]["all"]
        fabrication_pool = [
            (r, e) for r, e in rows if r["category"] in ("absent", "adversarial")
        ]
        fabricated = [(r, e) for r, e in fabrication_pool if e.get("fabricated")]
        answerable = [(r, e) for r, e in rows if r["category"] == "answerable"]

        per_category = {}
        for category in ("answerable", "absent", "adversarial"):
            subset = by_arm[arm][category]
            refused = sum(1 for _, e in subset if e.get("binary_refused"))
            fab = sum(1 for _, e in subset if e.get("fabricated"))
            per_category[category] = {
                "n": len(subset),
                "refused": refused,
                "refusal_rate": round(refused / len(subset), 4) if subset else None,
                "fabricated": fab,
            }

        grounding = Counter(
            e["grounding_assessment"]["verdict"]
            for _, e in answerable if "grounding_assessment" in e
        )
        wrong_company = [r["id"] for r, e in answerable if e.get("cites_wrong_company")]

        covered = sum(
            e["coverage_assessment"].get("covered_count", 0)
            for _, e in answerable if "coverage_assessment" in e
        )
        total_points = sum(
            e["coverage_assessment"].get("point_count", 0)
            for _, e in answerable if "coverage_assessment" in e
        )

        summary["arms"][arm] = {
            "answered_calls": len(rows),
            "failed_calls": len(by_arm[arm]["failed"]),
            "failed_ids": by_arm[arm]["failed"],
            "refusal": {
                "refused": sum(1 for _, e in rows if e.get("binary_refused")),
                "n": len(rows),
                "per_category": per_category,
            },
            "fabrication": {
                "count": len(fabricated),
                "n": len(fabrication_pool),
                "rate": round(len(fabricated) / len(fabrication_pool), 4)
                if fabrication_pool else None,
                "fabricated_ids": [r["id"] for r, _ in fabricated],
            },
            "grounding": {
                "n": len(answerable),
                "verdicts": dict(grounding),
                "cites_wrong_company": len(wrong_company),
                "wrong_company_ids": wrong_company,
            },
            "coverage": {
                "points_covered": covered,
                "points_total": total_points,
                "rate": round(covered / total_points, 4) if total_points else None,
            },
            "invalid_citations": sum(
                len(e.get("invalid_citations", [])) for _, e in rows
            ),
        }

    rag = summary["arms"]["with_context"]["fabrication"]
    base = summary["arms"]["no_context"]["fabrication"]
    summary["headline"] = {
        "fabrication_with_context": rag["count"],
        "fabrication_without_context": base["count"],
        "fabrication_pool": rag["n"],
        "delta": base["count"] - rag["count"],
        "statement": (
            f"Fabrication on the {rag['n']} absent + adversarial questions: "
            f"{rag['count']}/{rag['n']} with retrieval vs "
            f"{base['count']}/{base['n']} without."
        ),
    }
    return summary


# ------------------------------------------------------------------ reporting


def print_summary(summary: dict, records: list[dict]) -> None:
    _rule("SUMMARY BY CATEGORY AND ARM")
    header = (f"{'category':<14} {'arm':<14} {'n':>3} {'refused':>8} "
              f"{'refusal rate':>13} {'fabricated':>11}")
    print(header)
    print("-" * len(header))
    for category in ("answerable", "absent", "adversarial"):
        for arm in ARMS:
            stats = summary["arms"][arm]["refusal"]["per_category"][category]
            rate = f"{stats['refusal_rate']:.4f}" if stats["refusal_rate"] is not None else "-"
            fab = "-" if category == "answerable" else stats["fabricated"]
            print(f"{category:<14} {arm:<14} {stats['n']:>3} {stats['refused']:>8} "
                  f"{rate:>13} {str(fab):>11}")
        print()

    _rule("GROUNDING (answerable, RAG arm only - no citations exist without context)")
    grounding = summary["arms"]["with_context"]["grounding"]
    for verdict, count in sorted(grounding["verdicts"].items()):
        print(f"  {verdict:<20} {count}")
    print(f"\n  cites wrong company: {grounding['cites_wrong_company']} "
          f"{grounding['wrong_company_ids'] or ''}")

    _rule("COVERAGE (answerable, auto-generated ground truth - SECONDARY METRIC)")
    for arm in ARMS:
        cov = summary["arms"][arm]["coverage"]
        rate = f"{cov['rate']:.4f}" if cov["rate"] is not None else "-"
        print(f"  {arm:<14} {cov['points_covered']:>3}/{cov['points_total']:<3} "
              f"points  ({rate})")

    _rule("INVALID CITATIONS")
    for arm in ARMS:
        print(f"  {arm:<14} {summary['arms'][arm]['invalid_citations']}")

    _rule("HEADLINE")
    headline = summary["headline"]
    print(headline["statement"])
    print(f"\n  fabrication delta: {headline['delta']:+d} "
          f"({headline['fabrication_without_context']} without retrieval -> "
          f"{headline['fabrication_with_context']} with retrieval)")
    print("\nThis metric rests on category-defined expected behaviour fixed before")
    print("any model output existed. It does not depend on generated ground truth.")


def write_spotcheck(records: list[dict]) -> int:
    """A BLIND sample for hand-validating the scoring, weighted to fabrication."""
    # Stratified, weighted to the fabrication judgements that carry the headline
    # number, but not drawn entirely from one category - a sample that is all
    # 'absent' rows cannot validate the table-figure or grounding verdicts.
    quota = {"absent": 4, "adversarial": 4, "answerable": 2}

    pools: dict[str, list] = {category: [] for category in quota}
    for record in records:
        for arm in ARMS:
            entry = record.get(arm, {})
            if "answer" in entry:
                pools[record["category"]].append((record, arm, entry))

    chosen = []
    for category, wanted in quota.items():
        # Split by arm and draw from both. The headline number is a comparison
        # BETWEEN arms, so a sample drawn from only one cannot validate it.
        per_arm = {
            arm: sorted(
                (item for item in pools[category] if item[1] == arm),
                key=lambda item: item[0]["id"],
            )
            for arm in ARMS
        }
        want_rag = (wanted + 1) // 2
        for arm, count in (("with_context", want_rag), ("no_context", wanted - want_rag)):
            pool = per_arm[arm]
            if not pool or count == 0:
                continue
            step = max(1, len(pool) // count)
            chosen.extend(pool[::step][:count])

    SPOTCHECK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SPOTCHECK_FILE, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "arm", "category", "question", "answer",
                        "human_refused", "human_fabricated"],
        )
        writer.writeheader()

        def note(text, extra=""):
            writer.writerow({"id": "#", "arm": "", "category": "",
                             "question": text, "answer": extra,
                             "human_refused": "", "human_fabricated": ""})

        note("BLIND SPOT-CHECK - judge verdicts are deliberately not shown.")
        note("human_refused", "YES if the response declines to provide the "
                              "information (including declining by asking you to "
                              "supply documents); otherwise NO.")
        note("human_fabricated", "YES if the response asserts content the corpus "
                                 "cannot support - a figure, a non-existent event, "
                                 "or a projection stated as fact; otherwise NO.")
        note("Rows beginning with '#' are ignored. Edit only the two human_ columns.")

        for record, arm, entry in chosen:
            writer.writerow({
                "id": f"{record['id']}::{arm}",
                "arm": "RAG" if arm == "with_context" else "no-context",
                "category": record["category"],
                "question": record["question"],
                "answer": entry["answer"],
                "human_refused": "",
                "human_fabricated": "",
            })

    return len(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 RAG evaluation.")
    parser.add_argument("--run-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    arguments = parser.parse_args()

    questions = load_questions()

    if arguments.score_only:
        if not RUN_FILE.exists():
            raise SystemExit(f"No {RUN_FILE}. Run generation first.")
        run = json.loads(RUN_FILE.read_text(encoding="utf-8"))
    else:
        run = run_generation(questions)

    failed = [
        f"{r['id']}::{arm}" for r in run["results"] for arm in ARMS
        if "answer" not in r.get(arm, {})
    ]
    if failed:
        print(f"\n{len(failed)} call(s) failed: {failed}")
        print("Re-run to retry only the failures (generation is resumable).")

    if arguments.run_only:
        print(f"\nwrote {RUN_FILE}")
        return

    summary = score(questions, run["results"])
    print_summary(summary, run["results"])

    rows = write_spotcheck(run["results"])

    _rule("STEP 5: OUTPUTS")
    payload = {
        "phase": "7-eval-scores",
        "pinned_model": config.EXPERIMENT_MODEL,
        "model_parity_verified": True,
        "prompt_version": config.RAG_PROMPT_VERSION,
        "judge_model": config.JUDGE_MODEL,
        "question_count": len(questions),
        "metric_strength": {
            "fabrication": (
                "STRONG. Rests on category-defined expected behaviour fixed before "
                "any model output existed. Does not depend on generated ground "
                "truth. This is the headline metric."
            ),
            "refusal": (
                "MODERATE. Uses the refusal judge validated against 12 hand labels "
                "in Phase 6 (binary agreement 12/12, kappa 1.0). That validation "
                "was on Phase 6 answers, not these."
            ),
            "grounding": (
                "WEAK. LLM judge, not human-validated. See eval_spotcheck.csv."
            ),
            "coverage": (
                "WEAKEST. Depends on auto-generated ground-truth points AND an "
                "unvalidated LLM judge. Secondary metric only. One question "
                "(ANS-KO-02) produced zero ground-truth points despite relevant "
                "material being retrieved, which is itself evidence of the "
                "weakness."
            ),
        },
        "not_yet_validated": (
            "The Phase 7 scoring judges have NOT been validated against human "
            "labels. results/eval_spotcheck.csv is a blind 10-row sample provided "
            "for that purpose and has not been labelled."
        ),
        **summary,
    }
    SCORES_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"{RUN_FILE}")
    print(f"{SCORES_FILE}")
    print(f"{SPOTCHECK_FILE}  ({rows} blind rows, weighted to fabrication cases)")


if __name__ == "__main__":
    main()
