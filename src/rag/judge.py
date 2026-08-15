"""LLM-as-judge for refusal classification, plus human-agreement validation.

Classifies an assistant response as REFUSED / ANSWERED / PARTIAL.

    from src.rag.judge import RefusalJudge
    verdict = RefusalJudge().classify(question, answer)

CLI:

    python -m src.rag.judge --template results/rag_smoke_test.json labels.csv
    # open labels.csv, fill in human_verdict on every row
    python -m src.rag.judge --agreement labels.csv

The template is BLIND: it carries the question, the arm and the answer, but not
the judge's verdict or justification. A rater who can see the judge's answer
anchors to it, and agreement between an anchored rater and the source of the
anchor measures nothing. Verdicts are rejoined from the results file when
agreement is computed.

Why this replaces keyword matching
----------------------------------
Refusal was originally detected with a substring list. It missed

    "Based on the provided context, there are no disclosures regarding aircraft
     fleet fuel hedging or aircraft lease obligations."

which is unambiguously a refusal, and scored it as an answer. Widening the list
only defers the problem to the next phrasing nobody anticipated - and each miss
biases the measurement in the same direction, understating refusals and
therefore overstating the answer rate in exactly the comparison Phase 7 exists
to make.

The keyword detector is retained as a cheap secondary signal. Where it and the
judge disagree, the disagreement is recorded; that rate is itself a reported
number, since a keyword detector that agrees with the judge everywhere would
have made the judge unnecessary.

METHODOLOGICAL LIMITATION - READ BEFORE QUOTING ANY NUMBER FROM THIS
--------------------------------------------------------------------
**An LLM grading another LLM's output is not an independent measurement.** This
judge is Gemini classifying Gemini, within the same model family, sharing
training data, tokenizer, and failure modes. It is not a neutral instrument and
must not be described as one. Specific reasons for caution:

  * Correlated blind spots. A phrasing the answering model produces because of
    some quirk of its training is a phrasing the judge may misread for the same
    reason. The errors are not independent of what is being measured.
  * No ground truth. The judge's output is another model's opinion. It is
    cheaper and more consistent than keyword matching, not more authoritative.
  * Self-preference. Models have been shown to rate outputs from their own
    family more favourably. Judging refusal-vs-answer is less exposed to this
    than judging quality, but it is not immune.

This is why human-agreement validation below is not optional. Hand-label a
subset, compute agreement, and report it alongside any judged result. An
unvalidated judge produces numbers with no known error rate, and a Phase 7
hallucination rate resting on one is not defensible. If agreement with human
labels is poor, the judge's verdicts should be discarded, not adjusted.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VERDICTS = ("REFUSED", "ANSWERED", "PARTIAL")


class JudgeVerdict(BaseModel):
    """Schema enforced on the judge via structured output."""

    verdict: Literal["REFUSED", "ANSWERED", "PARTIAL"] = Field(
        description="How the assistant responded, ignoring factual correctness."
    )
    justification: str = Field(
        description="One sentence citing the decisive part of the response."
    )


@dataclass
class RefusalAssessment:
    """A judge verdict alongside the cheap keyword signal, and their agreement."""

    verdict: str
    justification: str
    keyword_refused: bool
    judge_model: str
    prompt_version: str
    error: str | None = None

    @property
    def judge_refused(self) -> bool:
        return self.verdict == "REFUSED"

    @property
    def signals_agree(self) -> bool:
        return self.judge_refused == self.keyword_refused

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "justification": self.justification,
            "judge_refused": self.judge_refused,
            "keyword_refused": self.keyword_refused,
            "signals_agree": self.signals_agree,
            "judge_model": self.judge_model,
            "judge_prompt_version": self.prompt_version,
            **({"error": self.error} if self.error else {}),
        }


def build_judge_prompt(question: str, answer: str) -> str:
    return (
        f"QUESTION ASKED\n{question}\n\n"
        f"ASSISTANT RESPONSE\n{answer}\n\n"
        "Classify the response."
    )


class RefusalJudge:
    """Classifies responses with a structured-output LLM call."""

    def __init__(self, model: str | None = None, llm_factory=None):
        self.model = model or config.JUDGE_MODEL
        self._llm_factory = llm_factory or self._default_factory
        self._client = None

    def _default_factory(self, model: str):
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=config.LLM_TEMPERATURE,
            google_api_key=config.GOOGLE_API_KEY,
            max_retries=2,
        )
        # json_schema mode makes the provider enforce the shape, so the verdict
        # cannot come back as prose that then needs fragile parsing.
        return llm.with_structured_output(JudgeVerdict)

    @property
    def client(self):
        if self._client is None:
            self._client = self._llm_factory(self.model)
        return self._client

    def classify(self, question: str, answer: str) -> RefusalAssessment:
        from src.rag.chain import RATE_LIMITER, looks_like_refusal

        keyword = looks_like_refusal(answer)

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=config.JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=build_judge_prompt(question, answer)),
        ]

        try:
            # Judging draws on the same per-minute quota as generation.
            RATE_LIMITER.wait()
            result = self.client.invoke(messages)
            verdict, justification = normalise_verdict(result)
            error = None
        except Exception as exc:
            # A failed judge call must be visibly distinct from a REFUSED
            # verdict, for the same reason a failed generation call must be
            # distinct from a refusal: it would otherwise be silently scored.
            verdict, justification = "ERROR", ""
            error = f"{type(exc).__name__}: {exc}"

        return RefusalAssessment(
            verdict=verdict,
            justification=justification,
            keyword_refused=keyword,
            judge_model=self.model,
            prompt_version=config.JUDGE_PROMPT_VERSION,
            error=error,
        )


def normalise_verdict(result) -> tuple[str, str]:
    """Accept a pydantic model or a dict; reject anything outside the vocabulary."""
    if isinstance(result, JudgeVerdict):
        verdict, justification = result.verdict, result.justification
    elif isinstance(result, dict):
        verdict = str(result.get("verdict", "")).strip().upper()
        justification = str(result.get("justification", "")).strip()
    else:
        verdict = str(getattr(result, "verdict", "")).strip().upper()
        justification = str(getattr(result, "justification", "")).strip()

    verdict = verdict.strip().upper()
    if verdict not in VERDICTS:
        raise ValueError(
            f"Judge returned {verdict!r}, expected one of {VERDICTS}"
        )
    return verdict, justification


# -- human agreement -------------------------------------------------------


def cohens_kappa(a: list[str], b: list[str]) -> float:
    """Chance-corrected agreement.

    Raw agreement flatters a judge on a skewed label distribution: if 80% of
    responses are ANSWERED, always guessing ANSWERED scores 0.80. Kappa
    subtracts the agreement expected from the marginals, so 0 means "no better
    than chance" regardless of skew.
    """
    if not a:
        return 0.0
    labels = sorted(set(a) | set(b))
    total = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / total

    expected = sum(
        (a.count(label) / total) * (b.count(label) / total) for label in labels
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


# Rubric shown to the human rater. Written into the template so both raters
# apply the same rule, and so the labelling standard is recorded alongside the
# labels rather than living only in someone's memory.
#
# Three categories, not four. A CLARIFICATION category was considered for the
# "Please provide the specific SEC filings..." response, but the judge already
# classifies that as REFUSED on defensible grounds - a request for input in
# place of an answer is a refusal. Adding a fourth category after seeing the
# judge's output would manufacture disagreement on a row it got right, and
# would be fitting the rubric to the result.
RUBRIC = [
    ("REFUSED", "Declines to provide the requested information. INCLUDES answers "
                "that decline by asking you to supply documents, specify a "
                "company, or restate the question - a request for input in place "
                "of an answer is a refusal."),
    ("ANSWERED", "Provides a substantive answer to what was asked."),
    ("PARTIAL", "Answers some of the question, or hedges while still supplying "
                "substantive content."),
]

TEMPLATE_COLUMNS = ["id", "arm", "question", "answer", "human_verdict"]
COMMENT_PREFIX = "#"


def _template_rows(payload: dict) -> list[dict]:
    rows = []
    for record in payload.get("results", []):
        for arm in ("with_context", "no_context"):
            entry = record.get(arm, {})
            if "answer" not in entry:
                continue
            rows.append(
                {
                    "id": f"{record['id']}::{arm}",
                    "arm": "RAG" if arm == "with_context" else "no-context",
                    "question": record["question"],
                    "answer": entry["answer"],
                    "human_verdict": "",
                }
            )
    return rows


# Phase 7 asks "did the model answer or decline?", which is binary. PARTIAL
# collapses to ANSWERED because a partial answer still asserts content that can
# be checked for groundedness - the distinction that matters for hallucination
# is whether anything was asserted at all.
BINARY_COLLAPSE = {"REFUSED": "REFUSED", "ANSWERED": "ANSWERED", "PARTIAL": "ANSWERED"}


def collapse_binary(labels: list[str]) -> list[str]:
    return [BINARY_COLLAPSE[label] for label in labels]


def confusion_matrix(human: list[str], judge: list[str], labels: list[str]) -> dict:
    """Nested dict: matrix[human_label][judge_label] = count."""
    return {
        h: {j: sum(1 for a, b in zip(human, judge) if a == h and b == j)
            for j in labels}
        for h in labels
    }


def distribution(labels: list[str], vocabulary: list[str]) -> dict:
    return {label: labels.count(label) for label in vocabulary}


def score_pair(human: list[str], judge: list[str], vocabulary: list[str]) -> dict:
    agree = sum(1 for a, b in zip(human, judge) if a == b)
    return {
        "n": len(human),
        "exact_agreement": agree,
        "agreement_rate": round(agree / len(human), 4),
        "cohens_kappa": round(cohens_kappa(human, judge), 4),
        "human_distribution": distribution(human, vocabulary),
        "judge_distribution": distribution(judge, vocabulary),
        "confusion_matrix": confusion_matrix(human, judge, vocabulary),
        "confusion_matrix_note": "matrix[human_label][judge_label] = count",
    }


def write_template(source: str, destination: str) -> int:
    """Emit a BLIND hand-labelling template as CSV.

    The judge's verdict and justification are deliberately NOT included. Showing
    them would anchor the human rater to the judge's answer, and agreement
    between an anchored rater and the thing that anchored them measures nothing.
    Verdicts are rejoined from the results file at agreement time.

    CSV rather than JSON because the rater edits exactly one short column: in
    JSON that means scrolling past multi-thousand-character answer strings to
    find each field, and a single stray quote invalidates the whole document.
    A damaged CSV cell stays local and recoverable. Written as utf-8-sig so
    Excel renders the typographic quotes in the answers correctly.
    """
    import csv

    payload = json.loads(open(source, encoding="utf-8").read())
    rows = _template_rows(payload)

    with open(destination, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEMPLATE_COLUMNS)
        writer.writeheader()

        # Rubric as leading rows marked with '#'. Valid CSV, visible at the top
        # of the spreadsheet, and skipped when the file is read back.
        def note(text, answer=""):
            writer.writerow({
                "id": COMMENT_PREFIX, "arm": "", "question": text,
                "answer": answer, "human_verdict": "",
            })

        note("LABELLING RUBRIC - fill in human_verdict on every row below.")
        note("Judge only WHETHER the assistant answered, not whether it was correct.")
        for label, description in RUBRIC:
            note(label, description)
        note("Rows beginning with '#' are ignored. Do not edit any other column.")

        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {destination}")
    print("\nRUBRIC")
    for label, description in RUBRIC:
        print(f"  {label:<9} {description}")
    print(f"\nFill in human_verdict, then:\n"
          f"  python -m src.rag.judge --agreement {destination}")
    return len(rows)


def _read_labels(path: str) -> list[dict]:
    """Read a filled-in template, CSV or JSON, skipping rubric rows."""
    if path.lower().endswith(".csv"):
        import csv

        with open(path, encoding="utf-8-sig", newline="") as handle:
            return [
                row for row in csv.DictReader(handle)
                if not str(row.get("id", "")).startswith(COMMENT_PREFIX)
            ]

    payload = json.loads(open(path, encoding="utf-8").read())
    return payload["rows"] if isinstance(payload, dict) else payload


def _judge_verdicts(results_path: str) -> dict[str, str]:
    """id -> judge verdict, rejoined from the results file."""
    payload = json.loads(open(results_path, encoding="utf-8").read())
    verdicts = {}
    for record in payload.get("results", []):
        for arm in ("with_context", "no_context"):
            entry = record.get(arm, {})
            assessment = entry.get("refusal_assessment")
            if assessment:
                verdicts[f"{record['id']}::{arm}"] = assessment["verdict"]
    return verdicts


def report_agreement(path: str, results_path: str | None = None) -> dict:
    """Compare blind hand labels against the judge's verdicts."""
    results_path = results_path or str(config.RESULTS_DIR / "rag_smoke_test.json")
    labels = _read_labels(path)
    verdicts = _judge_verdicts(results_path)

    rows, unmatched = [], []
    for row in labels:
        human = (row.get("human_verdict") or "").strip().upper()
        if not human:
            continue
        judge_verdict = verdicts.get(row["id"])
        if judge_verdict is None:
            unmatched.append(row["id"])
            continue
        rows.append({**row, "judge_verdict": judge_verdict, "human_verdict": human})

    if unmatched:
        print(f"WARNING: {len(unmatched)} labelled row(s) have no judge verdict "
              f"in {results_path}: {unmatched}")

    if not rows:
        raise SystemExit(
            f"No usable rows. Fill in human_verdict in {path} "
            f"(one of {', '.join(VERDICTS)})."
        )

    bad = sorted({r["human_verdict"] for r in rows} - set(VERDICTS))
    if bad:
        raise SystemExit(
            f"Unrecognised human_verdict value(s): {bad}. "
            f"Use exactly one of {', '.join(VERDICTS)}."
        )

    human = [row["human_verdict"] for row in rows]
    judge = [row["judge_verdict"].strip().upper() for row in rows]

    three_class = score_pair(human, judge, list(VERDICTS))
    binary = score_pair(
        collapse_binary(human), collapse_binary(judge), ["REFUSED", "ANSWERED"]
    )

    disagreements = [
        {
            "id": row["id"],
            "arm": row.get("arm", ""),
            "question": row.get("question", ""),
            "answer": row["answer"],
            "human": h,
            "judge": j,
            "collapses_under_binary": BINARY_COLLAPSE[h] == BINARY_COLLAPSE[j],
        }
        for row, h, j in zip(rows, human, judge)
        if h != j
    ]

    print(f"labelled rows      : {len(rows)}")
    print("\n-- 3-class (REFUSED / ANSWERED / PARTIAL) --")
    print(f"exact agreement    : {three_class['exact_agreement']}/{three_class['n']} "
          f"({three_class['agreement_rate']:.1%})")
    print(f"Cohen's kappa      : {three_class['cohens_kappa']:.4f}")

    print("\n-- binary (PARTIAL collapsed into ANSWERED) --")
    print(f"exact agreement    : {binary['exact_agreement']}/{binary['n']} "
          f"({binary['agreement_rate']:.1%})")
    print(f"Cohen's kappa      : {binary['cohens_kappa']:.4f}")
    print("  Phase 7's primary scoring axis: the question there is whether the")
    print("  model asserted anything checkable, and a PARTIAL answer did.")

    print(
        "\n  (kappa: <=0 no better than chance, 0.41-0.60 moderate, "
        "0.61-0.80 substantial, >0.80 almost perfect)"
    )

    distinct = sorted(set(human) | set(judge))
    print(f"classes present    : {', '.join(distinct)}")
    if len(distinct) < 2:
        print("  NOTE: only one class appears, so kappa is undefined in practice "
              "and reports 0. Exact agreement is the meaningful number here.")

    if disagreements:
        print(f"\ndisagreements ({len(disagreements)}):")
        for item in disagreements:
            note = " (disappears under binary collapse)" if item["collapses_under_binary"] else ""
            print(f"  {item['id']}: human={item['human']} judge={item['judge']}{note}")
            print(f"    {' '.join(item['answer'].split())[:160]}...")
    else:
        print("\nno disagreements")

    if len(rows) < 20:
        print(
            f"\nCAVEAT: {len(rows)} labelled rows is a small validation set. "
            "Treat kappa as indicative, not established."
        )

    return {
        "labelled_rows": len(rows),
        "three_class": three_class,
        "binary": binary,
        "binary_collapse_rule": BINARY_COLLAPSE,
        "primary_axis_for_phase_7": "binary",
        "disagreements": disagreements,
        # Kept flat for backwards compatibility with earlier callers.
        "exact_agreement": three_class["exact_agreement"],
        "agreement_rate": three_class["agreement_rate"],
        "cohens_kappa": three_class["cohens_kappa"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refusal judge utilities.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--template", nargs=2, metavar=("RESULTS_JSON", "OUTPUT_JSON"),
        help="emit a hand-labelling template from a results file",
    )
    group.add_argument(
        "--agreement", metavar="LABELS_CSV",
        help="report human/judge agreement from a filled-in template",
    )
    group.add_argument(
        "--classify", nargs=2, metavar=("QUESTION", "ANSWER"),
        help="classify a single question/answer pair (calls the API)",
    )
    parser.add_argument(
        "--results", metavar="RESULTS_JSON", default=None,
        help="results file to rejoin judge verdicts from "
             "(default results/rag_smoke_test.json)",
    )
    parser.add_argument(
        "--save", metavar="OUTPUT_JSON", default=None,
        help="write the full validation record to this path",
    )
    arguments = parser.parse_args()

    if arguments.template:
        write_template(*arguments.template)
    elif arguments.agreement:
        summary = report_agreement(arguments.agreement, arguments.results)
        if arguments.save:
            payload = {
                "phase": "6-judge-validation",
                "purpose": (
                    "Human validation of the LLM refusal judge. Without this the "
                    "judge is an instrument with no known error rate."
                ),
                "judge_model": config.JUDGE_MODEL,
                "judge_prompt_version": config.JUDGE_PROMPT_VERSION,
                "labels_file": arguments.agreement,
                "results_file": arguments.results
                or str(config.RESULTS_DIR / "rag_smoke_test.json"),
                "rubric": {label: text for label, text in RUBRIC},
                "rubric_note": (
                    "Three categories, not four. A CLARIFICATION category was "
                    "considered for a response that declined by asking for "
                    "documents, but the judge already classified that as REFUSED "
                    "on defensible grounds. Adding a category after seeing the "
                    "judge's output would manufacture disagreement on a row it "
                    "got right."
                ),
                "blind_labelling": (
                    "The template carried id, arm, question and answer only. The "
                    "judge's verdict and justification were stripped and their "
                    "absence verified by audit and by a permanent test; verdicts "
                    "were rejoined by row id at scoring time."
                ),
                "limitation": (
                    "LLM-as-judge is not independent measurement: Gemini grading "
                    "Gemini shares training data and failure modes. n=12 is "
                    "indicative, not established. The same person wrote the "
                    "questions, the judge prompt and the human labels."
                ),
                **summary,
            }
            path = Path(arguments.save)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"\nsaved {path}")
    else:
        question, answer = arguments.classify
        assessment = RefusalJudge().classify(question, answer)
        print(json.dumps(assessment.to_dict(), indent=2))


if __name__ == "__main__":
    main()
