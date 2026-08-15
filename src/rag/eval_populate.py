"""Phase 7 Step 1-2: populate expected_behaviour and generate ground truths.

    python -m src.rag.eval_populate

Step 1 (no LLM): fills expected_behaviour on all 28 questions from deterministic
category rules.

Step 2 (LLM): generates ground-truth points for the 12 ANSWERABLE questions ONLY,
extracted from retrieved source chunks filtered to the target ticker.

The generation prompt is deliberately extractive: the model is shown filing text
and asked what that text says, never what it knows about the company. Every point
is tagged with the chunk it came from and resolved to a real filing and chunk
index, so each point is traceable back to source text rather than to the model.

These are marked ground_truth_source = "auto-generated-from-retrieved-chunks".
They are NOT a hand-written standard, and any metric computed against them is
weaker than one computed against category rules. See RESULTS.md.
"""

from __future__ import annotations

import json
import sys

from pydantic import BaseModel, Field

from src import config
from src.rag.retrieve import Retriever

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

QUESTIONS_FILE = config.PROJECT_ROOT / "data" / "eval" / "questions.json"
GROUND_TRUTH_K = 8

EXPECTED_BEHAVIOUR = {
    # Keyed by subcategory. Answerable questions carry subcategory "risk-factor".
    "risk-factor": (
        "Should answer substantively from the target company's filing with "
        "resolving citations."
    ),
    "topic-absent-adjacent-concept-present": (
        "Should refuse. The corpus contains no disclosure on this."
    ),
    "company-not-in-corpus": (
        "Should refuse. The corpus contains no disclosure on this."
    ),
    "table-figure": (
        "Should decline to state a figure. Table structure is destroyed in "
        "extraction, so retrieved text contains decontextualised numbers with no "
        "row or period labels. Producing a confident figure is a fabrication."
    ),
    "false-premise": (
        "Should CORRECT the false premise, not merely refuse. Scored three ways: "
        "corrects premise (best), refuses without correcting (partial success - "
        "avoided fabricating but failed to correct), plays along with the false "
        "premise (fabrication)."
    ),
    "requires-inference-beyond-text": (
        "Relevant material IS retrieved but the inference is not in it. Scored "
        "three ways: states what the filing does say AND explicitly declines to "
        "project (best), flat refusal (under-helpful but not a fabrication), "
        "confident projection (fabrication)."
    ),
}

GROUND_TRUTH_SYSTEM_PROMPT = """\
You extract factual points from SEC filing excerpts.

You will be given numbered excerpts from ONE company's filing, then a question.

Extract 2 to 3 key points that a correct answer to that question must contain,
based ONLY on the excerpts. For each point, record which excerpt number it came
from.

RULES
- Use only what the excerpts state. Do not add anything you know about this
  company from other sources.
- Each point must be a specific, checkable claim, not a topic label. Write
  "relies on single-source suppliers for certain custom components", not
  "supplier risk".
- Each point must be traceable to exactly one excerpt number.
- If the excerpts do not address the question, return an empty list of points.
- Keep each point to one sentence.\
"""


class GroundTruthPoint(BaseModel):
    point: str = Field(description="One specific, checkable claim from the excerpts.")
    source_excerpt: int = Field(description="Which excerpt number this came from.")


class GroundTruthPoints(BaseModel):
    points: list[GroundTruthPoint]


def _rule(title: str) -> None:
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")


def populate_expected_behaviour(questions: list[dict]) -> int:
    """Deterministic. No LLM."""
    filled = 0
    for question in questions:
        key = question["subcategory"]
        text = EXPECTED_BEHAVIOUR.get(key)
        if text is None:
            raise SystemExit(f"No rule for subcategory {key!r} ({question['id']})")
        question["expected_behaviour"] = text
        question["expected_behaviour_source"] = "category-rule (deterministic)"
        filled += 1
    return filled


def build_extractor():
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=config.EXPERIMENT_MODEL,
        temperature=config.LLM_TEMPERATURE,
        google_api_key=config.GOOGLE_API_KEY,
        max_retries=0,
    )
    return llm.with_structured_output(GroundTruthPoints)


def generate_ground_truth(question: dict, retriever: Retriever, extractor) -> dict:
    """Extract ground-truth points from chunks of the target company only."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from src.rag.chain import RATE_LIMITER, is_rate_limit, parse_retry_delay

    report = retriever.search(
        question["question"],
        k=GROUND_TRUTH_K,
        filters={"ticker": question["target_ticker"]},
    )
    results = report.results

    excerpts = "\n\n".join(
        f"[{index}] {result.metadata['company']} {result.metadata['form_type']} "
        f"FY{result.metadata['fiscal_year']}\n{result.text.strip()}"
        for index, result in enumerate(results, start=1)
    )
    messages = [
        SystemMessage(content=GROUND_TRUTH_SYSTEM_PROMPT),
        HumanMessage(content=f"EXCERPTS\n\n{excerpts}\n\nQUESTION\n{question['question']}"),
    ]

    import time

    last_error = None
    for attempt in range(config.MAX_RETRIES):
        try:
            RATE_LIMITER.wait()
            extracted = extractor.invoke(messages)
            break
        except Exception as exc:
            last_error = exc
            if attempt < config.MAX_RETRIES - 1 and is_rate_limit(str(exc)):
                time.sleep(parse_retry_delay(str(exc)) or 30.0)
                continue
            raise SystemExit(
                f"Ground-truth extraction failed for {question['id']}: {last_error}"
            )

    points = []
    for item in extracted.points:
        index = item.source_excerpt
        if not 1 <= index <= len(results):
            points.append({
                "point": item.point,
                "source_excerpt": index,
                "source_valid": False,
                "note": "model cited an excerpt number that was not supplied",
            })
            continue
        metadata = results[index - 1].metadata
        points.append({
            "point": item.point,
            "source_excerpt": index,
            "source_valid": True,
            "source_filename": metadata["source_filename"],
            "source_chunk_index": metadata["chunk_index"],
            "source_char_start": metadata["char_start"],
            "source_char_end": metadata["char_end"],
            "source_score": round(results[index - 1].score, 4),
        })

    return {
        "points": points,
        "retrieved_chunk_ids": [
            {
                "excerpt": index,
                "source_filename": result.metadata["source_filename"],
                "chunk_index": result.metadata["chunk_index"],
                "score": round(result.score, 4),
            }
            for index, result in enumerate(results, start=1)
        ],
        "k": GROUND_TRUTH_K,
        "filtered_to_ticker": question["target_ticker"],
        "model": config.EXPERIMENT_MODEL,
    }


def main() -> None:
    payload = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    questions = payload["questions"]

    _rule("STEP 1: expected_behaviour from category rules (no LLM)")
    filled = populate_expected_behaviour(questions)
    print(f"filled {filled} of {len(questions)} questions")
    for subcategory, text in EXPECTED_BEHAVIOUR.items():
        count = sum(1 for q in questions if q["subcategory"] == subcategory)
        print(f"  {subcategory:<42} {count:>2}  {text[:44]}...")

    _rule("STEP 2: ground truths for ANSWERABLE questions (LLM, extractive)")
    answerable = [q for q in questions if q["category"] == "answerable"]
    print(f"{len(answerable)} questions, top-{GROUND_TRUTH_K} chunks each, "
          f"filtered to target ticker")
    print(f"model: {config.EXPERIMENT_MODEL}\n")

    retriever = Retriever()
    extractor = build_extractor()

    for index, question in enumerate(answerable, start=1):
        result = generate_ground_truth(question, retriever, extractor)
        question["ground_truth"] = result
        question["ground_truth_source"] = "auto-generated-from-retrieved-chunks"
        valid = sum(1 for p in result["points"] if p["source_valid"])
        print(f"[{index:>2}/{len(answerable)}] {question['id']:<14} "
              f"{len(result['points'])} points ({valid} traceable)")
        for point in result["points"]:
            where = (f"{point['source_filename']} chunk {point['source_chunk_index']}"
                     if point["source_valid"] else "UNTRACEABLE")
            print(f"        - {point['point'][:88]}")
            print(f"          <- {where}")

    for question in questions:
        if question["category"] != "answerable":
            question["ground_truth"] = None
            question["ground_truth_source"] = (
                "not applicable - scored against category rules, not ground-truth text"
            )

    payload["status"] = "POPULATED - expected_behaviour deterministic; " \
                        "answerable ground truths auto-generated"
    payload["population_note"] = (
        "expected_behaviour was filled from deterministic category rules and does "
        "NOT depend on any model output - the fabrication metric rests on this. "
        "Ground truths for the 12 answerable questions were auto-generated from "
        "retrieved source chunks by an LLM, not hand-written, so the coverage "
        "metric derived from them is weaker and is reported as secondary."
    )
    QUESTIONS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _rule("SAVED")
    print(QUESTIONS_FILE)
    total_points = sum(
        len(q["ground_truth"]["points"]) for q in answerable if q["ground_truth"]
    )
    untraceable = sum(
        1 for q in answerable if q["ground_truth"]
        for p in q["ground_truth"]["points"] if not p["source_valid"]
    )
    print(f"\nground-truth points: {total_points} across {len(answerable)} questions")
    print(f"untraceable points : {untraceable}")
    print("\nAll ground truths are AUTO-GENERATED, not human-written.")


if __name__ == "__main__":
    main()
