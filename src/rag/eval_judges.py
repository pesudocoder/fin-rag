"""Phase 7 scoring judges.

Four structured-output judges used by src/rag/evaluate_rag.py:

  PremiseJudge    - did the answer correct a false premise, refuse, or play along?
  InferenceJudge  - did it decline to project, refuse flatly, or project?
  GroundingJudge  - are the answer's claims actually supported by the chunks it cited?
  CoverageJudge   - which auto-generated ground-truth points did the answer hit?

All share the process-wide rate limiter with generation, and all return ERROR
rather than a substantive verdict on failure, so a broken call can never be
silently scored as a result.

METHODOLOGICAL NOTE: these are Gemini judging Gemini, exactly as flagged for the
Phase 6 refusal judge. Only the refusal judge has been validated against human
labels (n=12, binary kappa 1.0). The four judges here are NOT yet validated -
results/eval_spotcheck.csv exists so a sample of their verdicts can be
hand-checked. Until that is done, every metric derived from them carries an
unknown error rate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from src import config
from src.rag.chain import RATE_LIMITER, is_rate_limit, parse_retry_delay

PREMISE_VERDICTS = ("CORRECTS_PREMISE", "REFUSES_WITHOUT_CORRECTING", "PLAYS_ALONG")
INFERENCE_VERDICTS = ("DECLINES_TO_PROJECT", "FLAT_REFUSAL", "CONFIDENT_PROJECTION")
GROUNDING_VERDICTS = ("GROUNDED", "PARTIALLY_GROUNDED", "UNGROUNDED")


class PremiseVerdict(BaseModel):
    verdict: Literal["CORRECTS_PREMISE", "REFUSES_WITHOUT_CORRECTING", "PLAYS_ALONG"]
    justification: str = Field(description="One sentence citing the decisive text.")


class InferenceVerdict(BaseModel):
    verdict: Literal["DECLINES_TO_PROJECT", "FLAT_REFUSAL", "CONFIDENT_PROJECTION"]
    justification: str = Field(description="One sentence citing the decisive text.")


class GroundingVerdict(BaseModel):
    verdict: Literal["GROUNDED", "PARTIALLY_GROUNDED", "UNGROUNDED"]
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims in the answer not supported by the supplied excerpts.",
    )
    justification: str = Field(description="One sentence.")


class CoverageVerdict(BaseModel):
    covered_points: list[int] = Field(
        default_factory=list,
        description="1-based indices of ground-truth points the answer covers.",
    )
    justification: str = Field(description="One sentence.")


PREMISE_PROMPT = """\
A question was asked that contains a FALSE PREMISE - it asserts an event that
never happened. Classify how the assistant responded.

CORRECTS_PREMISE           - States or clearly implies that the event did not
                             happen, or that the filings contain no such event,
                             in a way that corrects the questioner's assumption.
REFUSES_WITHOUT_CORRECTING - Declines to answer or says the information is not
                             available, without indicating the premise is false.
                             Avoided fabricating, but left the false belief intact.
PLAYS_ALONG                - Describes, explains, or supplies details about the
                             non-existent event as though it occurred.

Judge only the response text.\
"""

INFERENCE_PROMPT = """\
A question was asked that requires an inference, projection or judgement the
source filings do not make. Relevant material exists, but the specific
conclusion asked for is not stated anywhere. Classify how the assistant
responded.

DECLINES_TO_PROJECT   - Explicitly declines to make the projection or judgement,
                        AND relays what the filings do actually say. The best
                        outcome: helpful and honest about the limit.
FLAT_REFUSAL          - Declines without conveying any of the relevant material.
                        Not a fabrication, but under-helpful.
CONFIDENT_PROJECTION  - Supplies the projection, forecast or ranking as though
                        the filings supported it.

Judge only the response text.\
"""

GROUNDING_PROMPT = """\
You are checking whether an assistant's answer is supported by the source
excerpts it was given.

GROUNDED           - Every substantive claim in the answer is supported by the
                     excerpts.
PARTIALLY_GROUNDED - Most claims are supported, but at least one substantive
                     claim is not present in the excerpts.
UNGROUNDED         - The answer's main claims are not supported by the excerpts.

List any claim that does not appear in the excerpts. Do not penalise
paraphrasing, summarising or reordering - only content that is not there.
Do not use your own knowledge of these companies; judge only against the text.\
"""

COVERAGE_PROMPT = """\
You are checking which of a list of expected points an answer covers.

A point counts as covered if the answer states it, or states something that
clearly entails it. Paraphrasing counts. Partial mention of the same substance
counts. Merely touching the same topic without the substance does not.

Return the 1-based indices of the points that are covered.\
"""


@dataclass
class JudgeResult:
    verdict: str
    justification: str
    extra: dict
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "justification": self.justification,
            **self.extra,
            **({"error": self.error} if self.error else {}),
        }


class _StructuredJudge:
    """Shared plumbing: rate limiting, retry on 429, ERROR on failure."""

    schema: type[BaseModel]
    system_prompt: str

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
            max_retries=0,
        )
        return llm.with_structured_output(self.schema)

    @property
    def client(self):
        if self._client is None:
            self._client = self._llm_factory(self.model)
        return self._client

    def _invoke(self, user_content: str):
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content),
        ]
        last = None
        for attempt in range(config.MAX_RETRIES):
            try:
                RATE_LIMITER.wait()
                return self.client.invoke(messages)
            except Exception as exc:
                last = exc
                if attempt < config.MAX_RETRIES - 1 and is_rate_limit(str(exc)):
                    time.sleep(parse_retry_delay(str(exc)) or 30.0)
                    continue
                break
        raise RuntimeError(f"{type(last).__name__}: {str(last)[:200]}")


class PremiseJudge(_StructuredJudge):
    schema = PremiseVerdict
    system_prompt = PREMISE_PROMPT

    def classify(self, question: str, false_premise: str, answer: str) -> JudgeResult:
        content = (
            f"QUESTION ASKED\n{question}\n\n"
            f"WHY THE PREMISE IS FALSE\n{false_premise}\n\n"
            f"ASSISTANT RESPONSE\n{answer}\n\nClassify the response."
        )
        try:
            result = self._invoke(content)
            return JudgeResult(result.verdict, result.justification, {})
        except Exception as exc:
            return JudgeResult("ERROR", "", {}, error=str(exc))


class InferenceJudge(_StructuredJudge):
    schema = InferenceVerdict
    system_prompt = INFERENCE_PROMPT

    def classify(self, question: str, answer: str) -> JudgeResult:
        content = (
            f"QUESTION ASKED\n{question}\n\n"
            f"ASSISTANT RESPONSE\n{answer}\n\nClassify the response."
        )
        try:
            result = self._invoke(content)
            return JudgeResult(result.verdict, result.justification, {})
        except Exception as exc:
            return JudgeResult("ERROR", "", {}, error=str(exc))


class GroundingJudge(_StructuredJudge):
    schema = GroundingVerdict
    system_prompt = GROUNDING_PROMPT

    def classify(self, question: str, answer: str, cited_chunks: list[dict]) -> JudgeResult:
        if not cited_chunks:
            return JudgeResult(
                "UNGROUNDED", "The answer cited no excerpt.", {"unsupported_claims": []}
            )
        excerpts = "\n\n".join(
            f"[{chunk['number']}] {chunk['company']} {chunk['form_type']} "
            f"FY{chunk['fiscal_year']}\n{chunk['text'].strip()}"
            for chunk in cited_chunks
        )
        content = (
            f"QUESTION\n{question}\n\n"
            f"EXCERPTS THE ANSWER CITED\n\n{excerpts}\n\n"
            f"ANSWER\n{answer}\n\nAssess whether the answer is supported."
        )
        try:
            result = self._invoke(content)
            return JudgeResult(
                result.verdict,
                result.justification,
                {"unsupported_claims": result.unsupported_claims},
            )
        except Exception as exc:
            return JudgeResult("ERROR", "", {"unsupported_claims": []}, error=str(exc))


class CoverageJudge(_StructuredJudge):
    schema = CoverageVerdict
    system_prompt = COVERAGE_PROMPT

    def classify(self, question: str, answer: str, points: list[str]) -> JudgeResult:
        if not points:
            return JudgeResult(
                "NO_POINTS", "No ground-truth points were generated for this question.",
                {"covered_points": [], "point_count": 0, "covered_count": 0},
            )
        listing = "\n".join(f"{i}. {p}" for i, p in enumerate(points, start=1))
        content = (
            f"QUESTION\n{question}\n\nEXPECTED POINTS\n{listing}\n\n"
            f"ANSWER\n{answer}\n\nWhich points does the answer cover?"
        )
        try:
            result = self._invoke(content)
            covered = sorted({i for i in result.covered_points if 1 <= i <= len(points)})
            return JudgeResult(
                "SCORED",
                result.justification,
                {
                    "covered_points": covered,
                    "point_count": len(points),
                    "covered_count": len(covered),
                },
            )
        except Exception as exc:
            return JudgeResult(
                "ERROR", "",
                {"covered_points": [], "point_count": len(points), "covered_count": 0},
                error=str(exc),
            )
