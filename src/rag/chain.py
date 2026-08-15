"""Phase 6: the RAG generation chain.

Composes the Phase 5 Retriever with Google Gemini to produce grounded, cited
answers. This is the first module in the project that calls an LLM.

    from src.rag.chain import RAGChain
    chain = RAGChain()
    result = chain.answer("What supply chain risks does Apple disclose?")

Why refusal is SEMANTIC, not a similarity threshold
---------------------------------------------------
The obvious way to make a RAG system refuse is to check the top retrieval score
and bail below some cutoff. Phase 5 measured whether that works on this corpus.
It does not.

  * highest out-of-corpus probe score : 0.4630
    ("aircraft fleet fuel hedging and aircraft lease obligations" - the corpus
    has no airline filings, but JPMorgan discusses leases and hedging at length,
    so the semantically adjacent text scores well)
  * lowest legitimate top-1 score     : 0.5750
    ("water scarcity and poor water quality affecting our production" - a real
    Coca-Cola risk factor that the corpus genuinely answers)

That is a margin of 0.1120, estimated from 16 probes. A 0.50 cutoff admits the
out-of-corpus query; a 0.60 cutoff refuses the legitimate one. The two
populations are not separable by score at this corpus size, and a boundary fitted
to 16 observations would not be trustworthy even if one existed.

So refusal is delegated to the model, instructed at the prompt level to judge
whether the retrieved text actually answers the question. Retrieval scores are
still recorded on every result (max_score, mean_score) so Phase 7 can test
post-hoc whether score correlates with refusal - but nothing is gated on them.

Citation by number
------------------
The model cites excerpts as [1]..[k] and nothing else. Numbers are mapped back to
accession numbers, EDGAR URLs and character spans in this module, from the
retriever's own metadata. Asking an LLM to reproduce an accession number like
0000320193-24-000123 invites transcription errors, and a subtly wrong accession
number is worse than none: it looks authoritative and resolves to nothing. The
model is only asked to do the part it can do reliably - point at which excerpt it
used.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from src import config
from src.rag.retrieve import Retriever, SearchResult
from src.utils import extract_text

# Matches [1], [2][4], [1, 3]. Bracketed integers only.
CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# Substrings that indicate the model declined to answer from the context. Used
# only for REPORTING - never to gate or rewrite the answer.
REFUSAL_MARKERS = (
    "do not contain",
    "does not contain",
    "do not provide",
    "does not provide",
    "not contain this information",
    "no information",
    "cannot be determined",
    "cannot determine",
    "not mentioned",
    "not stated",
    "not disclosed",
    "do not include",
    "does not include",
    "unable to answer",
    "insufficient information",
)


class RateLimiter:
    """Process-wide spacing between LLM calls.

    The free tier meters requests per minute per model, and generation and
    judging both draw on the same pool, so the limiter is shared rather than
    per-client. Without it a batch run issues calls as fast as the network
    allows and exhausts the window in seconds.
    """

    def __init__(self, min_interval: float = config.LLM_MIN_REQUEST_INTERVAL):
        self.min_interval = min_interval
        self._last_call: float | None = None

    def wait(self) -> None:
        # None rather than 0.0: the first call must not wait, and comparing
        # against a 0.0 sentinel would depend on the monotonic clock's origin.
        if self._last_call is not None:
            remaining = self.min_interval - (time.monotonic() - self._last_call)
            if remaining > 0:
                time.sleep(remaining)
        self._last_call = time.monotonic()


# Shared by RAGChain and RefusalJudge - they compete for the same quota.
RATE_LIMITER = RateLimiter()

# "Please retry in 55.244663777s" / "'retryDelay': '55s'"
RETRY_DELAY_PATTERN = re.compile(r"retry in (\d+(?:\.\d+)?)s|'retryDelay':\s*'(\d+)s'")


def parse_retry_delay(message: str) -> float | None:
    """Pull the server's requested wait out of a 429 payload.

    Exponential backoff is the wrong tool for a per-minute quota: 1s/2s/4s never
    reaches the ~55s the window actually needs. The server states the required
    delay; using it is both faster and more reliable than guessing.
    """
    match = RETRY_DELAY_PATTERN.search(message)
    if not match:
        return None
    value = match.group(1) or match.group(2)
    return min(float(value) + 1.0, config.LLM_RATE_LIMIT_MAX_WAIT)


def is_rate_limit(message: str) -> bool:
    lowered = message.lower()
    return (
        "429" in lowered
        or "quota" in lowered
        or "rate limit" in lowered
        or "resource_exhausted" in lowered
    )


class LLMCallError(RuntimeError):
    """Every retry and both models failed.

    Deliberately raised rather than returned as an answer string. Phase 7 scores
    refusals; a failed API call that got recorded as "the filings do not contain
    this information" would be counted as a correct refusal and would silently
    corrupt the hallucination measurement.
    """


@dataclass
class Citation:
    """A chunk number the model cited, resolved to a real source."""

    number: int
    valid: bool
    score: float | None = None
    metadata: dict = field(default_factory=dict)

    def render(self) -> str:
        if not self.valid:
            return f"[{self.number}] INVALID - no such excerpt was supplied"
        m = self.metadata
        return (
            f"[{self.number}] {m['company']} {m['form_type']} FY{m['fiscal_year']} "
            f"- chunk {m['chunk_index']}, chars {m['char_start']}-{m['char_end']} "
            f"(score {self.score:.4f})\n      {m['source_url']}"
        )


@dataclass
class RAGResult:
    """Everything Phase 7 needs to score one question."""

    question: str
    answer: str
    used_context: bool
    model: str
    fallback_used: bool
    latency_seconds: float
    prompt_version: str
    chunks: list[dict] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    max_score: float | None = None
    mean_score: float | None = None
    refused: bool = False
    attempts: int = 1

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "used_context": self.used_context,
            "model": self.model,
            "fallback_used": self.fallback_used,
            "latency_seconds": round(self.latency_seconds, 3),
            "prompt_version": self.prompt_version,
            "attempts": self.attempts,
            "refused": self.refused,
            "retrieval": {
                "chunk_count": len(self.chunks),
                "max_score": self.max_score,
                "mean_score": self.mean_score,
                "chunks": self.chunks,
            },
            "citations": [
                {
                    "number": c.number,
                    "valid": c.valid,
                    "score": c.score,
                    **({} if not c.valid else {
                        "ticker": c.metadata["ticker"],
                        "company": c.metadata["company"],
                        "form_type": c.metadata["form_type"],
                        "fiscal_year": c.metadata["fiscal_year"],
                        "accession_number": c.metadata["accession_number"],
                        "source_url": c.metadata["source_url"],
                        "chunk_index": c.metadata["chunk_index"],
                        "char_start": c.metadata["char_start"],
                        "char_end": c.metadata["char_end"],
                    }),
                }
                for c in self.citations
            ],
            "invalid_citations": self.invalid_citations,
        }


def build_context(results: list[SearchResult]) -> str:
    """Number the retrieved chunks [1]..[k] with their provenance headers."""
    blocks = []
    for position, result in enumerate(results, start=1):
        m = result.metadata
        blocks.append(
            f"[{position}] {m['company']} ({m['ticker']}) - {m['form_type']}, "
            f"fiscal year {m['fiscal_year']}\n{result.text.strip()}"
        )
    return "\n\n".join(blocks)


def parse_citations(answer: str, supplied: int) -> tuple[list[int], list[int]]:
    """Extract cited numbers, split into valid and out-of-range.

    A model citing [7] when only 5 excerpts were supplied has invented a source.
    That is a hallucination signal in its own right and is counted separately
    rather than quietly dropped.
    """
    seen: list[int] = []
    for match in CITATION_PATTERN.finditer(answer):
        for part in match.group(1).split(","):
            number = int(part.strip())
            if number not in seen:
                seen.append(number)

    valid = [n for n in seen if 1 <= n <= supplied]
    invalid = [n for n in seen if not 1 <= n <= supplied]
    return valid, invalid


def looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


class RAGChain:
    """Retriever + Gemini, with a matched no-context baseline path.

    pinned_model
    ------------
    When set, every call uses that model and per-call fallback is disabled. Batch
    and experiment paths MUST pin, because per-call fallback resolves
    nondeterministically under rate limiting and can hand the two arms different
    models - turning a context comparison into a partial model comparison. See
    config.EXPERIMENT_MODEL.

    When None (the interactive default), the chain tries LLM_MODEL and falls back
    to LLM_FALLBACK_MODEL, which is the right behaviour for a single ad-hoc
    question where getting an answer matters more than comparability.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm_factory=None,
        pinned_model: str | None = None,
    ):
        self.retriever = retriever if retriever is not None else Retriever()
        self._llm_factory = llm_factory or self._default_llm_factory
        self._clients: dict[str, Any] = {}
        self.pinned_model = pinned_model

    @property
    def model_chain(self) -> list[tuple[str, bool]]:
        """(model, is_fallback) pairs to try, in order."""
        if self.pinned_model:
            return [(self.pinned_model, False)]
        return [(config.LLM_MODEL, False), (config.LLM_FALLBACK_MODEL, True)]

    @staticmethod
    def _default_llm_factory(model: str):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=config.LLM_TEMPERATURE,
            google_api_key=config.GOOGLE_API_KEY,
            # Retries are handled here so backoff and model fallback stay
            # visible in the result rather than hidden inside the client.
            max_retries=0,
        )

    def _client(self, model: str):
        if model not in self._clients:
            self._clients[model] = self._llm_factory(model)
        return self._clients[model]

    # -- LLM invocation ----------------------------------------------------

    def _invoke(self, system: str, user: str) -> tuple[str, str, bool, int]:
        """Call the LLM with backoff, then fall back to the lighter model.

        Returns (text, model_used, fallback_used, attempts). Raises LLMCallError
        if every attempt on both models fails - never returns a placeholder
        string, so a transport failure can never be mistaken for a refusal.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        attempts = 0
        errors: list[str] = []

        for model, is_fallback in self.model_chain:
            for retry in range(config.MAX_RETRIES):
                attempts += 1
                try:
                    RATE_LIMITER.wait()
                    response = self._client(model).invoke(messages)
                    # .content is a LIST of content blocks, not a string.
                    # Calling .strip() on it raises AttributeError.
                    return extract_text(response.content).strip(), model, is_fallback, attempts
                except Exception as exc:
                    message = str(exc)
                    errors.append(
                        f"{model} attempt {retry + 1}: {type(exc).__name__}: "
                        f"{message[:200]}"
                    )
                    if retry < config.MAX_RETRIES - 1 and is_rate_limit(message):
                        # Honour the server's stated delay; fall back to a long
                        # wait, since the quota window is per minute.
                        time.sleep(
                            parse_retry_delay(message)
                            or min(15.0 * (retry + 1), config.LLM_RATE_LIMIT_MAX_WAIT)
                        )
                        continue
                    break   # non-retryable, or retries exhausted: try fallback

        raise LLMCallError(
            "All LLM attempts failed on both models:\n  " + "\n  ".join(errors)
        )

    # -- public API --------------------------------------------------------

    def answer(
        self,
        question: str,
        k: int | None = None,
        filters: dict | None = None,
        dedupe: bool = True,
        use_context: bool = True,
    ) -> RAGResult:
        """Answer a question, with retrieved context or without it."""
        k = k or config.TOP_K
        started = time.perf_counter()

        if not use_context:
            # Baseline arm. Same model, same temperature, same question wording -
            # the only difference from the RAG arm is that no context is supplied.
            text, model, fallback, attempts = self._invoke(
                config.NO_CONTEXT_SYSTEM_PROMPT, question
            )
            return RAGResult(
                question=question,
                answer=text,
                used_context=False,
                model=model,
                fallback_used=fallback,
                latency_seconds=time.perf_counter() - started,
                prompt_version=config.RAG_PROMPT_VERSION,
                refused=looks_like_refusal(text),
                attempts=attempts,
            )

        report = self.retriever.search(question, k=k, filters=filters, dedupe=dedupe)
        results = report.results
        context = build_context(results)

        prompt = (
            f"EXCERPTS FROM SEC FILINGS\n\n{context}\n\n"
            f"QUESTION\n{question}"
        )
        text, model, fallback, attempts = self._invoke(config.RAG_SYSTEM_PROMPT, prompt)

        valid, invalid = parse_citations(text, len(results))
        citations = [
            Citation(
                number=number,
                valid=True,
                score=round(results[number - 1].score, 4),
                metadata=results[number - 1].metadata,
            )
            for number in valid
        ] + [Citation(number=number, valid=False) for number in invalid]

        scores = [result.score for result in results]
        return RAGResult(
            question=question,
            answer=text,
            used_context=True,
            model=model,
            fallback_used=fallback,
            latency_seconds=time.perf_counter() - started,
            prompt_version=config.RAG_PROMPT_VERSION,
            chunks=[
                {
                    "number": position,
                    "score": round(result.score, 4),
                    "text": result.text,
                    **result.metadata,
                }
                for position, result in enumerate(results, start=1)
            ],
            citations=citations,
            invalid_citations=invalid,
            max_score=round(max(scores), 4) if scores else None,
            mean_score=round(sum(scores) / len(scores), 4) if scores else None,
            refused=looks_like_refusal(text),
            attempts=attempts,
        )
