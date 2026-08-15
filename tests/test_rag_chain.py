"""Phase 6 RAG chain tests.

    pytest tests/test_rag_chain.py -v

NO API CALLS. The LLM is replaced with a stub that records what it was sent and
returns a scripted reply, and the retriever is a fake holding fixed chunks. These
tests cover the deterministic logic around the model - context assembly and
numbering, citation parsing, out-of-range citation detection, and the guarantee
that the no-context arm supplies no chunks - none of which needs a live model to
verify, and all of which would be slow, costly and flaky to test against one.
"""

from __future__ import annotations

import pytest

from src import config
from src.rag.chain import (
    LLMCallError,
    RAGChain,
    build_context,
    looks_like_refusal,
    parse_citations,
)
from src.rag.retrieve import SearchResult, SearchReport


def make_result(rank, score, ticker, company, year, chunk_index, text):
    return SearchResult(
        rank=rank,
        score=score,
        row=chunk_index,
        text=text,
        metadata={
            "source_filename": f"{ticker}_10-K_{year}.htm",
            "ticker": ticker,
            "company": company,
            "form_type": "10-K",
            "fiscal_year": year,
            "accession_number": f"000000{chunk_index}-{str(year)[2:]}-000001",
            "source_url": f"https://www.sec.gov/Archives/{ticker}/{chunk_index}.htm",
            "chunk_index": chunk_index,
            "char_start": chunk_index * 1000,
            "char_end": chunk_index * 1000 + len(text),
            "token_count": 42,
        },
    )


CHUNKS = [
    make_result(1, 0.81, "AAPL", "Apple Inc.", 2024, 45, "Apple relies on single-source suppliers."),
    make_result(2, 0.77, "AAPL", "Apple Inc.", 2023, 47, "Manufacturing is concentrated in Asia."),
    make_result(3, 0.69, "KO", "COCA COLA CO", 2024, 90, "Water scarcity may disrupt production."),
]


class FakeRetriever:
    """Stands in for the Phase 5 Retriever without touching FAISS or disk."""

    def __init__(self, results=None):
        self.results = results if results is not None else CHUNKS
        self.calls = []

    def search(self, query, k=5, filters=None, dedupe=True):
        self.calls.append({"query": query, "k": k, "filters": filters, "dedupe": dedupe})
        results = self.results[:k]
        return SearchReport(
            query=query, k=k, results=results, filled=len(results) >= k, fetched=k * 4
        )


class StubLLM:
    """Records the messages it received and returns a scripted reply.

    content is returned as a LIST of blocks, mirroring the real Gemini client -
    the project has already been bitten by assuming .content is a string, so the
    stub reproduces the shape that caused it.
    """

    def __init__(self, reply="Answer [1].", fail_times=0, error=None):
        self.reply = reply
        self.fail_times = fail_times
        self.error = error or RuntimeError("429 quota exceeded")
        self.received = []

    def invoke(self, messages):
        self.received.append(messages)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.error
        return type("Response", (), {"content": [{"type": "text", "text": self.reply}]})()


def make_chain(reply="Answer [1].", results=None, **stub_kwargs):
    stub = StubLLM(reply=reply, **stub_kwargs)
    chain = RAGChain(retriever=FakeRetriever(results), llm_factory=lambda model: stub)
    return chain, stub


@pytest.fixture(autouse=True)
def no_real_throttling(monkeypatch):
    """Neutralise the shared rate limiter for every test in this file.

    RATE_LIMITER is module-level and paces real API calls at 4.5s. Without this
    the suite would sleep for real between stubbed calls - it added 100s before
    being disabled. TestRateLimiting re-patches the clock explicitly to test the
    limiter's own behaviour.
    """
    monkeypatch.setattr("src.rag.chain.RATE_LIMITER.wait", lambda: None)


# -- context assembly ------------------------------------------------------


class TestContextAssembly:
    def test_chunks_are_numbered_from_one(self):
        context = build_context(CHUNKS)
        assert context.startswith("[1]")
        assert "[2]" in context and "[3]" in context
        assert "[0]" not in context
        assert "[4]" not in context

    def test_each_block_carries_provenance(self):
        context = build_context(CHUNKS)
        assert "Apple Inc. (AAPL) - 10-K, fiscal year 2024" in context
        assert "COCA COLA CO (KO) - 10-K, fiscal year 2024" in context

    def test_chunk_text_is_included(self):
        context = build_context(CHUNKS)
        for chunk in CHUNKS:
            assert chunk.text in context

    def test_accession_numbers_are_not_sent_to_the_model(self):
        """The model cites by number; identifiers are resolved in our code.

        Asking a model to reproduce an accession number invites transcription
        errors, and a subtly wrong one looks authoritative and resolves to
        nothing.
        """
        context = build_context(CHUNKS)
        for chunk in CHUNKS:
            assert chunk.metadata["accession_number"] not in context
            assert chunk.metadata["source_url"] not in context

    def test_context_reaches_the_prompt(self):
        chain, stub = make_chain()
        chain.answer("Why?")
        prompt = stub.received[0][1].content
        assert "Apple relies on single-source suppliers." in prompt
        assert "[1]" in prompt
        assert "QUESTION" in prompt and "Why?" in prompt

    def test_rag_system_prompt_is_used(self):
        chain, stub = make_chain()
        chain.answer("Why?")
        assert stub.received[0][0].content == config.RAG_SYSTEM_PROMPT


# -- citation parsing ------------------------------------------------------


class TestCitationParsing:
    @pytest.mark.parametrize(
        "answer, expected",
        [
            ("Apple relies on suppliers [1].", [1]),
            ("Two sources [1][2].", [1, 2]),
            ("Comma form [1, 3].", [1, 3]),
            ("Repeated [2] and again [2].", [2]),
            ("No citations at all.", []),
            ("Out of order [3][1].", [3, 1]),
        ],
    )
    def test_valid_citations(self, answer, expected):
        valid, invalid = parse_citations(answer, supplied=3)
        assert valid == expected
        assert invalid == []

    def test_out_of_range_citation_is_flagged(self):
        """A model citing [7] when 3 excerpts were supplied invented a source."""
        valid, invalid = parse_citations("Claim [1] and claim [7].", supplied=3)
        assert valid == [1]
        assert invalid == [7]

    def test_zero_is_out_of_range(self):
        valid, invalid = parse_citations("Bad [0].", supplied=3)
        assert valid == []
        assert invalid == [0]

    def test_invalid_citations_surface_on_the_result(self):
        chain, _ = make_chain(reply="Grounded [1]. Fabricated [9].")
        result = chain.answer("Why?")

        assert result.invalid_citations == [9]
        numbers = {c.number: c.valid for c in result.citations}
        assert numbers == {1: True, 9: False}

        invalid = next(c for c in result.citations if not c.valid)
        assert "INVALID" in invalid.render()

    def test_valid_citations_resolve_to_full_provenance(self):
        chain, _ = make_chain(reply="See [2].")
        result = chain.answer("Why?")

        citation = result.citations[0]
        assert citation.valid
        assert citation.metadata["ticker"] == "AAPL"
        assert citation.metadata["fiscal_year"] == 2023
        assert citation.metadata["chunk_index"] == 47
        assert citation.score == 0.77
        assert citation.metadata["accession_number"]
        assert citation.metadata["source_url"]

    def test_citation_maps_to_the_right_chunk(self):
        """Off-by-one here would attribute claims to the wrong filing."""
        chain, _ = make_chain(reply="Third one [3].")
        result = chain.answer("Why?")
        assert result.citations[0].metadata["ticker"] == "KO"
        assert result.citations[0].metadata["chunk_index"] == 90


# -- the two arms ----------------------------------------------------------


class TestNoContextArm:
    def test_no_chunks_are_supplied(self):
        chain, stub = make_chain()
        result = chain.answer("Why?", use_context=False)

        assert result.used_context is False
        assert result.chunks == []
        assert result.citations == []
        assert result.max_score is None
        assert result.mean_score is None

    def test_retriever_is_not_called(self):
        chain, _ = make_chain()
        chain.answer("Why?", use_context=False)
        assert chain.retriever.calls == []

    def test_neutral_system_prompt_and_bare_question(self):
        chain, stub = make_chain()
        chain.answer("What are Apple's risks?", use_context=False)

        system, human = stub.received[0]
        assert system.content == config.NO_CONTEXT_SYSTEM_PROMPT
        assert human.content == "What are Apple's risks?"
        assert "EXCERPTS" not in human.content

    def test_question_wording_is_identical_across_arms(self):
        """Phase 7's comparison is only valid if context is the sole difference."""
        question = "What risks does Apple disclose?"
        chain, stub = make_chain()
        chain.answer(question, use_context=True)
        chain.answer(question, use_context=False)

        with_context = stub.received[0][1].content
        without_context = stub.received[1][1].content
        assert without_context == question
        assert with_context.endswith(question)


class TestRetrievalScoreLogging:
    def test_scores_are_recorded_but_not_gated_on(self):
        chain, _ = make_chain(reply="Answer [1].")
        result = chain.answer("Why?")

        assert result.max_score == 0.81
        assert result.mean_score == round((0.81 + 0.77 + 0.69) / 3, 4)

    def test_low_scores_still_produce_an_answer(self):
        """Refusal is semantic; a low score must not suppress generation."""
        weak = [make_result(1, 0.11, "KO", "COCA COLA CO", 2024, 5, "Unrelated text.")]
        chain, _ = make_chain(reply="Answer [1].", results=weak)
        result = chain.answer("Why?")

        assert result.max_score == 0.11
        assert result.answer == "Answer [1]."
        assert result.chunks


# -- failure handling ------------------------------------------------------


class TestFailureHandling:
    def test_rate_limit_is_retried_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("src.rag.chain.time.sleep", lambda _: None)
        chain, stub = make_chain(reply="Recovered [1].", fail_times=2)
        result = chain.answer("Why?")

        assert result.answer == "Recovered [1]."
        assert result.attempts == 3

    def test_total_failure_raises_and_is_not_an_answer(self, monkeypatch):
        """A failed call must never be mistakable for a refusal.

        Phase 7 scores refusals. If a transport error were returned as text
        resembling "the filings do not contain this information", it would be
        counted as a correct refusal and corrupt the measurement.
        """
        monkeypatch.setattr("src.rag.chain.time.sleep", lambda _: None)
        chain, _ = make_chain(fail_times=99)

        with pytest.raises(LLMCallError) as excinfo:
            chain.answer("Why?")
        assert "All LLM attempts failed" in str(excinfo.value)

    def test_list_content_is_extracted_not_stringified(self):
        """Guards the known .content-is-a-list bug."""
        chain, _ = make_chain(reply="Plain answer [1].")
        result = chain.answer("Why?")
        assert result.answer == "Plain answer [1]."
        assert "type" not in result.answer and "{" not in result.answer


class TestRateLimiting:
    """Free tier meters 15 requests/minute per model; batch runs must self-pace."""

    def test_retry_delay_is_parsed_from_prose(self):
        from src.rag.chain import parse_retry_delay

        assert parse_retry_delay("Please retry in 55.244663777s.") == pytest.approx(56.24, abs=0.01)

    def test_retry_delay_is_parsed_from_retryinfo(self):
        from src.rag.chain import parse_retry_delay

        assert parse_retry_delay("{'retryDelay': '43s'}") == pytest.approx(44.0)

    def test_retry_delay_is_capped(self):
        from src.rag.chain import parse_retry_delay

        assert parse_retry_delay("retry in 9999s") == config.LLM_RATE_LIMIT_MAX_WAIT

    def test_no_delay_in_unrelated_error(self):
        from src.rag.chain import parse_retry_delay

        assert parse_retry_delay("connection reset by peer") is None

    @pytest.mark.parametrize(
        "message, expected",
        [
            ("429 RESOURCE_EXHAUSTED", True),
            ("Quota exceeded for metric", True),
            ("rate limit reached", True),
            ("connection reset by peer", False),
            ("invalid api key", False),
        ],
    )
    def test_rate_limit_detection(self, message, expected):
        from src.rag.chain import is_rate_limit

        assert is_rate_limit(message) is expected

    def test_first_call_never_waits(self, monkeypatch):
        from src.rag.chain import RateLimiter

        slept = []
        monkeypatch.setattr("src.rag.chain.time.sleep", slept.append)
        monkeypatch.setattr("src.rag.chain.time.monotonic", lambda: 1000.0)

        RateLimiter(min_interval=4.5).wait()
        assert slept == []

    def test_limiter_spaces_calls(self, monkeypatch):
        from src.rag.chain import RateLimiter

        slept = []
        monkeypatch.setattr("src.rag.chain.time.sleep", slept.append)
        # wait() reads the clock once to measure, once to record.
        clock = iter([1000.0, 1000.5, 1000.5])
        monkeypatch.setattr("src.rag.chain.time.monotonic", lambda: next(clock))

        limiter = RateLimiter(min_interval=4.5)
        limiter.wait()   # records t=1000.0, no wait
        limiter.wait()   # 0.5s later: must sleep the remaining 4.0s
        assert slept == [pytest.approx(4.0)]

    def test_limiter_does_not_sleep_when_interval_elapsed(self, monkeypatch):
        from src.rag.chain import RateLimiter

        slept = []
        monkeypatch.setattr("src.rag.chain.time.sleep", slept.append)
        clock = iter([1000.0, 1100.0, 1100.0])
        monkeypatch.setattr("src.rag.chain.time.monotonic", lambda: next(clock))

        limiter = RateLimiter(min_interval=4.5)
        limiter.wait()
        limiter.wait()
        assert slept == []

    def test_rate_limited_call_waits_the_server_delay(self, monkeypatch):
        """Backoff must honour the stated delay, not use 1s/2s/4s."""
        monkeypatch.setattr("src.rag.chain.RATE_LIMITER.wait", lambda: None)
        slept = []
        monkeypatch.setattr("src.rag.chain.time.sleep", slept.append)

        error = RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 55s.")
        chain, _ = make_chain(reply="Recovered [1].", fail_times=1, error=error)
        chain.answer("Why?")

        assert slept and slept[0] == pytest.approx(56.0)


class TestModelPinning:
    """Experiments must pin the model; interactive use may fall back."""

    def test_unpinned_chain_tries_primary_then_fallback(self):
        chain, _ = make_chain()
        assert chain.model_chain == [
            (config.LLM_MODEL, False),
            (config.LLM_FALLBACK_MODEL, True),
        ]

    def test_pinned_chain_has_no_fallback(self):
        stub = StubLLM()
        chain = RAGChain(
            retriever=FakeRetriever(), llm_factory=lambda m: stub,
            pinned_model=config.EXPERIMENT_MODEL,
        )
        assert chain.model_chain == [(config.EXPERIMENT_MODEL, False)]

    def test_pinned_model_is_reported_on_the_result(self):
        stub = StubLLM()
        chain = RAGChain(
            retriever=FakeRetriever(), llm_factory=lambda m: stub,
            pinned_model=config.EXPERIMENT_MODEL,
        )
        result = chain.answer("Why?")
        assert result.model == config.EXPERIMENT_MODEL
        assert result.fallback_used is False

    def test_both_arms_use_the_pinned_model(self, monkeypatch):
        """The defect: per-call fallback gave the arms different models."""
        monkeypatch.setattr("src.rag.chain.time.sleep", lambda _: None)
        requested = []

        def factory(model):
            requested.append(model)
            return StubLLM()

        chain = RAGChain(
            retriever=FakeRetriever(), llm_factory=factory,
            pinned_model=config.EXPERIMENT_MODEL,
        )
        rag = chain.answer("Why?", use_context=True)
        base = chain.answer("Why?", use_context=False)

        assert rag.model == base.model == config.EXPERIMENT_MODEL
        assert set(requested) == {config.EXPERIMENT_MODEL}

    def test_pinned_failure_does_not_silently_switch_models(self, monkeypatch):
        """A pinned run must fail rather than quietly answer on another model."""
        monkeypatch.setattr("src.rag.chain.time.sleep", lambda _: None)
        requested = []

        def factory(model):
            requested.append(model)
            return StubLLM(fail_times=99)

        chain = RAGChain(
            retriever=FakeRetriever(), llm_factory=factory,
            pinned_model=config.EXPERIMENT_MODEL,
        )
        with pytest.raises(LLMCallError):
            chain.answer("Why?")
        assert set(requested) == {config.EXPERIMENT_MODEL}
        assert config.LLM_MODEL not in requested or config.LLM_MODEL == config.EXPERIMENT_MODEL


class TestRefusalDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "The filings provided do not contain this information.",
            "The excerpts does not contain the answer.",
            "This information is not disclosed in the provided excerpts.",
            "It cannot be determined from the excerpts.",
        ],
    )
    def test_refusal_phrasings_are_detected(self, text):
        assert looks_like_refusal(text)

    def test_substantive_answer_is_not_a_refusal(self):
        assert not looks_like_refusal("Apple relies on single-source suppliers [1].")

    def test_refusal_flag_appears_on_the_result(self):
        chain, _ = make_chain(reply="The filings provided do not contain this information.")
        assert chain.answer("Why?").refused is True
