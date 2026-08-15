"""Tests for the refusal judge. NO API CALLS - the judge LLM is stubbed.

    pytest tests/test_judge.py -v

Covers verdict parsing and normalisation, rejection of out-of-vocabulary
verdicts, judge-vs-keyword disagreement reporting, failure handling, and the
human-agreement statistics.
"""

from __future__ import annotations

import json

import pytest

from src import config
from src.rag.judge import (
    RUBRIC,
    TEMPLATE_COLUMNS,
    JudgeVerdict,
    RefusalJudge,
    _read_labels,
    cohens_kappa,
    normalise_verdict,
    report_agreement,
    write_template,
)

REFUSAL_TEXT = "The filings provided do not contain this information."
# The exact phrasing the keyword detector missed, which motivated the judge.
MISSED_BY_KEYWORDS = (
    "Based on the provided context, there are no disclosures regarding "
    "aircraft fleet fuel hedging or aircraft lease obligations."
)


class StubJudgeLLM:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.received = []

    def invoke(self, messages):
        self.received.append(messages)
        if self.error:
            raise self.error
        return self.result


def make_judge(result=None, error=None):
    stub = StubJudgeLLM(result=result, error=error)
    return RefusalJudge(llm_factory=lambda model: stub), stub


@pytest.fixture(autouse=True)
def no_real_throttling(monkeypatch):
    """The judge shares the process-wide rate limiter, which paces at 4.5s.

    Left live, these stubbed tests would sleep for real between calls.
    """
    monkeypatch.setattr("src.rag.chain.RATE_LIMITER.wait", lambda: None)


class TestVerdictNormalisation:
    def test_pydantic_model(self):
        verdict, why = normalise_verdict(
            JudgeVerdict(verdict="REFUSED", justification="Declines to answer.")
        )
        assert verdict == "REFUSED"
        assert why == "Declines to answer."

    def test_dict_payload(self):
        verdict, why = normalise_verdict(
            {"verdict": "ANSWERED", "justification": "Gives substance."}
        )
        assert verdict == "ANSWERED"

    def test_lowercase_is_normalised(self):
        verdict, _ = normalise_verdict({"verdict": "partial", "justification": "x"})
        assert verdict == "PARTIAL"

    def test_whitespace_is_stripped(self):
        verdict, _ = normalise_verdict({"verdict": "  refused  ", "justification": "x"})
        assert verdict == "REFUSED"

    @pytest.mark.parametrize("bad", ["MAYBE", "", "REFUSE", "yes"])
    def test_unknown_verdict_rejected(self, bad):
        """Silently accepting an unexpected label would corrupt every count."""
        with pytest.raises(ValueError):
            normalise_verdict({"verdict": bad, "justification": "x"})


class TestClassification:
    def test_refusal_verdict(self):
        judge, _ = make_judge(
            JudgeVerdict(verdict="REFUSED", justification="States info unavailable.")
        )
        assessment = judge.classify("What was the figure?", REFUSAL_TEXT)

        assert assessment.verdict == "REFUSED"
        assert assessment.judge_refused is True
        assert assessment.keyword_refused is True
        assert assessment.signals_agree is True

    def test_judge_catches_what_keywords_miss(self):
        """The defect that motivated this module."""
        judge, _ = make_judge(
            JudgeVerdict(verdict="REFUSED", justification="Says no disclosures exist.")
        )
        assessment = judge.classify("Aircraft hedging?", MISSED_BY_KEYWORDS)

        assert assessment.keyword_refused is False, (
            "keyword detector unexpectedly catches this phrasing now"
        )
        assert assessment.judge_refused is True
        assert assessment.signals_agree is False

    def test_answered_verdict(self):
        judge, _ = make_judge(
            JudgeVerdict(verdict="ANSWERED", justification="Lists risks.")
        )
        assessment = judge.classify("Risks?", "Apple relies on single-source suppliers.")

        assert assessment.verdict == "ANSWERED"
        assert assessment.judge_refused is False
        assert assessment.signals_agree is True

    def test_partial_is_not_counted_as_refusal(self):
        judge, _ = make_judge(
            JudgeVerdict(verdict="PARTIAL", justification="Answers half.")
        )
        assessment = judge.classify("Two things?", "One is X. The other is not stated.")
        assert assessment.verdict == "PARTIAL"
        assert assessment.judge_refused is False

    def test_question_and_answer_reach_the_prompt(self):
        judge, stub = make_judge(
            JudgeVerdict(verdict="ANSWERED", justification="x")
        )
        judge.classify("What is the risk?", "The risk is high.")

        system, human = stub.received[0]
        assert system.content == config.JUDGE_SYSTEM_PROMPT
        assert "What is the risk?" in human.content
        assert "The risk is high." in human.content

    def test_failure_is_distinct_from_a_refusal(self):
        """A broken judge call must never be silently scored as REFUSED."""
        judge, _ = make_judge(error=RuntimeError("429 quota exceeded"))
        assessment = judge.classify("Q", "A")

        assert assessment.verdict == "ERROR"
        assert assessment.judge_refused is False
        assert assessment.error and "429" in assessment.error

    def test_invalid_verdict_is_recorded_as_error(self):
        judge, _ = make_judge({"verdict": "DUNNO", "justification": "x"})
        assessment = judge.classify("Q", "A")
        assert assessment.verdict == "ERROR"
        assert "ValueError" in assessment.error

    def test_to_dict_shape(self):
        judge, _ = make_judge(JudgeVerdict(verdict="REFUSED", justification="why"))
        payload = judge.classify("Q", REFUSAL_TEXT).to_dict()

        assert set(payload) >= {
            "verdict", "justification", "judge_refused", "keyword_refused",
            "signals_agree", "judge_model", "judge_prompt_version",
        }


class TestCohensKappa:
    def test_perfect_agreement(self):
        labels = ["REFUSED", "ANSWERED", "REFUSED", "PARTIAL"]
        assert cohens_kappa(labels, list(labels)) == pytest.approx(1.0)

    def test_chance_level_agreement_is_near_zero(self):
        """Raw agreement flatters on skewed labels; kappa should not."""
        human = ["ANSWERED"] * 8 + ["REFUSED"] * 2
        judge = ["ANSWERED"] * 10   # always guesses the majority class
        assert sum(1 for a, b in zip(human, judge) if a == b) / len(human) == 0.8
        assert cohens_kappa(human, judge) == pytest.approx(0.0, abs=1e-9)

    def test_total_disagreement_is_negative(self):
        human = ["REFUSED", "ANSWERED", "REFUSED", "ANSWERED"]
        judge = ["ANSWERED", "REFUSED", "ANSWERED", "REFUSED"]
        assert cohens_kappa(human, judge) < 0

    def test_empty_input(self):
        assert cohens_kappa([], []) == 0.0


RESULTS_FIXTURE = {
    "results": [
        {
            "id": "q1",
            "question": "What risks, and how many, are there?",
            "with_context": {
                "answer": "The filings provided do not contain this information.",
                "refusal_assessment": {"verdict": "REFUSED",
                                       "justification": "Declines to answer."},
            },
            "no_context": {
                "answer": "Apple relies on suppliers.\n* Bullet, with comma\n* Another “smart quoted” line",
                "refusal_assessment": {"verdict": "ANSWERED",
                                       "justification": "Gives substance."},
            },
        }
    ]
}


def write_results(tmp_path, payload=None):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(payload or RESULTS_FIXTURE), encoding="utf-8")
    return path


def fill(csv_path, verdicts: dict):
    """Simulate a human filling in human_verdict, preserving everything else."""
    import csv as csvmod

    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csvmod.DictReader(handle))
        fields = list(rows[0].keys())
    for row in rows:
        if row["id"] in verdicts:
            row["human_verdict"] = verdicts[row["id"]]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csvmod.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class TestBlindTemplate:
    def test_row_count_and_ids(self, tmp_path):
        source = write_results(tmp_path)
        destination = tmp_path / "labels.csv"
        count = write_template(str(source), str(destination))

        rows = _read_labels(str(destination))
        assert count == len(rows) == 2
        assert {row["id"] for row in rows} == {"q1::with_context", "q1::no_context"}

    def test_judge_verdict_is_not_present_anywhere(self, tmp_path):
        """The whole point: a rater who sees the judge's answer anchors to it."""
        source = write_results(tmp_path)
        destination = tmp_path / "labels.csv"
        write_template(str(source), str(destination))

        raw = destination.read_text(encoding="utf-8-sig")
        assert "judge_verdict" not in raw
        assert "justification" not in raw
        assert "Declines to answer." not in raw
        assert "Gives substance." not in raw

        rows = _read_labels(str(destination))
        for row in rows:
            assert set(row) == set(TEMPLATE_COLUMNS)
            assert row["human_verdict"] == ""

    def test_rubric_rows_present_and_skipped(self, tmp_path):
        source = write_results(tmp_path)
        destination = tmp_path / "labels.csv"
        write_template(str(source), str(destination))

        raw = destination.read_text(encoding="utf-8-sig")
        for label, _ in RUBRIC:
            assert label in raw
        assert "request for input in place" in raw   # the edge-case rule
        assert len(_read_labels(str(destination))) == 2   # rubric rows skipped

    def test_arm_is_labelled_readably(self, tmp_path):
        source = write_results(tmp_path)
        destination = tmp_path / "labels.csv"
        write_template(str(source), str(destination))
        arms = {row["arm"] for row in _read_labels(str(destination))}
        assert arms == {"RAG", "no-context"}

    def test_answer_text_survives_csv_round_trip(self, tmp_path):
        """Answers contain newlines, commas, markdown bullets and smart quotes."""
        source = write_results(tmp_path)
        destination = tmp_path / "labels.csv"
        write_template(str(source), str(destination))

        rows = {row["id"]: row for row in _read_labels(str(destination))}
        original = RESULTS_FIXTURE["results"][0]["no_context"]["answer"]
        assert rows["q1::no_context"]["answer"] == original
        assert "\n" in original and "," in original and "“" in original

    def test_written_as_utf8_sig_for_excel(self, tmp_path):
        source = write_results(tmp_path)
        destination = tmp_path / "labels.csv"
        write_template(str(source), str(destination))
        assert destination.read_bytes().startswith(b"\xef\xbb\xbf")


class TestAgreementWorkflow:
    def test_perfect_agreement(self, tmp_path):
        source = write_results(tmp_path)
        labels = tmp_path / "labels.csv"
        write_template(str(source), str(labels))
        fill(labels, {"q1::with_context": "REFUSED", "q1::no_context": "ANSWERED"})

        summary = report_agreement(str(labels), str(source))
        assert summary["labelled_rows"] == 2
        assert summary["exact_agreement"] == 2
        assert summary["disagreements"] == []

    def test_disagreement_is_reported(self, tmp_path):
        source = write_results(tmp_path)
        labels = tmp_path / "labels.csv"
        write_template(str(source), str(labels))
        fill(labels, {"q1::with_context": "ANSWERED", "q1::no_context": "ANSWERED"})

        summary = report_agreement(str(labels), str(source))
        assert summary["exact_agreement"] == 1
        assert len(summary["disagreements"]) == 1
        item = summary["disagreements"][0]
        assert item["id"] == "q1::with_context"
        assert item["human"] == "ANSWERED" and item["judge"] == "REFUSED"

    def test_lowercase_labels_accepted(self, tmp_path):
        source = write_results(tmp_path)
        labels = tmp_path / "labels.csv"
        write_template(str(source), str(labels))
        fill(labels, {"q1::with_context": " refused ", "q1::no_context": "answered"})
        assert report_agreement(str(labels), str(source))["exact_agreement"] == 2

    def test_unlabelled_rows_are_skipped(self, tmp_path):
        source = write_results(tmp_path)
        labels = tmp_path / "labels.csv"
        write_template(str(source), str(labels))
        fill(labels, {"q1::with_context": "REFUSED"})
        assert report_agreement(str(labels), str(source))["labelled_rows"] == 1

    def test_no_labels_raises(self, tmp_path):
        source = write_results(tmp_path)
        labels = tmp_path / "labels.csv"
        write_template(str(source), str(labels))
        with pytest.raises(SystemExit):
            report_agreement(str(labels), str(source))

    def test_typo_in_verdict_raises_rather_than_miscounting(self, tmp_path):
        source = write_results(tmp_path)
        labels = tmp_path / "labels.csv"
        write_template(str(source), str(labels))
        fill(labels, {"q1::with_context": "REFUSE", "q1::no_context": "ANSWERED"})
        with pytest.raises(SystemExit) as excinfo:
            report_agreement(str(labels), str(source))
        assert "REFUSE" in str(excinfo.value)

    def test_small_sample_caveat_is_printed(self, tmp_path, capsys):
        source = write_results(tmp_path)
        labels = tmp_path / "labels.csv"
        write_template(str(source), str(labels))
        fill(labels, {"q1::with_context": "REFUSED", "q1::no_context": "ANSWERED"})
        report_agreement(str(labels), str(source))
        assert "CAVEAT" in capsys.readouterr().out


class TestKappaIsMulticlass:
    def test_matches_sklearn_on_three_classes(self):
        """Cohen's kappa here must be the general form, not a binary special case."""
        from sklearn.metrics import cohen_kappa_score

        human = ["REFUSED", "ANSWERED", "PARTIAL", "REFUSED", "ANSWERED",
                 "PARTIAL", "ANSWERED", "REFUSED"]
        judge = ["REFUSED", "ANSWERED", "ANSWERED", "PARTIAL", "ANSWERED",
                 "PARTIAL", "ANSWERED", "REFUSED"]
        assert cohens_kappa(human, judge) == pytest.approx(
            cohen_kappa_score(human, judge)
        )

    def test_matches_sklearn_on_two_classes(self):
        from sklearn.metrics import cohen_kappa_score

        human = ["REFUSED", "ANSWERED", "ANSWERED", "REFUSED", "ANSWERED"]
        judge = ["REFUSED", "ANSWERED", "REFUSED", "REFUSED", "ANSWERED"]
        assert cohens_kappa(human, judge) == pytest.approx(
            cohen_kappa_score(human, judge)
        )

    def test_all_three_classes_perfect(self):
        labels = ["REFUSED", "ANSWERED", "PARTIAL"] * 4
        assert cohens_kappa(labels, list(labels)) == pytest.approx(1.0)
