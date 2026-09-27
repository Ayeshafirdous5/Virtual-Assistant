"""Measurement and diagnosis of the parser-layer gaps.

This step is **measurement only**. No production code is changed, and
nothing here proposes an edit.

The question
------------
Eleven framing steps kept meeting the same thing from the far side: a
sentence that framing never sees, because the parser returned ``None``
first. Two corpora still carry such cases. This module gathers all of
them, adds neighbours for contrast, and answers three questions:

1. What does the parser **actually** do with each case, with evidence?
2. Is the behaviour a **gap**, or a safeguard working?
3. If it is a gap, would closing it be **safe**, or would it mean
   guessing at what the user meant?

The third question is the one that matters. A sentence the parser cannot
understand is not automatically a sentence it should understand. Most of
the cases here name no action and no topic, and a parser that answered
them would be guessing. Those are recorded as
:data:`~tests.nlu_parser_gap_corpus.SHOULD_STAY_UNSUPPORTED`, and
counting them as fixable would be the wrong conclusion.

Evidence, not inference
-----------------------
Every diagnosis is read off the parser's own output, not guessed from
the sentence:

``score_intents``
  what the scoring layer produced, before the action boundary is applied.
  A case that produced *nothing* has a different cause from one that
  produced a candidate and was then held back.
``parse``
  what survived ``MIN_CONFIDENCE``, and whether the margin made it
  ambiguous.
the constants themselves
  ``CONTAINMENT_SCORE`` sits below ``MIN_CONFIDENCE`` by design, so a
  containment-only candidate is a refusal, not a miss.

Measurement, not enforcement
----------------------------
**Unexpected results do not fail the suite.** They are the point. The
counts are pinned so a later change to the parser has to move them
deliberately, in either direction.
"""

from __future__ import annotations

import functools

import pytest

from assistant.nlu.normalize import normalize
from assistant.nlu.parser import parse
from assistant.nlu.scoring import score_intents
from tests.nlu_parser_gap_corpus import (
    CAUSES,
    CATEGORIES,
    CAUSE_MISSING_ALIAS,
    CAUSE_NONE,
    CORPUS,
    MATCH,
    NO_MATCH,
    SAFE_TO_IMPROVE,
    SHOULD_STAY_UNSUPPORTED,
    ParserGapCase,
    by_category,
)

#: Size of the corpus, pinned so cases cannot be dropped.
RECORDED_CORPUS_SIZE = 47

#: Counts measured when this corpus was first run, against the parser as
#: it stands at the end of Phase 5 Step 11. Pinned so a later change to
#: the parser has to move them deliberately.
RECORDED_CORRECT = 47
RECORDED_UNEXPECTED = 0
RECORDED_UNEXPECTED_NO_MATCH = 0
RECORDED_UNEXPECTED_MATCH = 0
RECORDED_WRONG_INTENT = 0
RECORDED_AMBIGUOUS = 0

#: How the gaps are judged. Only a missing alias is safe to close; the
#: rest would mean guessing or undoing a safeguard. Fifteen are safe and
#: sixteen must stay, which is the finding this step exists to produce:
#: **most parser no-matches are correct behaviour**.
RECORDED_SAFE_TO_IMPROVE = 15
RECORDED_SHOULD_STAY_UNSUPPORTED = 16


@functools.lru_cache(maxsize=1)
def _lexicon():
    """The runtime lexicon, built once. The real one, not a stub."""
    from assistant.app import build_runtime_lexicon
    from assistant.tools import build_default_router

    return build_runtime_lexicon(build_default_router())


def actual_outcome(utterance: str) -> str:
    """What the real parser returns, as one of the three outcomes."""
    parsed = parse(utterance, _lexicon())
    if parsed is None:
        return NO_MATCH
    if parsed.is_ambiguous:
        return "ambiguous"
    return MATCH


def actual_intent(utterance: str) -> str:
    """The intent the parser chose, or ``""`` when it chose nothing."""
    parsed = parse(utterance, _lexicon())
    return parsed.name if parsed is not None else ""


def candidates(utterance: str):
    """What the scoring layer produced, before the action boundary.

    This is the evidence the diagnoses are read from. A case that
    produced nothing has a different cause from one that produced a
    candidate and was then held back.
    """
    return score_intents(normalize(utterance), _lexicon())


#: Result labels, named so a report reads as a sentence.
RESULT_CORRECT = "correct"
RESULT_UNEXPECTED_NO_MATCH = "unexpected-no-match"
RESULT_UNEXPECTED_MATCH = "unexpected-match"
RESULT_WRONG_INTENT = "wrong-intent"
RESULT_AMBIGUOUS = "ambiguous-unexpected"

RESULTS = (
    RESULT_CORRECT,
    RESULT_UNEXPECTED_NO_MATCH,
    RESULT_UNEXPECTED_MATCH,
    RESULT_WRONG_INTENT,
    RESULT_AMBIGUOUS,
)


class Result:
    """One case, what the parser did, and how that compares."""

    def __init__(self, case: ParserGapCase) -> None:
        self.case = case
        self.outcome = actual_outcome(case.utterance)
        self.intent = actual_intent(case.utterance)
        self.candidates = candidates(case.utterance)
        self.top = self.candidates[0] if self.candidates else None

    @property
    def result(self) -> str:
        """Classify the outcome against what the case expected."""
        if self.outcome != self.case.expected:
            if self.outcome == NO_MATCH:
                return RESULT_UNEXPECTED_NO_MATCH
            if self.outcome == "ambiguous":
                return RESULT_AMBIGUOUS
            return RESULT_UNEXPECTED_MATCH
        if self.outcome == MATCH and self.intent != self.case.intent:
            return RESULT_WRONG_INTENT
        return RESULT_CORRECT

    @property
    def is_unexpected(self) -> bool:
        return self.result != RESULT_CORRECT

    @property
    def evidence(self) -> str:
        """What the scoring layer actually produced, for the report."""
        if not self.candidates:
            return "no candidate at any stage"
        return " | ".join(
            f"{c.intent} {c.method} {c.score:.2f}" for c in self.candidates[:3]
        )

    def describe(self) -> str:
        return (
            f"{self.case.utterance!r} expected {self.case.expected}, "
            f"got {self.outcome} [{self.intent or '-'}] "
            f"({self.result}; {self.case.cause})"
        )


def results(cases=CORPUS) -> list[Result]:
    return [Result(c) for c in cases]


def by_result(label: str, cases=CORPUS) -> list[Result]:
    return [r for r in results(cases) if r.result == label]


def unexpected(cases=CORPUS) -> list[Result]:
    return [r for r in results(cases) if r.is_unexpected]


def gaps(cases=CORPUS) -> list[Result]:
    """Cases the parser does not do what the corpus expects."""
    return [r for r in results(cases) if r.case.is_gap]


def build_report() -> str:
    """Render the whole measurement as readable text."""
    all_results = results()
    bad = unexpected()

    lines = [
        "parser-gap evaluation",
        "  (MEASUREMENT AND DIAGNOSIS ONLY - no production change)",
        f"  total cases        : {len(CORPUS)}",
        f"  as expected        : {len(CORPUS) - len(bad)}",
        f"  unexpected         : {len(bad)}",
        "",
        "  the headline split:",
    ]
    with_trigger = [r for r in gaps() if r.case.cause == CAUSE_NONE]
    lines += [
        "    A parser gap is not automatically a bug. The corpus records",
        "    what a human would expect; the cause column records why the",
        "    parser does something else, and whether that is defensible.",
        "",
        f"    expected but not delivered : {len(gaps())}",
        f"    safe to close with an alias : {len([r for r in gaps() if r.case.cause in SAFE_TO_IMPROVE])}",
        f"    should stay unsupported     : {len([r for r in gaps() if r.case.cause in SHOULD_STAY_UNSUPPORTED])}",
        "",
        "  outcomes:",
    ]
    for label in (MATCH, "ambiguous", NO_MATCH):
        count = sum(1 for r in all_results if r.outcome == label)
        lines.append(f"    {label:<24} {count}")

    lines += ["", "  by category:"]
    for category in CATEGORIES:
        subset = by_category(category)
        if not subset:
            continue
        hits = sum(1 for c in subset if not Result(c).is_unexpected)
        lines.append(f"    {category:<24} {hits:>2}/{len(subset):<2} as expected")

    lines += ["", "  likely causes, grouped:"]
    seen_causes = sorted({r.case.cause for r in gaps()})
    for cause in seen_causes:
        group = [r for r in gaps() if r.case.cause == cause]
        lines.append("")
        lines.append(f"  {cause} ({len(group)}):")
        lines.append(f"    {CAUSES[cause]}")
        for item in group:
            lines.append(f"    - {item.case.utterance!r}")
            lines.append(f"        evidence: {item.evidence}")

    lines += ["", "  unexpected results:"]
    if not bad:
        lines.append("    none")
    for item in bad:
        lines.append(f"    - {item.describe()}")

    lines += [
        "",
        "  framing reachability:",
        "    Framing is consulted only after a CLEAR parse. A no-match and",
        "    an ambiguous result both return before it, so neither is a",
        "    framing finding.",
    ]
    reachable = [r for r in all_results if r.outcome == MATCH]
    lines.append(f"    cases where framing runs : {len(reachable)}")
    lines.append(f"    cases where it does not   : {len(all_results) - len(reachable)}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Corpus integrity
# ----------------------------------------------------------------------
class TestCorpus:
    def test_corpus_is_large_enough_to_be_useful(self):
        assert len(CORPUS) >= 30

    def test_corpus_size_is_pinned(self):
        assert len(CORPUS) == RECORDED_CORPUS_SIZE

    def test_every_expected_outcome_is_legal(self):
        for case in CORPUS:
            assert case.expected in (MATCH, "ambiguous", NO_MATCH)

    def test_every_category_is_represented(self):
        assert {case.category for case in CORPUS} == set(CATEGORIES)

    def test_every_category_is_non_trivial(self):
        for category in CATEGORIES:
            assert len(by_category(category)) >= 5, category

    def test_no_duplicate_utterances(self):
        utterances = [c.utterance for c in CORPUS]
        assert len(utterances) == len(set(utterances))

    def test_intents_are_real_tool_names(self):
        from assistant.tools import build_default_router

        known = {tool.name for tool in build_default_router().tools}
        assert {case.intent for case in CORPUS} <= known

    def test_every_case_carries_a_reason(self):
        for case in CORPUS:
            assert case.note, case.utterance

    def test_every_cause_is_a_documented_cause(self):
        for case in CORPUS:
            assert case.cause in CAUSES, case.cause

    def test_every_cause_is_judged_one_way_or_the_other(self):
        """A cause nobody has ruled on is not a diagnosis yet."""
        judged = SAFE_TO_IMPROVE | SHOULD_STAY_UNSUPPORTED
        for case in CORPUS:
            if case.cause == CAUSE_NONE:
                continue
            assert case.cause in judged, case.cause

    def test_every_carried_over_case_records_where_it_came_from(self):
        """Anything lifted from a framing corpus must name its origin.

        These six are the sentences eleven framing steps kept running
        into, so losing the provenance would lose the reason the corpus
        exists. The two controls beside them are new and carry no
        source, which is why this counts sources rather than the whole
        category.
        """
        carried = [c for c in CORPUS if c.source]
        assert len(carried) == 6, [c.utterance for c in carried]
        for case in carried:
            assert "corpus" in case.source, case.source

    def test_no_case_claims_a_source_it_does_not_have(self):
        """A source is only for a case that really came from one."""
        for case in CORPUS:
            if case.source:
                assert "corpus" in case.source, case.source
                assert case.category == "verb-form-gap", case.utterance

    def test_the_earlier_corpora_are_untouched(self):
        from tests import nlu_framing_adversarial_corpus as adversarial
        from tests import nlu_framing_corpus as original
        from tests import nlu_framing_tell_corpus as tell
        from tests import nlu_framing_telling_corpus as telling
        from tests import nlu_framing_verb_forms_corpus as verb_forms

        assert len(original.CORPUS) == 32
        assert len(adversarial.CORPUS) == 46
        assert len(tell.CORPUS) == 57
        assert len(verb_forms.CORPUS) == 49
        assert len(telling.CORPUS) == 41


# ----------------------------------------------------------------------
# The production facts the diagnosis rests on
# ----------------------------------------------------------------------
class TestProductionFacts:
    """Pinned so a diagnosis cannot quietly become untrue."""

    def test_the_thresholds_are_what_the_report_says(self):
        from assistant.nlu.parser import AMBIGUITY_MARGIN, MIN_CONFIDENCE
        from assistant.nlu.scoring import CONTAINMENT_SCORE

        assert MIN_CONFIDENCE == 0.65
        assert AMBIGUITY_MARGIN == 0.05

    def test_containment_sits_below_the_action_boundary(self):
        """The reason a containment hit is a refusal, not a miss."""
        from assistant.nlu.parser import MIN_CONFIDENCE
        from assistant.nlu.scoring import CONTAINMENT_SCORE

        assert CONTAINMENT_SCORE < MIN_CONFIDENCE

    def test_framing_runs_only_after_a_clear_parse(self):
        """A no-match and an ambiguous result both return before it.

        This is the fact the whole corpus is organised around, and it
        is why an ambiguous case is recorded with framing_relevant false.
        """
        from assistant.app import STATUS_AMBIGUOUS, STATUS_NO_MATCH, resolve_detail
        from assistant.core.context import AppContext
        from assistant.tools import build_default_router

        class Silent:
            def speak(self, text: str) -> None:
                pass

            def listen(self) -> str:
                return ""

        ctx = AppContext()
        ctx.speaker = Silent()
        ctx.listener = Silent()
        router = build_default_router()
        lexicon = _lexicon()

        no_match = resolve_detail(ctx, router, "quite good", lexicon)
        assert no_match.status == STATUS_NO_MATCH
        assert no_match.framing == ""

        # The ambiguous path asks the user, so it needs a real speaker.
        ambiguous = resolve_detail(
            ctx, router, "I heard the news, tell me another joke", lexicon
        )
        assert ambiguous.status == STATUS_AMBIGUOUS
        assert ambiguous.framing == ""

        # And a clear parse is the only path that reaches it.
        clear = resolve_detail(ctx, router, "tell me a joke", lexicon)
        assert clear.framing == "request"


# ----------------------------------------------------------------------
# The measurement
# ----------------------------------------------------------------------
class TestMeasurement:
    def test_every_case_is_evaluated(self):
        assert len(results()) == len(CORPUS)

    def test_evaluation_is_deterministic(self):
        assert [actual_outcome(c.utterance) for c in CORPUS] == [
            actual_outcome(c.utterance) for c in CORPUS
        ]

    def test_every_result_is_classified(self):
        for item in results():
            assert item.result in RESULTS

    def test_mismatches_do_not_fail_the_suite(self):
        """This step measures. It does not enforce."""
        assert isinstance(unexpected(), list)

    def test_the_measured_counts_still_hold(self):
        """Pinned so a future parser change has to be deliberate."""
        assert len(results()) - len(unexpected()) == RECORDED_CORRECT
        assert len(unexpected()) == RECORDED_UNEXPECTED
        assert len(by_result(RESULT_UNEXPECTED_NO_MATCH)) == (
            RECORDED_UNEXPECTED_NO_MATCH
        )
        assert len(by_result(RESULT_UNEXPECTED_MATCH)) == RECORDED_UNEXPECTED_MATCH
        assert len(by_result(RESULT_WRONG_INTENT)) == RECORDED_WRONG_INTENT
        assert len(by_result(RESULT_AMBIGUOUS)) == RECORDED_AMBIGUOUS

    def test_the_judgement_counts_still_hold(self):
        """The point of the step: how many gaps are worth closing."""
        safe = [r for r in gaps() if r.case.cause in SAFE_TO_IMPROVE]
        keep = [r for r in gaps() if r.case.cause in SHOULD_STAY_UNSUPPORTED]
        assert len(safe) == RECORDED_SAFE_TO_IMPROVE
        assert len(keep) == RECORDED_SHOULD_STAY_UNSUPPORTED
        assert len(safe) + len(keep) == len(gaps())

    def test_a_safeguard_is_never_called_fixable(self):
        """The containment floor and the fuzzy guard are not bugs."""
        from tests.nlu_parser_gap_corpus import (
            CAUSE_CONTAINMENT_FLOOR,
            CAUSE_TYPO_APPEND,
        )

        for cause in (CAUSE_CONTAINMENT_FLOOR, CAUSE_TYPO_APPEND):
            assert cause not in SAFE_TO_IMPROVE
            assert cause in SHOULD_STAY_UNSUPPORTED

    def test_a_containment_case_really_did_produce_a_candidate(self):
        """Evidence, not inference: the diagnosis is checked, not assumed."""
        for item in gaps():
            if item.case.cause != "intentional-safety-lower-containment-below-action-boundary":
                continue
            assert item.top is not None, item.case.utterance
            assert item.top.method == "containment"

    def test_a_no_trigger_case_really_produced_nothing(self):
        for item in gaps():
            if item.case.cause != "genuinely-unsupported-wording":
                continue
            assert not item.candidates, item.case.utterance

    def test_the_verb_form_control_proves_the_cause(self):
        """The same verb form parses when a trigger word is present.

        This is the evidence that the eight framing-corpus gaps are not
        about "telling" at all. Without this, a future reader could
        reasonably assume a gerund rule would fix them.
        """
        no_trigger = Result(
            next(c for c in CORPUS if c.utterance == "I liked you telling that story")
        )
        with_trigger = Result(
            next(c for c in CORPUS if c.utterance == "I heard you telling a joke")
        )
        assert not no_trigger.candidates
        assert with_trigger.outcome == MATCH
        assert with_trigger.intent == "jokes"

    def test_every_gap_is_in_a_group_that_can_be_judged(self):
        """No gap may be left unclassified by cause."""
        for item in gaps():
            assert item.case.cause != CAUSE_NONE

    def test_the_ambiguous_case_never_reaches_framing(self):
        ambiguous = next(c for c in CORPUS if c.expected == "ambiguous")
        assert ambiguous.framing_relevant is False
        result = Result(ambiguous)
        assert result.outcome == "ambiguous"
        assert len(result.candidates) >= 2

    def test_only_clear_matches_can_reach_framing(self):
        for case in CORPUS:
            if case.expected != MATCH:
                assert case.framing_relevant is False, case.utterance

    def test_the_report_groups_by_cause(self):
        report = build_report()
        for cause in sorted({r.case.cause for r in gaps()}):
            assert cause in report, cause

    def test_the_report_names_every_gap(self):
        report = build_report()
        for item in gaps():
            assert item.case.utterance in report, item.case.utterance

    def test_the_report_states_the_judgement(self):
        report = build_report()
        assert "safe to close with an alias" in report
        assert "should stay unsupported" in report
        assert "expected but not delivered" in report

    def test_the_report_explains_framing_reachability(self):
        report = build_report()
        assert "only after a CLEAR parse" in report
        assert "framing reachability" in report

    def test_the_report_is_printed(self, capsys):
        report = build_report()
        with capsys.disabled():
            print(report)

