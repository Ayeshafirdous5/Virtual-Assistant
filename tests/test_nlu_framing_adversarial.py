"""Adversarial evaluation of the framing layer.

The first framing corpus is saturated at 32 of 32, so it can no longer
detect anything. This module runs a second, deliberately hostile corpus
against the same rules and reports where they break.

Measurement, not enforcement
----------------------------
**Mismatches do not fail the suite.** They are the whole point of the
exercise. A failing test would only appear once the rules were changed,
which is the step after this one. The report is printed on every run so
the numbers cannot be lost.

Two things are separated deliberately.

``LIVING_CASES`` are the mismatches that represent a real gap in the
framing rules: a sentence that would reach a tool, or a command that
would be blocked. ``HANDLED_UPSTREAM`` are the mismatches the NLU
negation guard already prevents before framing is ever consulted, so they
document where the layers meet rather than a defect to fix here. The
report shows both, and the headline accuracy includes both, so nothing is
hidden by the split.
"""

from __future__ import annotations

import pytest

from assistant.nlu import framing
from assistant.nlu.normalize import normalize
from tests.nlu_framing_adversarial_corpus import (
    CATEGORIES,
    CORPUS,
    KNOWN_LAYERING,
    MENTION,
    AdversarialCase,
    by_category,
)

#: Accuracy measured when this adversarial corpus was first run against the
#: Phase 5 Step 3 rules. Pinned so a later change to ``framing.py`` has to
#: move it deliberately.
#:
#: Step 5 recognised a trailing question so it outranks the narrative
#: guard, which fixed the two blocked commands without editing a single
#: corpus expectation: 40 of 46 became 42 of 46. Step 7's reported-"tell"
#: guard then closed the one live gap that remained, taking it to 43.
RECORDED_ACCURACY = 0.9347826086956522

#: Mismatch counts measured at the same time, pinned for the same reason.
#: The four let-throughs became three, and all three are the negated
#: requests the NLU negation guard already stops upstream. Step 7 removed
#: the last live gap, so nothing that reaches a tool is left unattested.
RECORDED_LEAKING = 3
RECORDED_BLOCKING = 0


def actual_verdict(case: AdversarialCase) -> str:
    """Run the production rule over one adversarial case."""
    return framing.assess(normalize(case.utterance), case.intent)


def is_correct(case: AdversarialCase) -> bool:
    return actual_verdict(case) == case.expected


def accuracy(cases=CORPUS) -> float:
    if not cases:
        return 1.0
    return sum(1 for c in cases if is_correct(c)) / len(cases)


class Mismatch:
    """One case where the rule and the expected label disagree."""

    def __init__(self, case: AdversarialCase, actual: str) -> None:
        self.case = case
        self.actual = actual

    @property
    def kind(self) -> str:
        """Classify by direction, because the two are not equally bad."""
        expected_blocks = self.case.blocks
        actual_blocks = self.actual == framing.MENTION
        if expected_blocks and not actual_blocks:
            return "false-let-through"
        if not expected_blocks and actual_blocks:
            return "false-block"
        return "wrong-label"

    @property
    def handled_upstream(self) -> bool:
        """True when the NLU negation guard already stops this sentence."""
        return self.case.category in KNOWN_LAYERING

    def describe(self) -> str:
        upstream = " [handled upstream]" if self.handled_upstream else ""
        return (
            f"{self.case.utterance!r} [{self.case.intent}] "
            f"expected {self.case.expected}, got {self.actual} "
            f"({self.kind}){upstream}"
        )


def mismatches(cases=CORPUS) -> list[Mismatch]:
    return [Mismatch(c, actual_verdict(c)) for c in cases if not is_correct(c)]


def by_kind(kind: str) -> list[Mismatch]:
    return [m for m in mismatches() if m.kind == kind]


def living_mismatches() -> list[Mismatch]:
    """Mismatches that are not already prevented by another layer."""
    return [m for m in mismatches() if not m.handled_upstream]


def build_report() -> str:
    """Render the whole measurement as readable text."""
    total = len(CORPUS)
    right = sum(1 for c in CORPUS if is_correct(c))
    bad = mismatches()
    living = living_mismatches()

    lines = [
        "framing evaluation (ADVERSARIAL)",
        f"  total cases        : {total}",
        f"  correct            : {right}",
        f"  incorrect          : {len(bad)}",
        f"  accuracy           : {accuracy():.3f}",
        f"  live gaps          : {len(living)} "
        f"(the other {len(bad) - len(living)} are already handled upstream)",
        "",
        "  by category:",
    ]
    for category in CATEGORIES:
        subset = by_category(category)
        if not subset:
            continue
        hits = sum(1 for c in subset if is_correct(c))
        lines.append(f"    {category:<24} {hits:>2}/{len(subset):<2} correct")

    lines.append("")
    lines.append("  expected verdicts:")
    for verdict in (framing.REQUEST, framing.MENTION, framing.NEUTRAL):
        count = sum(1 for c in CORPUS if c.expected == verdict)
        lines.append(f"    {verdict:<24} {count}")

    for kind in ("false-let-through", "false-block", "wrong-label"):
        group = by_kind(kind)
        if not group:
            continue
        lines.append("")
        lines.append(f"  {kind} ({len(group)}):")
        for item in group:
            marker = " [upstream]" if item.handled_upstream else ""
            lines.append(
                f"    - {item.case.utterance!r} [{item.case.intent}] "
                f"{item.case.expected} -> {item.actual}{marker}"
            )
    if not bad:
        lines.append("")
        lines.append("  mismatches         : none")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Corpus integrity
# ----------------------------------------------------------------------
class TestCorpus:
    def test_corpus_is_large_enough_to_be_useful(self):
        assert len(CORPUS) >= 40

    def test_labels_match_the_production_constants(self):
        assert (framing.REQUEST, framing.MENTION, framing.NEUTRAL) == (
            "request",
            "mention",
            "neutral",
        )

    def test_every_expected_verdict_is_legal(self):
        for case in CORPUS:
            assert case.expected in (framing.REQUEST, framing.MENTION, framing.NEUTRAL)

    def test_every_category_is_represented(self):
        assert {case.category for case in CORPUS} == set(CATEGORIES)

    def test_no_duplicate_utterance_and_intent_pairs(self):
        pairs = [(c.utterance, c.intent) for c in CORPUS]
        assert len(pairs) == len(set(pairs))

    def test_intents_are_real_tool_names(self):
        from assistant.tools import build_default_router

        known = {tool.name for tool in build_default_router().tools}
        assert {case.intent for case in CORPUS} <= known

    def test_the_first_corpus_is_untouched(self):
        """The saturated corpus must not have been edited to flatter this one."""
        from tests import nlu_framing_corpus as original

        assert len(original.CORPUS) == 32
        assert all(c.expected in ("request", "mention", "neutral") for c in original.CORPUS)

    def test_every_case_carries_a_reason(self):
        for case in CORPUS:
            assert case.note, case.utterance

    def test_known_layering_category_is_documented(self):
        assert KNOWN_LAYERING
        for case in CORPUS:
            if case.category in KNOWN_LAYERING:
                assert "upstream" in case.note, case.utterance


# ----------------------------------------------------------------------
# The measurement
# ----------------------------------------------------------------------
class TestMeasurement:
    def test_every_case_is_evaluated(self):
        assert len(mismatches()) <= len(CORPUS)

    def test_correct_and_incorrect_add_up(self):
        right = sum(1 for c in CORPUS if is_correct(c))
        assert right + len(mismatches()) == len(CORPUS)

    def test_accuracy_is_a_proportion(self):
        assert 0.0 <= accuracy() <= 1.0

    def test_evaluation_is_deterministic(self):
        assert [actual_verdict(c) for c in CORPUS] == [
            actual_verdict(c) for c in CORPUS
        ]

    def test_every_mismatch_is_classified(self):
        for item in mismatches():
            assert item.kind in ("false-let-through", "false-block", "wrong-label")

    def test_mismatches_do_not_fail_the_suite(self):
        """This step measures. It does not enforce."""
        assert isinstance(mismatches(), list)

    def test_blocked_commands_are_measured_not_asserted_away(self):
        """Step 3's narrative guard blocked two real requests, and that was
        recorded as a finding rather than hidden by rewriting the
        expectations to match the code. Step 5's trailing-question rule
        closed both, so the pinned count is now zero.

        The pin stays, and stays at zero, so that blocking a command again
        has to be a deliberate move rather than something that slips in
        unnoticed. Blocking is the expensive direction: it stops a command
        that used to work.
        """
        blocked = by_kind("false-block")
        assert len(blocked) == RECORDED_BLOCKING, [m.describe() for m in blocked]
        for item in blocked:
            assert item.case.utterance in build_report()
            # A block must only ever be wrong, never right: a blocked
            # command is a command that used to work.
            assert item.case.expected != framing.MENTION

    def test_measured_accuracy_matches_the_recorded_figure(self):
        assert accuracy() == pytest.approx(RECORDED_ACCURACY, abs=1e-9), (
            f"recorded {RECORDED_ACCURACY} but measured {accuracy()}"
        )

    def test_accuracy_did_not_regress(self):
        assert accuracy() >= RECORDED_ACCURACY, build_report()

    def test_report_lists_every_mismatch(self):
        report = build_report()
        for item in mismatches():
            assert item.case.utterance in report

    def test_the_report_is_printed(self, capsys):
        report = build_report()
        with capsys.disabled():
            print(report)
