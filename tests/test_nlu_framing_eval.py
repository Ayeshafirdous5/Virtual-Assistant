"""Evaluation of the framing layer against its measurement corpus.

This step is **measurement only**. It records what the current rules
actually do on a broader set of utterances, so a later step can change
them with evidence rather than by feel.

Two design choices matter here.

First, **mismatches do not fail the suite.** They are the finding, not a
defect in the test. A failure would only be reported once the rules were
changed, which would hide the very thing this step exists to capture.
The report below is printed on every run so the numbers cannot be lost.

Second, the headline metric is not the three-way label but whether a case
**blocks**, because ``MENTION`` stops a command while ``REQUEST`` and
``NEUTRAL`` both let it through. A case expected to be blocked and not
blocked is a remark reaching a tool; the reverse is a broken command.
Those two are counted separately because they are not equally bad.
"""

from __future__ import annotations

import pytest

from assistant.nlu import framing
from assistant.nlu.normalize import normalize
from tests.nlu_framing_corpus import (
    CATEGORIES,
    CATEGORY_COMPETING,
    CATEGORY_REMARK,
    CORPUS,
    MENTION,
    FramingCase,
    by_category,
)

#: The accuracy measured when this corpus was first run against the
#: Phase 5 Step 1 rules: 0.875, with four remarks leaking through. After
#: the Step 3 narrative guard the corpus is fully correct. Pinned so that
#: any later change to ``framing.py`` has to be deliberate: it moves this
#: number and the suite says so.
RECORDED_ACCURACY = 1.0

#: How many remarks were measured as reaching a tool. This is the costliest
#: error direction: the user was talking, and a tool ran anyway. Was 4
#: before the Step 3 narrative guard.
RECORDED_LEAKING = 0


def actual_verdict(case: FramingCase) -> str:
    """Run the production rule over one corpus case."""
    return framing.assess(normalize(case.utterance), case.intent)


def is_correct(case: FramingCase) -> bool:
    return actual_verdict(case) == case.expected


def blocks(verdict: str) -> bool:
    """True when a verdict would stop the command running."""
    return verdict == framing.MENTION


class Mismatch:
    """One case where the rule and the expected label disagree."""

    def __init__(self, case: FramingCase, actual: str) -> None:
        self.case = case
        self.actual = actual

    @property
    def kind(self) -> str:
        """Classify the failure by the direction that matters.

        ``false-let-through`` is the expensive one: a remark that was meant
        to be blocked, and was not.
        """
        expected_blocks = self.case.blocks
        actual_blocks = blocks(self.actual)
        if expected_blocks and not actual_blocks:
            return "false-let-through"
        if not expected_blocks and actual_blocks:
            return "false-block"
        return "wrong-label"

    def describe(self) -> str:
        return (
            f"{self.case.utterance!r} [{self.case.intent}] "
            f"expected {self.case.expected}, got {self.actual} "
            f"({self.kind})"
        )


def mismatches(cases=CORPUS) -> list[Mismatch]:
    return [Mismatch(c, actual_verdict(c)) for c in cases if not is_correct(c)]


def accuracy(cases=CORPUS) -> float:
    if not cases:
        return 1.0
    return sum(1 for c in cases if is_correct(c)) / len(cases)


def build_report(cases=CORPUS) -> str:
    """Render the whole measurement as readable text."""
    total = len(cases)
    right = sum(1 for c in cases if is_correct(c))
    bad = mismatches(cases)

    lines = [
        "framing evaluation",
        f"  total cases        : {total}",
        f"  correct            : {right}",
        f"  incorrect          : {len(bad)}",
        f"  accuracy           : {accuracy(cases):.3f}",
        "",
        "  by category:",
    ]
    for category in CATEGORIES:
        subset = by_category(category) if cases is CORPUS else [
            c for c in cases if c.category == category
        ]
        if not subset:
            continue
        hits = sum(1 for c in subset if is_correct(c))
        lines.append(
            f"    {category:<18} {hits:>2}/{len(subset):<2} correct"
        )

    lines.append("")
    lines.append("  expected verdicts:")
    for verdict in (framing.REQUEST, framing.MENTION, framing.NEUTRAL):
        count = sum(1 for c in cases if c.expected == verdict)
        lines.append(f"    {verdict:<18} {count}")

    if bad:
        lines.append("")
        lines.append("  mismatches by kind:")
        for kind in ("false-let-through", "false-block", "wrong-label"):
            group = [m for m in bad if m.kind == kind]
            if group:
                lines.append(f"    {kind} ({len(group)}):")
                for item in group:
                    lines.append(f"      - {item.describe()}")
    else:
        lines.append("")
        lines.append("  mismatches         : none")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Corpus integrity
# ----------------------------------------------------------------------
class TestCorpus:
    def test_corpus_is_populated(self):
        assert len(CORPUS) >= 30

    def test_labels_match_the_production_constants(self):
        # The corpus uses plain strings; they must mean the same thing.
        assert (framing.REQUEST, framing.MENTION, framing.NEUTRAL) == (
            "request",
            "mention",
            "neutral",
        )

    def test_every_expected_verdict_is_legal(self):
        for case in CORPUS:
            assert case.expected in (framing.REQUEST, framing.MENTION, framing.NEUTRAL)

    def test_every_category_is_represented(self):
        used = {case.category for case in CORPUS}
        assert used == set(CATEGORIES)

    def test_no_duplicate_utterance_and_intent_pairs(self):
        pairs = [(c.utterance, c.intent) for c in CORPUS]
        assert len(pairs) == len(set(pairs))

    def test_every_case_is_annotated(self):
        for case in CORPUS:
            assert case.intent, case.utterance
        # A label needs explaining only where it could be argued with.
        for case in CORPUS:
            if case.category in (CATEGORY_REMARK, CATEGORY_COMPETING):
                assert case.note, case.utterance

    def test_intents_are_real_tool_names(self):
        from assistant.tools import build_default_router

        known = {tool.name for tool in build_default_router().tools}
        assert {case.intent for case in CORPUS} <= known

    def test_each_verdict_is_well_represented(self):
        counts = {
            verdict: sum(1 for c in CORPUS if c.expected == verdict)
            for verdict in (framing.REQUEST, framing.MENTION, framing.NEUTRAL)
        }
        # A corpus with only one expected verdict could not measure anything.
        assert all(count > 0 for count in counts.values()), counts


# ----------------------------------------------------------------------
# The measurement
# ----------------------------------------------------------------------
class TestMeasurement:
    def test_every_case_is_evaluated(self):
        assert len(mismatches(CORPUS)) <= len(CORPUS)

    def test_correct_and_incorrect_add_up(self):
        right = sum(1 for c in CORPUS if is_correct(c))
        assert right + len(mismatches(CORPUS)) == len(CORPUS)

    def test_accuracy_is_a_proportion(self):
        value = accuracy()
        assert 0.0 <= value <= 1.0
        assert value == pytest.approx(
            sum(1 for c in CORPUS if is_correct(c)) / len(CORPUS)
        )

    def test_evaluation_is_deterministic(self):
        assert [actual_verdict(c) for c in CORPUS] == [
            actual_verdict(c) for c in CORPUS
        ]

    def test_report_mentions_every_case(self):
        assert str(len(CORPUS)) in build_report()

    def test_report_lists_every_mismatch(self):
        report = build_report()
        for item in mismatches():
            assert item.case.utterance in report, item.case.utterance

    def test_measured_accuracy_has_not_regressed(self):
        """A floor, so a future change cannot quietly make framing worse."""
        current = accuracy()
        assert current >= RECORDED_ACCURACY, (
            f"framing accuracy fell from {RECORDED_ACCURACY:.3f} "
            f"to {current:.3f}\n{build_report()}"
        )

    def test_the_report_is_printed(self, capsys):
        """Surfaces the full report so the numbers are visible on every run.

        ``capsys`` captures by default, which would swallow the output
        entirely, so capturing is disabled for the duration. The report then
        reaches the real terminal even without ``-s``.
        """
        report = build_report()
        with capsys.disabled():
            print(report)

    def test_measured_accuracy_matches_the_recorded_figure(self):
        """The recorded number must be the real one, not a stale guess."""
        assert accuracy() == pytest.approx(RECORDED_ACCURACY, abs=1e-9), (
            f"recorded {RECORDED_ACCURACY} but measured {accuracy()}; "
            "update RECORDED_ACCURACY if the rules intentionally changed"
        )

    def test_every_mismatch_is_classified(self):
        for item in mismatches():
            assert item.kind in ("false-let-through", "false-block", "wrong-label")

    def test_no_expensive_failure_direction_is_worse_than_recorded(self):
        """A remark leaking through to a tool is the costliest error."""
        leaking = [m for m in mismatches() if m.kind == "false-let-through"]
        assert len(leaking) <= RECORDED_LEAKING, (
            f"{len(leaking)} remarks now reach a tool\n{build_report()}"
        )

    def test_no_command_is_blocked_at_all(self):
        """A false block breaks a working command, so there must be none."""
        blocked = [m for m in mismatches() if m.kind == "false-block"]
        assert not blocked, [m.describe() for m in blocked]

    def test_the_corpus_is_fully_correct(self):
        """The Step 3 acceptance criterion: 32 of 32, nothing leaking."""
        bad = mismatches()
        assert not bad, [m.describe() for m in bad]
