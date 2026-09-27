"""Measurement of the ``tell`` distinction in the framing layer.

This step is **measurement only**. No production rule is changed here, and
nothing in this module asserts that the current behaviour is correct.

What is being measured
----------------------
``tell`` is in :data:`~assistant.nlu.framing.REQUEST_CUES`, and that list
is position independent, so a sentence is a request whenever it contains
the word at all. One fact therefore does two opposite jobs:

* it carries the largest family of real commands ("tell me a joke"), and
* it makes every sentence that only *reports* speech ("I heard you tell
  a joke") look like a request.

The second job is the one remaining live framing gap. This corpus
(:mod:`tests.nlu_framing_tell_corpus`) isolates that word so the size of
the problem, and the shape of any future fix, can be judged on numbers.

Measurement, not enforcement
----------------------------
**Mismatches do not fail the suite.** They are the whole point of the
step. The numbers are printed on every run and pinned below, so a later
change to ``framing.py`` has to move them deliberately rather than
quietly.

The most important line in the report is the blocking metric. A
``REQUEST`` when a remark was meant lets a remark reach a tool, which is
the expensive direction and is counted first. A ``MENTION`` when a
request was meant stops a command that used to work, and is counted
separately because the two are not equally bad.

Two kinds of leak are distinguished, because they need different fixes:

``tell``-cue leaks
    The request cue fired inside a narrative frame. A fix scoped to
    ``tell`` would address these.
Fall-through leaks
    No framing rule fired at all, so the verdict stayed :data:`NEUTRAL`.
    These are utterances about the past whose verbs ("tells", "telling")
    or perception verbs ("saw", "remember") are not in
    :data:`~assistant.nlu.framing.NARRATIVE_VERBS`, so **no change to
    ``tell`` alone would reach them**. Counting them separately is the
    point of running this study before proposing a fix.
"""

from __future__ import annotations

import pytest

from assistant.nlu import framing
from assistant.nlu.normalize import normalize
from tests.nlu_framing_tell_corpus import (
    CATEGORIES,
    CATEGORY_NARRATIVE,
    CATEGORY_TRAP,
    CORPUS,
    REQUEST,
    TellCase,
    by_category,
)

#: Size of the corpus, pinned so cases cannot be dropped to flatter a
#: future change.
RECORDED_CORPUS_SIZE = 57

#: Accuracy measured when this corpus was first run, against the rules as
#: they stand at the end of Phase 5 Step 5. Pinned so a later change to
#: ``framing.py`` has to move it deliberately.
#:
#: Step 7's reported-"tell" guard closed all eight cue leaks, taking the
#: corpus from 47 of 57 to 55 of 57. Step 9's declarative-"tells" guard
#: closed one fall-through, taking it to 56. Step 11's recall guard
#: closed the last one, so the corpus is now **57 of 57**. No expectation
#: was edited to get there.
RECORDED_ACCURACY = 1.0

#: Mismatch counts measured at the same time, pinned for the same reason.
#: Nothing leaks and nothing is blocked: this corpus is fully correct.
RECORDED_LEAKING = 0
RECORDED_BLOCKING = 0
RECORDED_WRONG = 0

#: The leak breakdown, which is the point of running this study. Step 7
#: removed the cue leaks, Step 9 closed "my brother tells me jokes", and
#: Step 11 closed "I remember you telling me a joke". The gerund is now
#: handled for the recall shape, so no leak of either kind is left.
RECORDED_LEAKING_FROM_CUE = 0
RECORDED_LEAKING_FALLTHROUGH = 0

#: Nothing is left. Named as an empty tuple rather than deleted, so the
#: fact that this corpus was fully closed is recorded in the place a
#: future step will look.
REMAINING_FALLTHROUGH: tuple[str, ...] = ()

#: The gap this study set out to close, recorded as closed. Step 6 named
#: it and Step 7 fixed it, so the test below asserts it stays fixed.
FIXED_LIVE_GAP = "I heard you tell a joke"



def actual_verdict(case: TellCase) -> str:
    """Run the production rule over one case."""
    return framing.assess(normalize(case.utterance), case.intent)


def is_correct(case: TellCase) -> bool:
    return actual_verdict(case) == case.expected


def verdict_of(utterance: str, intent: str) -> str:
    """Judge a bare utterance, for cases named outside the corpus."""
    return framing.assess(normalize(utterance), intent)


def accuracy(cases=CORPUS) -> float:
    if not cases:
        return 1.0
    return sum(1 for case in cases if is_correct(case)) / len(cases)


class Mismatch:
    """One case where the rule and the expected label disagree."""

    def __init__(self, case: TellCase, actual: str) -> None:
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
    def from_request_cue(self) -> bool:
        """True when the ``tell`` cue is what let this case through.

        These are the leaks a fix scoped to ``tell`` could reach. The
        rest fell through to :data:`~assistant.nlu.framing.NEUTRAL`
        because no rule matched, and need a different change entirely.
        """
        return self.actual == framing.REQUEST

    def describe(self) -> str:
        return (
            f"{self.case.utterance!r} [{self.case.intent}] "
            f"expected {self.case.expected}, got {self.actual} "
            f"({self.kind})"
        )


def mismatches(cases=CORPUS) -> list[Mismatch]:
    return [Mismatch(c, actual_verdict(c)) for c in cases if not is_correct(c)]


def by_kind(kind: str, cases=CORPUS) -> list[Mismatch]:
    return [m for m in mismatches(cases) if m.kind == kind]


def leaks_from_cue(cases=CORPUS) -> list[Mismatch]:
    """Leaks caused by the ``tell`` request cue firing."""
    return [m for m in by_kind("false-let-through", cases) if m.from_request_cue]


def fallthrough_leaks(cases=CORPUS) -> list[Mismatch]:
    """Leaks where no framing rule fired, so the verdict stayed neutral."""
    return [m for m in by_kind("false-let-through", cases) if not m.from_request_cue]


def blocking_metrics(cases=CORPUS) -> dict[str, int]:
    """Count the two error directions and the verdicts behind them."""
    expected_blocks = sum(1 for c in cases if c.blocks)
    actually_blocks = sum(1 for c in cases if actual_verdict(c) == framing.MENTION)
    return {
        "total": len(cases),
        "correct": sum(1 for c in cases if is_correct(c)),
        "expected_to_block": expected_blocks,
        "actually_blocked": actually_blocks,
        "false_let_through": len(by_kind("false-let-through", cases)),
        "false_block": len(by_kind("false-block", cases)),
        "wrong_label": len(by_kind("wrong-label", cases)),
    }


def build_report() -> str:
    """Render the whole measurement as readable text."""
    metrics = blocking_metrics()
    bad = mismatches()

    lines = [
        "tell-framing evaluation (MEASUREMENT ONLY)",
        f"  total cases        : {metrics['total']}",
        f"  correct            : {metrics['correct']}",
        f"  incorrect          : {len(bad)}",
        f"  accuracy           : {accuracy():.3f}",
        "",
        "  expected verdicts:",
    ]
    for verdict in (framing.REQUEST, framing.MENTION, framing.NEUTRAL):
        count = sum(1 for c in CORPUS if c.expected == verdict)
        lines.append(f"    {verdict:<24} {count}")

    lines += [
        "",
        "  blocking metrics:",
        f"    remarks meant to be blocked : {metrics['expected_to_block']}",
        f"    remarks actually blocked    : {metrics['actually_blocked']}",
        f"    remarks that ran a tool     : {metrics['false_let_through']}",
        f"    commands wrongly blocked    : {metrics['false_block']}",
        f"    wrong label either way      : {metrics['wrong_label']}",
        "",
        "  by category:",
    ]
    for category in CATEGORIES:
        subset = by_category(category)
        if not subset:
            continue
        hits = sum(1 for c in subset if is_correct(c))
        lines.append(f"    {category:<22} {hits:>2}/{len(subset):<2} correct")

    for category in CATEGORIES:
        group = [m for m in bad if m.case.category == category]
        if not group:
            continue
        lines.append("")
        lines.append(f"  mismatches in {category} ({len(group)}):")
        for item in group:
            lines.append(f"    - {item.describe()}")

    # Every kind is listed even when it is empty. A measurement that
    # silently omits a direction reads as if that direction were fine.
    for kind in ("false-let-through", "false-block", "wrong-label"):
        group = by_kind(kind)
        lines.append("")
        lines.append(f"  {kind} ({len(group)}):")
        if not group:
            lines.append("    none")
            continue
        for item in group:
            lines.append(f"    - {item.describe()}")

    lines += [
        "",
        "  leak breakdown:",
        f"    'tell' cue fired inside a narrative frame : {len(leaks_from_cue())}",
        f"    no framing rule fired (fell through)       : {len(fallthrough_leaks())}",
    ]
    if not bad:
        lines.append("")
        lines.append("  mismatches         : none")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Corpus integrity
# ----------------------------------------------------------------------
class TestCorpus:
    def test_corpus_is_large_enough_to_be_useful(self):
        assert len(CORPUS) >= 30

    def test_corpus_size_is_pinned(self):
        """Cases must not be dropped to flatter a future change."""
        assert len(CORPUS) == RECORDED_CORPUS_SIZE

    def test_labels_match_the_production_constants(self):
        assert (framing.REQUEST, framing.MENTION, framing.NEUTRAL) == (
            REQUEST,
            "mention",
            "neutral",
        )

    def test_every_expected_verdict_is_legal(self):
        for case in CORPUS:
            assert case.expected in (framing.REQUEST, framing.MENTION, framing.NEUTRAL)

    def test_every_category_is_represented(self):
        assert {case.category for case in CORPUS} == set(CATEGORIES)

    def test_every_category_is_non_trivial(self):
        """A one-case category cannot measure anything."""
        for category in CATEGORIES:
            assert len(by_category(category)) >= 5, category

    def test_no_duplicate_utterance_and_intent_pairs(self):
        pairs = [(c.utterance, c.intent) for c in CORPUS]
        assert len(pairs) == len(set(pairs))

    def test_intents_are_real_tool_names(self):
        from assistant.tools import build_default_router

        known = {tool.name for tool in build_default_router().tools}
        assert {case.intent for case in CORPUS} <= known

    def test_every_case_carries_a_reason(self):
        for case in CORPUS:
            assert case.note, case.utterance

    def test_both_verdicts_are_represented(self):
        counts = {c.expected for c in CORPUS}
        assert framing.REQUEST in counts
        assert framing.MENTION in counts

    def test_negation_is_left_to_the_layer_that_handles_it(self):
        """Negated requests are already stopped upstream, so not measured here."""
        for case in CORPUS:
            assert "do not" not in case.utterance, case.utterance
            assert "never" not in case.utterance, case.utterance

    def test_the_word_tell_is_really_under_study(self):
        """The corpus must be about "tell", with only documented exceptions.

        Matched on the family rather than the bare string, because half
        the study is about the forms that are *not* the cue: "told" is a
        narrative verb, while "tells" and "telling" are in neither list.

        The only cases allowed to sit outside the family are the lexical
        traps, which exist precisely to test the neighbouring verbs
        ("said", "mention", "say", "saw", "heard") that a future fix
        could reach for.
        """
        family = ("tell", "told", "tells", "telling")
        outside = [c for c in CORPUS if not any(w in c.utterance for w in family)]
        assert all(c.category == CATEGORY_TRAP for c in outside), [
            c.utterance for c in outside
        ]
        assert len(outside) <= 5, [c.utterance for c in outside]

    def test_the_earlier_corpora_are_untouched(self):
        """A single-word study must not be paid for by editing the others."""
        from tests import nlu_framing_adversarial_corpus as adversarial
        from tests import nlu_framing_corpus as original

        assert len(original.CORPUS) == 32
        assert len(adversarial.CORPUS) == 46


# ----------------------------------------------------------------------
# The production facts this measurement rests on
# ----------------------------------------------------------------------
class TestProductionFactsUnderMeasurement:
    """Pinned so the numbers above are attributable to a known cause.

    This is the whole point of the study: the leaks come from ``tell``
    being an unconditional cue, and these assertions record that.
    """

    def test_tell_is_a_request_cue(self):
        assert "tell" in framing.REQUEST_CUES

    def test_told_is_a_narrative_verb(self):
        """The past form is handled; the bare form is not."""
        assert "told" in framing.NARRATIVE_VERBS

    def test_tell_is_not_also_a_narrative_verb(self):
        """It cannot be both, or "tell me a joke" would be blocked."""
        assert "tell" not in framing.NARRATIVE_VERBS

    def test_the_other_forms_of_tell_are_unlisted_everywhere(self):
        """Why "tells" and "telling" fall through to neutral."""
        for word in ("telling", "tells"):
            assert word not in framing.REQUEST_CUES
            assert word not in framing.NARRATIVE_VERBS

    def test_perception_verbs_are_not_narrative_verbs(self):
        """Why "I saw you tell a joke" is not caught either."""
        for word in ("saw", "see", "remember"):
            assert word not in framing.NARRATIVE_VERBS
            assert word not in framing.REQUEST_CUES


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

    def test_the_blocking_metrics_add_up(self):
        """The three kinds must partition every incorrect case."""
        metrics = blocking_metrics()
        assert (
            metrics["false_let_through"]
            + metrics["false_block"]
            + metrics["wrong_label"]
            == len(mismatches())
        )

    def test_mismatches_do_not_fail_the_suite(self):
        """This step measures. It does not enforce."""
        assert isinstance(mismatches(), list)

    def test_measured_accuracy_matches_the_recorded_figure(self):
        assert accuracy() == pytest.approx(RECORDED_ACCURACY, abs=1e-9), (
            f"recorded {RECORDED_ACCURACY} but measured {accuracy()}; "
            "update RECORDED_ACCURACY if a rule was changed deliberately"
        )

    def test_accuracy_did_not_regress(self):
        assert accuracy() >= RECORDED_ACCURACY, build_report()

    def test_the_recorded_mismatch_counts_still_hold(self):
        assert len(by_kind("false-let-through")) == RECORDED_LEAKING
        assert len(by_kind("false-block")) == RECORDED_BLOCKING
        assert len(by_kind("wrong-label")) == RECORDED_WRONG

    def test_the_leak_breakdown_is_recorded(self):
        """A "tell" fix reaches the cue leaks and only those."""
        assert len(leaks_from_cue()) == RECORDED_LEAKING_FROM_CUE
        assert len(fallthrough_leaks()) == RECORDED_LEAKING_FALLTHROUGH

    def test_the_two_leak_groups_partition_the_leaks(self):
        assert len(leaks_from_cue()) + len(fallthrough_leaks()) == RECORDED_LEAKING

    def test_the_cheap_direction_has_no_errors_either(self):
        """Every case meant to be blocked is already blocked correctly.

        Worth pinning on its own: a future fix that narrows the ``tell``
        cue will show up here first, as a newly blocked command.
        """
        blocked_wrongly = by_kind("false-block")
        assert not blocked_wrongly, [m.describe() for m in blocked_wrongly]

    def test_the_known_live_gap_is_now_closed(self):
        """Step 6 measured it, Step 7 closed it, and it stays closed.

        This is the sentence the whole corpus was built around, so it is
        asserted by name in both directions: the old expectation is gone
        from the mismatch list, and the new one is right.
        """
        utterances = [m.case.utterance for m in mismatches()]
        assert FIXED_LIVE_GAP not in utterances
        assert verdict_of(FIXED_LIVE_GAP, "jokes") == framing.MENTION

    def test_only_the_fallthroughs_remain(self):
        """The two survivors are named, and they are the documented two."""
        remaining = sorted(m.case.utterance for m in mismatches())
        assert remaining == sorted(REMAINING_FALLTHROUGH)

    def test_the_corpus_is_now_fully_correct(self):
        """Steps 7, 9 and 11 between them closed every leak here."""
        assert not mismatches()
        assert accuracy() == 1.0

    def test_the_gerund_leak_is_blocked_not_merely_unlisted(self):
        """"I remember you telling me a joke" was the last one.

        It is blocked by the recall guard, which is a different thing
        from being a request cue: the request side of the gerund still
        runs.
        """
        case = next(
            c for c in CORPUS if c.utterance == "I remember you telling me a joke"
        )
        assert verdict_of(case.utterance, case.intent) == framing.MENTION

    def test_a_genuine_telling_order_still_runs(self):
        """The guard is narrow: the request side of the gerund is spared."""
        assert verdict_of("keep telling me more jokes", "jokes") == framing.NEUTRAL
        assert verdict_of("keep telling me more", "jokes") == framing.NEUTRAL

    def test_the_fallthroughs_are_unrelated_to_tell(self):
        """Kept as a guard on the assertion itself, now that it is empty.

        Were a leak to reappear here, this is the check that would say
        whether it is a "tell" problem or something else.
        """
        for utterance in REMAINING_FALLTHROUGH:
            tokens = normalize(utterance).tokens
            assert "tell" not in tokens, utterance

    def test_the_remaining_leaks_are_in_the_documented_categories(self):
        """Vacuous while nothing leaks, and kept for the same reason."""
        assert not mismatches()

    def test_genuine_requests_are_never_blocked_today(self):
        """Every case that should run is running. The danger is the reverse."""
        assert not by_kind("false-block")

    def test_mismatches_are_grouped_by_category_in_the_report(self):
        report = build_report()
        for item in mismatches():
            assert f"mismatches in {item.case.category}" in report
            assert item.describe() in report

    def test_the_report_names_every_kind_it_counts(self):
        report = build_report()
        for kind in ("false-let-through", "false-block", "wrong-label"):
            assert kind in report
        assert "blocking metrics" in report
        assert "leak breakdown" in report

    def test_the_report_is_printed(self, capsys):
        report = build_report()
        with capsys.disabled():
            print(report)

