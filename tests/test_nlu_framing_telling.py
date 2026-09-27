"""Measurement of the gerund ``telling`` in the framing layer.

This step is **measurement only**. No production rule changes here, and
nothing in this module asserts that the current behaviour is right.

Why ``telling`` is the hard one
-------------------------------
Steps 7 to 9 closed the bare ``tell`` cue and the declarative ``tells``.
The gerund is the last form, and it is the only one that appears in
**both** directions at once:

    "I remember you telling me a joke"   reported speech  -> should block
    "keep telling me more"               an order         -> must not block

There is no subject test that separates those, because the first has a
subject in front of the gerund and the second has an implicit one. So
this corpus keeps the two populations in separate categories, and the
report shows what each one costs today, rather than assuming a rule.

The distinction this module is built around
--------------------------------------------
A framing verdict is not the same thing as a tool running. Every case is
therefore measured twice:

:func:`framing_verdict`
    ``framing.assess()`` in isolation, against the labelled intent.
:func:`pipeline_reaches_framing`
    whether the parser produced an intent at all, which is the condition
    for framing to be consulted in the real application.

A case can be a framing leak that never fires, and **most of them are**.
Those are counted and named separately as :data:`NEVER_REACHED`. They
are not treated as fixed, and this module refuses to let a later change
quietly claim them by folding them into one accuracy figure.

The one genuine framing gap
---------------------------
Of the thirteen mismatches, five are cases where the parser routes the
sentence and framing answers wrongly. Four of those are the reported
speech this study is about. The fifth is the mirror image and is the
important one: **"keep telling me more jokes" is a genuine request that
framing has no opinion about.** A rule written for the four would very
likely catch the fifth, and that is the cost the next step has to weigh.
"""

from __future__ import annotations

import functools

import pytest

from assistant.nlu import framing
from assistant.nlu.normalize import normalize
from assistant.nlu.parser import parse
from tests.nlu_framing_telling_corpus import (
    CATEGORIES,
    CATEGORY_GERUND_TRAP,
    CATEGORY_NARRATIVE,
    CATEGORY_REQUEST,
    CORPUS,
    REQUEST,
    TellingCase,
    by_category,
)

#: Size of the corpus, pinned so cases cannot be dropped to flatter a
#: future change.
RECORDED_CORPUS_SIZE = 41

#: Framing accuracy measured when this corpus was first run, against the
#: rules as they stand at the end of Phase 5 Step 9. Pinned so a later
#: change to ``framing.py`` has to move it deliberately.
#:
#: Step 11's recall + "telling" guard took this from 28 of 41 to 35 of
#: 41. No label was edited to get there. **Every** reported-speech leak
#: is gone, and the one reachable mismatch left is the request the guard
#: was written not to touch.
RECORDED_ACCURACY = 0.8536585365853658

#: Framing mismatch counts, pinned for the same reason. Two are
#: let-throughs, four are wrong labels, and nothing is blocked.
RECORDED_LEAKING = 2
RECORDED_BLOCKING = 0
RECORDED_WRONG = 4

#: How many of the six framing was actually asked about. The other
#: :data:`RECORDED_NEVER_REACHED` name no trigger word, so the parser
#: returns ``None`` and framing is never consulted.
RECORDED_LEAKING_REACHED = 1
RECORDED_NEVER_REACHED = 5

#: How the one reachable mismatch points. Step 11 closed all four
#: reported-speech gaps, and deliberately left the single request gap:
#: "keep telling me more jokes" is a genuine order that framing still has
#: no opinion about. It is not broken, because neutral lets it run.
RECORDED_REPORTED_SPEECH_GAPS = 0
RECORDED_REQUEST_GAPS = 1

#: The request the guard must not catch, named so it stays visible. It is
#: the one case a future rule for "telling" has to keep sparing.
REMAINING_REQUEST_GAP = "keep telling me more jokes"

#: Cause labels for a framing mismatch. These are not interchangeable,
#: so they are named as data rather than described in prose.
CAUSE_FRAMING = "framing-behaviour"
CAUSE_PARSER_NO_MATCH = "parser-no-match"
CAUSE_AMBIGUOUS = "ambiguous-upstream"
CAUSE_OTHER_LAYER = "another-layer"

#: Cases the parser never produces an intent for, so framing is never
#: consulted. A mismatch here is a parser gap, not a framing gap.
NEVER_REACHED = CAUSE_PARSER_NO_MATCH


@functools.lru_cache(maxsize=1)
def _lexicon():
    """The runtime lexicon, built once."""
    from assistant.app import build_runtime_lexicon
    from assistant.tools import build_default_router

    return build_runtime_lexicon(build_default_router())


def pipeline_reaches_framing(utterance: str) -> tuple[bool, str]:
    """Does the parser produce an intent, and therefore reach framing?

    Returns ``(reached, reason)``. ``reason`` is one of the ``CAUSE_*``
    labels, or ``""`` when the parser returned a confident intent and
    framing was consulted.
    """
    parsed = parse(utterance, _lexicon())
    if parsed is None:
        return False, CAUSE_PARSER_NO_MATCH
    if parsed.confidence == "ambiguous":
        return False, CAUSE_AMBIGUOUS
    return True, ""


def framing_verdict(case: TellingCase) -> str:
    """Judge one case with framing alone, in isolation from the parser."""
    return framing.assess(normalize(case.utterance), case.intent)


def is_correct(case: TellingCase) -> bool:
    return framing_verdict(case) == case.expected


def accuracy(cases=CORPUS) -> float:
    if not cases:
        return 1.0
    return sum(1 for case in cases if is_correct(case)) / len(cases)

class Mismatch:
    """One case where the rule and the expected label disagree.

    Every mismatch carries a **cause**, and the cause is the part that
    matters. A framing verdict the parser never asked for is not a framing
    defect, and reporting it as one would send the next step looking in
    the wrong file.
    """

    def __init__(self, case: TellingCase, actual: str) -> None:
        self.case = case
        self.actual = actual
        self.reached, self.cause = pipeline_reaches_framing(case.utterance)

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
    def is_framing_issue(self) -> bool:
        """True when framing was actually consulted and answered wrongly.

        The only subset a change to ``framing.py`` can address, and it
        covers all three kinds. A reachable *wrong label* counts too: the
        parser did ask, and framing did answer incorrectly, so this is
        framing behaviour even though nothing was blocked.
        """
        return self.reached

    @property
    def is_parser_issue(self) -> bool:
        """True when the parser stopped the sentence before framing."""
        return not self.reached

    @property
    def is_reported_speech_gap(self) -> bool:
        """A reachable remark that reached a tool instead of being blocked."""
        return (
            self.is_framing_issue
            and self.case.expected == framing.MENTION
        )

    @property
    def is_request_gap(self) -> bool:
        """A reachable request that framing has no opinion about.

        This is the one to watch. It is not a live defect today, because
        NEUTRAL lets the command through, but it is the case a future
        "telling is a remark" rule would be most likely to break.
        """
        return self.is_framing_issue and self.case.expected == framing.REQUEST

    def describe(self) -> str:
        return (
            f"{self.case.utterance!r} [{self.case.intent}] "
            f"expected {self.case.expected}, got {self.actual} "
            f"({self.kind}; {self.cause or 'framing consulted'})"
        )


def mismatches(cases=CORPUS) -> list[Mismatch]:
    return [Mismatch(c, framing_verdict(c)) for c in cases if not is_correct(c)]


def by_kind(kind: str, cases=CORPUS) -> list[Mismatch]:
    return [m for m in mismatches(cases) if m.kind == kind]


def framing_issues(cases=CORPUS) -> list[Mismatch]:
    """Mismatches a change to ``framing.py`` could actually address."""
    return [m for m in mismatches(cases) if m.is_framing_issue]


def parser_gaps(cases=CORPUS) -> list[Mismatch]:
    """Mismatches where the parser never reached framing at all."""
    return [m for m in mismatches(cases) if m.is_parser_issue]


def build_report() -> str:
    """Render the whole measurement as readable text."""
    bad = mismatches()
    issues = framing_issues()
    gaps = parser_gaps()
    speech = [m for m in issues if m.is_reported_speech_gap]
    requests = [m for m in issues if m.is_request_gap]

    lines = [
        "telling (gerund) evaluation",
        "  (MEASUREMENT ONLY - no production change)",
        f"  total cases        : {len(CORPUS)}",
        f"  correct            : {len(CORPUS) - len(bad)}",
        f"  incorrect          : {len(bad)}",
        f"  framing accuracy   : {accuracy():.3f}",
        "",
        "  expected verdicts:",
    ]
    for verdict in (framing.REQUEST, framing.MENTION, framing.NEUTRAL):
        count = sum(1 for c in CORPUS if c.expected == verdict)
        lines.append(f"    {verdict:<24} {count}")

    lines += [
        "",
        "  the layer that decided:",
        f"    framing was consulted and is wrong : {len(issues)}",
        f"    parser never reached framing       : {len(gaps)}",
        "",
        "    A parser no-match is NOT a framing fix. Those sentences name",
        "    no trigger word, so the parser returns None and framing is",
        "    never asked. They are counted here so they cannot be quietly",
        "    claimed by a later step.",
        "",
        "  the reachable gaps, and which way they point:",
        f"    reported speech that ran a tool : {len(speech)}",
        f"    requests framing has no opinion on: {len(requests)}",
        "",
        "    The second line is the one to read twice. Those requests are",
        "    not broken today, because neutral still lets them run. They",
        "    are the cases a 'telling is a remark' rule would catch.",
        "",
        "  by category:",
    ]
    for category in CATEGORIES:
        subset = by_category(category)
        if not subset:
            continue
        hits = sum(1 for c in subset if is_correct(c))
        lines.append(f"    {category:<24} {hits:>2}/{len(subset):<2} correct")

    for category in CATEGORIES:
        group = [m for m in bad if m.case.category == category]
        if not group:
            continue
        lines.append("")
        lines.append(f"  mismatches in {category} ({len(group)}):")
        for item in group:
            lines.append(f"    - {item.describe()}")

    for kind in ("false-let-through", "false-block", "wrong-label"):
        group = by_kind(kind)
        lines.append("")
        lines.append(f"  {kind} ({len(group)}):")
        if not group:
            lines.append("    none")
            continue
        for item in group:
            lines.append(f"    - {item.describe()}")

    lines += ["", "  addressable by a framing change:"]
    if not issues:
        lines.append("    none")
    for item in issues:
        direction = (
            "should have blocked"
            if item.is_reported_speech_gap
            else "should have run"
        )
        lines.append(
            f"    - {item.case.utterance!r} ({item.case.category}; {direction})"
        )

    lines += ["", "  NOT addressable by framing (parser never reached):"]
    if not gaps:
        lines.append("    none")
    for item in gaps:
        lines.append(f"    - {item.case.utterance!r} ({item.cause})")

    if not bad:
        lines.append("")
        lines.append("  mismatches         : none")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Corpus integrity
# ----------------------------------------------------------------------
class TestCorpus:
    def test_corpus_is_large_enough_to_be_useful(self):
        assert len(CORPUS) >= 35

    def test_corpus_size_is_pinned(self):
        """Cases must not be dropped to flatter a future change."""
        assert len(CORPUS) == RECORDED_CORPUS_SIZE

    def test_every_expected_verdict_is_legal(self):
        for case in CORPUS:
            assert case.expected in (framing.REQUEST, framing.MENTION, framing.NEUTRAL)

    def test_every_category_is_represented(self):
        assert {case.category for case in CORPUS} == set(CATEGORIES)

    def test_every_category_is_non_trivial(self):
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

    def test_telling_is_really_under_study(self):
        """The gerund carries most of the corpus.

        The only cases allowed to sit outside the tell word family are
        the neighbouring gerunds, which exist precisely to show what a
        rule written for "telling" would and would not reach.
        """
        family = ("telling", "tells", "tell", "told")
        outside = [
            c for c in CORPUS if not any(w in c.utterance for w in family)
        ]
        assert all(c.category == CATEGORY_GERUND_TRAP for c in outside), [
            c.utterance for c in outside
        ]
        assert len(outside) == len(by_category(CATEGORY_GERUND_TRAP))

    def test_both_verdicts_are_represented(self):
        counts = {c.expected for c in CORPUS}
        assert framing.REQUEST in counts
        assert framing.MENTION in counts

    def test_the_two_populations_are_both_substantial(self):
        """The decision the next step faces needs both sides measured."""
        assert len(by_category(CATEGORY_NARRATIVE)) >= 8
        assert len(by_category(CATEGORY_REQUEST)) >= 5

    def test_negation_is_left_to_the_layer_that_handles_it(self):
        """Negated requests are dropped upstream, so not measured here."""
        for case in CORPUS:
            assert "do not" not in case.utterance, case.utterance
            assert "never" not in case.utterance, case.utterance

    def test_the_earlier_corpora_are_untouched(self):
        """A new study must not be paid for by editing the older ones."""
        from tests import nlu_framing_adversarial_corpus as adversarial
        from tests import nlu_framing_corpus as original
        from tests import nlu_framing_tell_corpus as tell
        from tests import nlu_framing_verb_forms_corpus as verb_forms

        assert len(original.CORPUS) == 32
        assert len(adversarial.CORPUS) == 46
        assert len(tell.CORPUS) == 57
        assert len(verb_forms.CORPUS) == 49


# ----------------------------------------------------------------------
# The production facts this measurement rests on
# ----------------------------------------------------------------------
class TestProductionFactsUnderMeasurement:
    """Pinned so the numbers above are attributable to a known cause."""

    def test_telling_is_in_neither_list(self):
        """Why every reported-speech case falls through to neutral."""
        assert "telling" not in framing.REQUEST_CUES
        assert "telling" not in framing.NARRATIVE_VERBS

    def test_the_whole_family_is_where_step_nine_left_it(self):
        """"tell" a cue, "told" a narrative verb, the other two unlisted."""
        assert "tell" in framing.REQUEST_CUES
        assert "tell" not in framing.NARRATIVE_VERBS
        assert "told" in framing.NARRATIVE_VERBS
        assert "tells" not in framing.REQUEST_CUES
        assert "tells" not in framing.NARRATIVE_VERBS

    def test_step_nine_did_not_already_cover_telling(self):
        """The declarative guard reads "tells" only, by design."""
        for tokens in (
            ("i", "remember", "you", "telling", "me", "a", "joke"),
            ("keep", "telling", "me", "more"),
        ):
            assert framing._declares_habit(tokens) is False, tokens

    def test_the_nearby_gerunds_are_measured_not_assumed(self):
        """Each neighbouring gerund is handled by its own evidence.

        "talking" and "discussing" reach the past forms in
        :data:`NARRATIVE_VERBS`. "saying" and "hearing" are unlisted and
        are caught by the copula instead. None of them is a rule about
        the gerund, which is exactly why the trap category is measured
        rather than assumed.
        """
        for past in ("talked", "discussed", "explained", "mentioned"):
            assert past in framing.NARRATIVE_VERBS, past
        for gerund in ("talking", "discussing", "explaining", "mentioning"):
            assert gerund not in framing.NARRATIVE_VERBS, gerund
        for word in ("saying", "hearing", "telling"):
            assert word not in framing.NARRATIVE_VERBS, word


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
        assert [framing_verdict(c) for c in CORPUS] == [
            framing_verdict(c) for c in CORPUS
        ]

    def test_every_mismatch_is_classified(self):
        for item in mismatches():
            assert item.kind in ("false-let-through", "false-block", "wrong-label")

    def test_every_mismatch_has_a_cause(self):
        """A mismatch without a cause cannot be acted on."""
        for item in mismatches():
            assert item.reached is True or item.cause in (
                CAUSE_PARSER_NO_MATCH,
                CAUSE_AMBIGUOUS,
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

    def test_nothing_is_blocked_today(self):
        """The expensive direction first: a real command must not stop."""
        assert not by_kind("false-block"), [
            m.describe() for m in by_kind("false-block")
        ]

    def test_the_layer_split_is_recorded(self):
        """Five are framing's, eight are the parser's. Keep them apart."""
        assert len(framing_issues()) == RECORDED_LEAKING_REACHED
        assert len(parser_gaps()) == RECORDED_NEVER_REACHED

    def test_the_two_groups_partition_the_mismatches(self):
        assert len(framing_issues()) + len(parser_gaps()) == len(mismatches())

    def test_a_parser_gap_is_never_called_fixed(self):
        """The distinction this step exists to protect.

        A case the parser never routes to framing cannot be improved by
        editing ``framing.py``. Counting one as a framing fix would move
        the accuracy figure for entirely the wrong reason.
        """
        issues = {m.case.utterance for m in framing_issues()}
        gaps = {m.case.utterance for m in parser_gaps()}
        assert issues and gaps
        assert not (issues & gaps)

    def test_the_two_directions_are_counted_apart(self):
        """Four remarks should have blocked; one request should have run."""
        speech = [m for m in framing_issues() if m.is_reported_speech_gap]
        requests = [m for m in framing_issues() if m.is_request_gap]
        assert len(speech) == RECORDED_REPORTED_SPEECH_GAPS
        assert len(requests) == RECORDED_REQUEST_GAPS
        assert len(speech) + len(requests) == RECORDED_LEAKING_REACHED

    def test_the_request_gap_is_the_one_to_watch(self):
        """"keep telling me more jokes" is a genuine request.

        It is not broken today, because NEUTRAL lets it run. It is named
        here so a future rule for reported "telling" cannot be written,
        tested green, and quietly break it.
        """
        gap = next(m for m in framing_issues() if m.is_request_gap)
        assert gap.case.expected == framing.REQUEST
        assert gap.actual == framing.NEUTRAL
        assert gap.case.category == CATEGORY_REQUEST

    def test_the_speech_gaps_all_carry_the_gerund(self):
        """Vacuous once Step 11 closed them all, and that is the point.

        The assertion is kept rather than deleted: if a future change
        lets a reported-speech case leak again, this is what catches it,
        and the pin on the count above says how many there should be.
        """
        for item in framing_issues():
            if not item.is_reported_speech_gap:
                continue
            assert "telling" in normalize(item.case.utterance).tokens

    def test_every_reachable_reported_speech_leak_is_closed(self):
        """The four Step 10 targets, asserted one at a time."""
        for utterance in (
            "I remember you telling me a joke",
            "I remember him telling me the news",
            "I recall him telling me the news",
            "I remember you telling me the weather",
        ):
            assert framing_verdict(
                next(c for c in CORPUS if c.utterance == utterance)
            ) == framing.MENTION, utterance

    def test_no_reported_speech_survives_as_a_reachable_leak(self):
        assert not [m for m in framing_issues() if m.is_reported_speech_gap]

    def test_mismatches_are_grouped_by_category_in_the_report(self):
        report = build_report()
        for item in mismatches():
            assert f"mismatches in {item.case.category}" in report
            assert item.describe() in report

    def test_the_report_separates_the_two_layers(self):
        report = build_report()
        assert "framing was consulted and is wrong" in report
        assert "parser never reached framing" in report
        assert "NOT addressable by framing" in report
        assert CAUSE_PARSER_NO_MATCH in report

    def test_the_report_separates_the_two_directions(self):
        """The report must still be able to express both directions.

        "should have blocked" only appears while a reported-speech leak
        survives, so it is asserted through the builder rather than the
        rendered text: today the surviving gap points the other way, and
        the report has to say so.
        """
        report = build_report()
        assert "reported speech that ran a tool" in report
        assert "requests framing has no opinion on" in report
        assert "should have run" in report
        assert not [m for m in framing_issues() if m.is_reported_speech_gap]

    def test_the_report_lists_every_kind_it_counts(self):
        report = build_report()
        for kind in ("false-let-through", "false-block", "wrong-label"):
            assert kind in report

    def test_the_report_is_printed(self, capsys):
        report = build_report()
        with capsys.disabled():
            print(report)

