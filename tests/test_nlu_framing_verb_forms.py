"""Measurement of the ``tell`` verb forms in the framing layer.

This step is **measurement only**. No production rule changes here, and
nothing in this module asserts that the current behaviour is right.

What is being measured
----------------------
Step 7's reported-"tell" guard closed every leak caused by the bare
``tell`` request cue. Two survived, and they are a different problem:

    "my brother tells me jokes"        "tells"
    "I remember you telling me a joke" "telling"

Neither contains ``tell``. Both forms are in *neither* production list,
so framing has no opinion and returns :data:`~assistant.nlu.framing.NEUTRAL`,
which always proceeds. This corpus (:mod:`tests.nlu_framing_verb_forms_corpus`)
measures the four forms together, so a future step can decide what to do
about "tells" and "telling" on evidence.

The distinction this module is built around
------------------------------------------
A framing verdict is **not** the same thing as a tool running, and a
report that conflates the two is worse than no report. So every case is
put through two measurements:

:func:`framing_verdict`
    ``framing.assess()`` in isolation, against the labelled intent. This
    is the layer under study.
:func:`pipeline_reaches_framing`
    whether the parser produced an intent at all, which is the condition
    for framing to be consulted in the real application.

A case can therefore be a framing leak that never fires, and several are.
Those are counted and named separately as :data:`NEVER_REACHED`. They
are **not** treated as fixed, and this module refuses to let a future
change quietly claim them by adding a count that would confuse the two.

Reporting consequence
---------------------
A case the parser never saw is a real gap in a *different* layer. It
means the sentence names no trigger word, so the parser returns ``None``
and nothing downstream is consulted. The count is reported as
``parser no-match`` rather than folded into the framing accuracy, so the
two problems stay visible as two problems.
"""

from __future__ import annotations

import functools

import pytest

from assistant.nlu import framing
from assistant.nlu.normalize import normalize
from assistant.nlu.parser import parse
from tests.nlu_framing_verb_forms_corpus import (
    CATEGORIES,
    CORPUS,
    REQUEST,
    VERB_FORMS,
    VerbFormCase,
    by_category,
)

#: Size of the corpus, pinned so cases cannot be dropped to flatter a
#: future change.
RECORDED_CORPUS_SIZE = 49

#: Framing accuracy measured when this corpus was first run, against the
#: rules as they stand at the end of Phase 5 Step 7. Pinned so a later
#: change to ``framing.py`` has to move it deliberately.
#:
#: Step 9's declarative-"tells" guard took this from 35 of 49 to 42 of 49.
#: No label was edited to get there. The one framing leak left is
#: "I remember you telling me a joke", which is the gerund and was
#: explicitly out of scope.
RECORDED_ACCURACY = 0.8571428571428571

#: Framing mismatch counts, pinned for the same reason. Five are
#: let-throughs, two are wrong labels, and nothing is blocked: no
#: command is stopped that should run.
RECORDED_LEAKING = 5
RECORDED_BLOCKING = 0
RECORDED_WRONG = 2

#: How many of the seven framing was actually asked about. The other
#: :data:`RECORDED_NEVER_REACHED` name no trigger word, so the parser
#: returns ``None`` and framing is never consulted. Step 9 cut the
#: framing group from six to one, and could not touch the parser group,
#: which is exactly the split this module exists to keep visible.
RECORDED_LEAKING_REACHED = 1
RECORDED_NEVER_REACHED = 6

#: The two leaks that are *not* leaks. Both are labelled REQUEST and both
#: come back neutral, so they are wrong labels rather than let-throughs.
RECORDED_WRONG_UNREACHED = 2

#: The one framing leak Step 9 was not allowed to fix, named so it stays
#: visible. "telling" is the gerund and needs its own decision.
REMAINING_FRAMING_LEAK = "I remember you telling me a joke"

#: Cases whose verb form is the point of the case, named so a future
#: step cannot describe the corpus as being about "tell" alone.
FORMS_UNDER_STUDY = ("tell", "tells", "telling", "told")


def framing_verdict(case: VerbFormCase) -> str:
    """Judge one case with framing alone, in isolation from the parser."""
    return framing.assess(normalize(case.utterance), case.intent)


def is_correct(case: VerbFormCase) -> bool:
    return framing_verdict(case) == case.expected


def accuracy(cases=CORPUS) -> float:
    if not cases:
        return 1.0
    return sum(1 for case in cases if is_correct(case)) / len(cases)


#: Cause labels for a framing mismatch. The whole point of this step is
#: that these are not interchangeable, so they are named as data.
CAUSE_FRAMING = "framing-behaviour"
CAUSE_PARSER_NO_MATCH = "parser-no-match"
CAUSE_AMBIGUOUS = "ambiguous-upstream"
CAUSE_OTHER_LAYER = "another-layer"

#: Cases the parser never produces an intent for, so framing is never
#: consulted. A mismatch here is a parser gap, not a framing gap.
NEVER_REACHED = CAUSE_PARSER_NO_MATCH


@functools.lru_cache(maxsize=1)
def _lexicon():
    """The runtime lexicon, built once. It imports no tool into the NLU."""
    from assistant.app import build_runtime_lexicon
    from assistant.tools import build_default_router

    return build_runtime_lexicon(build_default_router())


def pipeline_reaches_framing(utterance: str) -> tuple[bool, str]:
    """Does the parser produce an intent, and therefore reach framing?

    Returns ``(reached, reason)``. ``reason`` is one of the ``CAUSE_*``
    labels, or ``""`` when the parser returned an intent confidently and
    framing was consulted.
    """
    parsed = parse(utterance, _lexicon())
    if parsed is None:
        return False, CAUSE_PARSER_NO_MATCH
    if parsed.confidence == "ambiguous":
        return False, CAUSE_AMBIGUOUS
    return True, ""


class Mismatch:
    """One case where the rule and the expected label disagree.

    Every mismatch carries a **cause**, and the cause is the part that
    matters. A framing verdict that the parser never asked for is not a
    framing defect, and reporting it as one would send the next step
    looking in the wrong file.
    """

    def __init__(self, case: VerbFormCase, actual: str) -> None:
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
        """True when framing was actually asked and answered wrongly.

        This is the only subset a change to ``framing.py`` can address.
        """
        return self.reached and self.kind in ("false-let-through", "false-block")

    @property
    def is_parser_issue(self) -> bool:
        """True when the parser stopped the sentence before framing."""
        return not self.reached

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

    lines = [
        "verb-form evaluation for tell / tells / telling / told",
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
        "    A parser gap is NOT a framing fix. Those sentences name no",
        "    trigger word, so the parser returns None and framing is never",
        "    asked. They are counted here so they cannot be quietly",
        "    claimed later.",
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

    lines += [
        "",
        "  addressable by a framing change:",
    ]
    if not issues:
        lines.append("    none")
    for item in issues:
        lines.append(f"    - {item.case.utterance!r} ({item.case.category})")

    lines += [
        "",
        "  NOT addressable by framing (parser never reached):",
    ]
    if not gaps:
        lines.append("    none")
    for item in gaps:
        lines.append(f"    - {item.case.utterance!r} ({item.cause})")

    lines += [
        "",
        "  current production treatment:",
    ]
    for form, treatment in VERB_FORMS.items():
        lines.append(f"    {form:<10} {treatment}")

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

    def test_all_four_verb_forms_appear(self):
        """The point of the corpus is the forms, so all four must be here."""
        for form in FORMS_UNDER_STUDY:
            hits = [c for c in CORPUS if form in c.utterance.split()]
            assert hits, form

    def test_both_verdicts_are_represented(self):
        counts = {c.expected for c in CORPUS}
        assert framing.REQUEST in counts
        assert framing.MENTION in counts

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

        assert len(original.CORPUS) == 32
        assert len(adversarial.CORPUS) == 46
        assert len(tell.CORPUS) == 57


# ----------------------------------------------------------------------
# The production facts this measurement rests on
# ----------------------------------------------------------------------
class TestProductionFactsUnderMeasurement:
    """Pinned so the numbers above are attributable to a known cause.

    This is the whole point of the study. The framing mismatches come
    from "tells" and "telling" being in *neither* list, and these
    assertions record that.
    """

    def test_tell_is_a_request_cue(self):
        assert "tell" in framing.REQUEST_CUES

    def test_told_is_a_narrative_verb(self):
        assert "told" in framing.NARRATIVE_VERBS

    def test_tells_and_telling_are_in_neither_list(self):
        """Why the two surviving leaks exist at all."""
        for word in ("tells", "telling"):
            assert word not in framing.REQUEST_CUES, word
            assert word not in framing.NARRATIVE_VERBS, word

    def test_the_verb_form_table_matches_production(self):
        """The documented table in the corpus must be true, not prose."""
        for form, treatment in VERB_FORMS.items():
            in_cues = form in framing.REQUEST_CUES
            in_narrative = form in framing.NARRATIVE_VERBS
            if treatment == "request-cue":
                assert in_cues and not in_narrative, form
            elif treatment == "narrative-verb":
                assert in_narrative and not in_cues, form
            else:
                assert not in_cues and not in_narrative, form

    def test_the_nearby_verbs_are_where_they_are_documented(self):
        """The past forms that are listed, and the ones deliberately not.

        "said" and "say" are **not** narrative verbs. "say" is absent
        because "what did I say" is the history alias, and "said" was
        left out for the same reason. Those two still work as remarks
        because a copula or another rule catches the sentences they
        appear in, which the trap category shows.
        """
        for word in ("heard", "mentioned", "explained", "discussed", "told"):
            assert word in framing.NARRATIVE_VERBS, word
        for word in ("say", "said", "mention", "explain", "discuss", "hear"):
            assert word not in framing.NARRATIVE_VERBS, word

    def test_the_step_seven_guard_only_reads_tell(self):
        """The guard from Step 7 must not already be reading these forms."""
        for tokens in (("tells", "me", "jokes"), ("telling", "me", "a", "joke")):
            assert not framing._reports_speech(tokens, 0), tokens



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
        """Six are framing's, eight are the parser's. Keep them apart."""
        assert len(framing_issues()) == RECORDED_LEAKING_REACHED
        assert len(parser_gaps()) == RECORDED_NEVER_REACHED

    def test_the_two_groups_partition_the_mismatches(self):
        assert len(framing_issues()) + len(parser_gaps()) == len(mismatches())

    def test_the_wrong_labels_are_all_unreached(self):
        """Both are labelled REQUEST and neither is a let-through."""
        wrong = by_kind("wrong-label")
        assert len(wrong) == RECORDED_WRONG_UNREACHED
        assert all(item.is_parser_issue for item in wrong)

    def test_the_two_surviving_tell_corpus_leaks_are_here(self):
        """Both Step 7 fall-throughs started this corpus, and only one
        of them is still a framing leak.

        "my brother tells me jokes" was closed by Step 9's declarative
        guard. "I remember you telling me a joke" is the gerund and was
        left alone on purpose.
        """
        utterances = {c.utterance for c in CORPUS}
        assert "my brother tells me jokes" in utterances
        assert REMAINING_FRAMING_LEAK in utterances

    def test_the_one_remaining_framing_leak_is_the_gerund(self):
        """Step 9 was scoped to "tells" and must not have touched this."""
        assert [m.case.utterance for m in framing_issues()] == [
            REMAINING_FRAMING_LEAK
        ]
        assert framing_verdict(
            next(c for c in CORPUS if c.utterance == REMAINING_FRAMING_LEAK)
        ) == framing.NEUTRAL

    def test_a_parser_gap_is_never_called_fixed(self):
        """The distinction this step exists to protect.

        A case the parser never routes to framing cannot be improved by
        editing ``framing.py``. If such a case were ever counted as a
        framing fix, the accuracy figure would improve for the wrong
        reason, so the two groups are asserted to be disjoint sets and
        both are pinned.
        """
        issues = {m.case.utterance for m in framing_issues()}
        gaps = {m.case.utterance for m in parser_gaps()}
        assert issues and gaps
        assert not (issues & gaps)

    def test_the_addressable_group_is_all_tells_or_telling(self):
        """The six framing leaks all carry an unlisted form."""
        for item in framing_issues():
            tokens = normalize(item.case.utterance).tokens
            assert "tells" in tokens or "telling" in tokens, item.case.utterance

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

    def test_the_report_lists_every_kind_it_counts(self):
        report = build_report()
        for kind in ("false-let-through", "false-block", "wrong-label"):
            assert kind in report

    def test_the_report_documents_the_verb_form_table(self):
        report = build_report()
        for form in FORMS_UNDER_STUDY:
            assert form in report, form

    def test_the_report_is_printed(self, capsys):
        report = build_report()
        with capsys.disabled():
            print(report)

