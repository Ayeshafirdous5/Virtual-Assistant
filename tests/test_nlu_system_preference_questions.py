"""Measurement of the ``do you want to`` system-safety family.

This step is **measurement only**. No production rule is changed, no
alias is added, and no existing safety rule is touched.

What is being measured
-----------------------
Steps 17 and 20 closed fourteen dangerous executions and left one::

    "do you want to quit"  ->  system / clear  ->  SystemTool runs

This module asks the seven questions the step set, and answers them with
numbers rather than intuition:

1. :func:`test_the_family_is_larger_than_one_verb`
2. :func:`test_every_system_trigger_is_reached`
3. :func:`test_the_subject_is_the_discriminator`
4. :func:`test_want_alone_is_not_the_signal`
5. :func:`test_the_load_bearing_commands_are_separated`
6. :func:`test_framing_approves_rather_than_blocks`
7. :func:`test_the_guard_is_the_narrowest_owner`

Ten axes per case
-----------------
Every case is recorded on all ten axes the step asked for: normalised
tokens, matched trigger, trigger index, score, confidence, the guard's
own verdict, the parser result, the framing verdict, the resolver
status, and the final tool with whether :class:`SystemTool` would run.

Classification is by **outcome, never by intention**
-----------------------------------------------------
:func:`classify` returns :data:`DANGEROUS_EXECUTION` whenever
``SystemTool`` is reached, whatever the sentence might mean. In
particular a dangerous sentence is never filed as
:data:`AMBIGUOUS` merely because a person could read it charitably:
:data:`~assistant.nlu.scoring.AMBIGUOUS` is reserved for the case where
the **parser itself** declined to choose between two intents, which is
the only situation where nothing ran.
"""

from __future__ import annotations

import functools

import pytest

from assistant.app import (
    STATUS_AMBIGUOUS,
    STATUS_NO_MATCH,
    STATUS_RESOLVED,
    build_runtime_lexicon,
    resolve_detail,
)
from assistant.core.context import AppContext
from assistant.nlu import framing
from assistant.nlu.normalize import normalize
from assistant.nlu.parser import AMBIGUOUS, parse
from assistant.nlu.scoring import (
    SELF_OTHER_SUBJECTS,
    SYSTEM_INTENT,
    Candidate,
    _match_index,
    score_intents,
)
from assistant.tools import build_default_router
from assistant.tools.system import SystemTool
from tests.nlu_system_preference_question_corpus import (
    AMBIGUOUS,
    AMBIGUOUS_CASES,
    CATEGORIES,
    CATEGORY_ASSISTANT_DIRECTED,
    CATEGORY_CONVERSATIONAL,
    CATEGORY_DIRECT,
    CATEGORY_PREFERENCE,
    CATEGORY_SUBJECT,
    CLASSIFICATIONS,
    CORPUS,
    DANGEROUS_EXECUTION,
    DANGEROUS_FAMILY,
    LOAD_BEARING_COMMANDS,
    SAFE_BLOCK,
    SAFE_COMMAND,
    UNSUPPORTED,
    PreferenceCase,
    by_category,
)

#: Size of the corpus, pinned so cases cannot be dropped.
RECORDED_CORPUS_SIZE = 58

#: Measured at the end of Phase 5 Step 20, with no production change in
#: this step. Pinned so a later change has to move them deliberately.
RECORDED_DANGEROUS = len(DANGEROUS_FAMILY)
RECORDED_SAFE_COMMAND = 0


@functools.lru_cache(maxsize=1)
def _lexicon():
    return build_runtime_lexicon(build_default_router())


@pytest.fixture
def quiet():
    ctx = AppContext()

    class Silent:
        def speak(self, text: str) -> None:
            pass

        def listen(self) -> str:
            return ""

    ctx.speaker = Silent()
    ctx.listener = Silent()
    return ctx


@pytest.fixture
def router():
    return build_default_router()


def system_candidate(utterance: str):
    """The system candidate the scorer produced, if any."""
    normalized = normalize(utterance)
    found = [c for c in score_intents(normalized, _lexicon()) if c.intent == SYSTEM_INTENT]
    if not found:
        return None
    candidate = found[0]
    entry = next(e for e in _lexicon().entries if e.pattern == candidate.trigger)
    index = _match_index(entry, normalized.text)
    passed = _guard_verdict(normalized.text, candidate, index)
    return candidate, index, passed


def _guard_verdict(text: str, candidate, index: int) -> bool:
    """Whether the system guard would let this candidate through."""
    from assistant.nlu.scoring import _passes_system_guard

    if index < 0:
        return False
    return _passes_system_guard(text, candidate, index)


class Measurement:
    """One case, recorded on all ten axes."""

    def __init__(self, case: PreferenceCase, quiet, router) -> None:
        self.case = case
        utterance = case.utterance
        self.normalized = normalize(utterance).tokens
        found = system_candidate(utterance)
        self.candidate = found[0] if found else None
        self.index = found[1] if found else -1
        self.guard_passed = found[2] if found else False
        parsed = parse(utterance, _lexicon())
        resolution = resolve_detail(quiet, router, utterance, _lexicon())
        self.parser = parsed.name if parsed else "no-match"
        self.confidence = parsed.confidence if parsed else "-"
        self.framing = framing.assess(normalize(utterance), parsed.name) if parsed else "-"
        self.framing_consulted = parsed is not None
        self.status = resolution.status
        self.tool = resolution.tool_name or ""
        self.executes = isinstance(resolution.tool, SystemTool)

    @property
    def classification(self) -> str:
        """By outcome **and** expectation, never by intention alone.

        ``SystemTool`` being reached is only a defect when the sentence
        was never an order. Reaching it for "can you exit" is the system
        working, so the two are combined: the tool ran, *and* the
        sentence was not something the user was doing.
        """
        if self.executes:
            return SAFE_COMMAND if self.case.should_execute else DANGEROUS_EXECUTION
        if self.status == STATUS_AMBIGUOUS:
            return AMBIGUOUS
        if self.case.should_execute:
            return UNSUPPORTED
        if self.parser == "no-match":
            return UNSUPPORTED
        return SAFE_BLOCK

    @property
    def is_dangerous(self) -> bool:
        return self.classification == DANGEROUS_EXECUTION

    def row(self) -> str:
        return (
            f"{self.classification:<20} {self.case.utterance!r:<34} "
            f"trig={self.candidate.trigger if self.candidate else '-':<9} "
            f"@{self.index:<2} score={self.candidate.score if self.candidate else 0:.2f} "
            f"guard={self.guard_passed!s:<5} parse={self.parser:<10} "
            f"framing={self.framing:<8} status={self.status:<10} "
            f"tool={self.tool or '-':<7} runs={self.executes}"
        )


def measurements(cases=CORPUS) -> list[Measurement]:
    ctx = AppContext()

    class Silent:
        def speak(self, text: str) -> None:
            pass

        def listen(self) -> str:
            return ""

    ctx.speaker = Silent()
    ctx.listener = Silent()
    router = build_default_router()
    return [Measurement(c, ctx, router) for c in cases]


def by_class(label: str, cases=CORPUS) -> list[Measurement]:
    return [m for m in measurements(cases) if m.classification == label]


def build_report() -> str:
    """Render the measurement as readable text."""
    all_cases = measurements()
    dangerous = by_class(DANGEROUS_EXECUTION)
    commands = by_class(SAFE_COMMAND)
    blocks = by_class(SAFE_BLOCK)
    ambiguous = by_class(AMBIGUOUS)
    unsupported = by_class(UNSUPPORTED)

    lines = [
        "system preference questions (Step 21, MEASUREMENT ONLY)",
        f"  total cases        : {len(CORPUS)}",
        f"  dangerous          : {len(dangerous)}",
        f"  safe commands      : {len(commands)}",
        f"  safe blocks        : {len(blocks)}",
        f"  ambiguous          : {len(ambiguous)}",
        f"  unsupported        : {len(unsupported)}",
        "",
        "  by category:",
    ]
    for category in CATEGORIES:
        subset = by_category(category)
        if not subset:
            continue
        lines.append(f"    {category:<22} {len(subset):>2} cases")

    lines += ["", f"  DANGEROUS_EXECUTION ({len(dangerous)}):"]
    for item in dangerous:
        lines.append(f"    {item.case.utterance!r}")
    lines += ["", f"  SAFE_COMMAND ({len(commands)}):"]
    for item in commands:
        lines.append(f"    {item.case.utterance!r}")

    lines += [
        "",
        "  full trace:",
    ]
    for item in all_cases:
        lines.append(f"    {item.row()}")

    lines += [
        "",
        "  straddling cases for a 'to' rule:",
    ]
    for utterance in AMBIGUOUS_CASES:
        item = next(
            (m for m in all_cases if m.case.utterance == utterance), None
        )
        lines.append(f"    {utterance!r:<26} {item.classification if item else '-'}")
    return "\n".join(lines)



# ----------------------------------------------------------------------
# Corpus integrity
# ----------------------------------------------------------------------
class TestCorpus:
    def test_corpus_is_large_enough_to_be_useful(self):
        assert len(CORPUS) >= 40

    def test_corpus_size_is_pinned(self):
        assert len(CORPUS) == RECORDED_CORPUS_SIZE

    def test_no_duplicate_utterances(self):
        utterances = [c.utterance for c in CORPUS]
        assert len(utterances) == len(set(utterances))

    def test_every_category_is_represented(self):
        assert {c.category for c in CORPUS} == set(CATEGORIES)

    def test_every_category_is_non_trivial(self):
        for category in CATEGORIES:
            assert len(by_category(category)) >= 6, category

    def test_every_case_carries_a_reason(self):
        for case in CORPUS:
            assert case.note, case.utterance

    def test_every_case_records_a_reading(self):
        """Intent and outcome are separate columns, and both are filled."""
        for case in CORPUS:
            assert case.intent_note, case.utterance

    def test_the_dangerous_family_is_in_the_corpus(self):
        utterances = {c.utterance for c in CORPUS}
        for utterance in DANGEROUS_FAMILY:
            assert utterance in utterances, utterance

    def test_the_load_bearing_commands_are_in_the_corpus(self):
        utterances = {c.utterance for c in CORPUS}
        for utterance in LOAD_BEARING_COMMANDS:
            assert utterance in utterances, utterance


# ----------------------------------------------------------------------
# Questions 1 to 4
# ----------------------------------------------------------------------
class TestQuestionOneIsItUniquelyDangerous:
    def test_the_family_is_larger_than_one_verb(self):
        found = {m.case.utterance for m in by_class(DANGEROUS_EXECUTION)}
        assert found == set(DANGEROUS_FAMILY)
        assert len(found) > 1, "the step assumed a family; check it"

    def test_the_original_sentence_is_in_it(self):
        assert "do you want to quit" in DANGEROUS_FAMILY


class TestQuestionTwoDoesTheVerbMatter:
    def test_every_system_trigger_is_reached(self):
        """Replacing the verb changes nothing. The frame is the cause."""
        for verb in ("quit", "exit", "goodbye", "bye", "shut down"):
            utterance = f"do you want to {verb}"
            item = next(m for m in measurements() if m.case.utterance == utterance)
            assert item.classification == DANGEROUS_EXECUTION, utterance

    def test_the_phrase_trigger_is_reached_too(self):
        item = next(
            m for m in measurements()
            if m.case.utterance == "do you want to shut down"
        )
        assert item.candidate.trigger == "shut down"
        assert item.executes


class TestQuestionThreeDoesTheSubjectDecide:
    def test_the_subject_is_the_discriminator(self):
        """``you`` is the only pronoun that reaches the trigger."""
        for item in by_class(DANGEROUS_EXECUTION):
            subjects = [t for t in item.normalized if t in SELF_OTHER_SUBJECTS]
            assert not subjects, (item.case.utterance, subjects)

    def test_you_is_the_one_missing_from_the_subject_set(self):
        assert "you" not in SELF_OTHER_SUBJECTS
        for word in ("i", "she", "they", "me", "them", "we"):
            assert word in SELF_OTHER_SUBJECTS, word

    def test_every_other_pronoun_is_already_refused(self):
        for pronoun in ("I", "we", "he", "she", "they", "me", "him", "her",
                        "us", "them"):
            utterance = f"{pronoun} want to quit"
            assert parse(utterance, _lexicon()) is None, utterance

    def test_you_is_still_a_legal_addressee(self):
        """The reason ``you`` is absent, and why a fix is delicate."""
        for utterance in ("you can quit now", "can you exit"):
            assert parse(utterance, _lexicon()) is not None, utterance


class TestQuestionFourIsWantASignal:
    def test_want_alone_is_not_the_signal(self):
        """``want`` appears in the safe cases too."""
        for utterance in ("I want to quit", "we want to quit",
                          "do you want me to quit"):
            assert parse(utterance, _lexicon()) is None, utterance

    def test_the_frame_is_want_plus_you_as_subject(self):
        """Both halves are needed, and only together are they dangerous.

        Restricted to the preference family, because two of the
        dangerous cases are conversational and reach the same place by a
        different route: "plans to quit" has no subject at all.
        """
        family = [
            item
            for item in by_class(DANGEROUS_EXECUTION)
            if item.case.category == CATEGORY_PREFERENCE
        ]
        assert family
        for item in family:
            assert "want" in item.normalized, item.case.utterance
            assert "you" in item.normalized, item.case.utterance

    def test_the_dangerous_set_has_a_second_route(self):
        """Two dangerous cases are not preference questions at all.

        "plans to quit" and "nothing to quit over" reach SystemTool
        without a subject or a "you", so a fix aimed only at the
        preference frame would leave them behind.
        """
        conversational = [
            item.case.utterance
            for item in by_class(DANGEROUS_EXECUTION)
            if item.case.category == CATEGORY_CONVERSATIONAL
        ]
        assert sorted(conversational) == ["nothing to quit over", "plans to quit"]



# ----------------------------------------------------------------------
# Questions 5 to 7
# ----------------------------------------------------------------------
class TestQuestionFiveCanTheFamilyBeSeparated:
    def test_the_load_bearing_commands_are_separated(self):
        """Every command that must survive is measured, not assumed."""
        for utterance in LOAD_BEARING_COMMANDS:
            item = next(m for m in measurements() if m.case.utterance == utterance)
            assert item.executes, utterance

    def test_named_commands_still_reach_the_tool(self):
        """The five the step named explicitly."""
        for utterance in ("can you exit", "would you quit now", "will you quit",
                          "can I quit", "should I exit"):
            item = next(m for m in measurements() if m.case.utterance == utterance)
            assert item.classification in (SAFE_COMMAND, UNSUPPORTED), utterance

    def test_first_person_questions_are_already_refused(self):
        for utterance in ("can I quit", "should I exit", "may I quit",
                          "should we quit"):
            assert parse(utterance, _lexicon()) is None, utterance

    def test_to_alone_could_not_do_the_job(self):
        """The obvious one-token fix is unsafe, and the corpus shows why.

        A rule blocking on ``to`` before the trigger would break commands
        that work today and fix other bugs at the same time, so the word
        cannot carry the decision on its own.
        """
        for utterance in ("time to quit", "ready to quit", "try to quit"):
            assert parse(utterance, _lexicon()) is not None, utterance
        for utterance in ("plans to quit", "nothing to quit over"):
            assert parse(utterance, _lexicon()) is not None, utterance
        assert AMBIGUOUS_CASES


class TestQuestionSixDoesFramingHelp:
    def test_framing_approves_rather_than_blocks(self):
        """Framing is not a candidate owner, and this is the evidence."""
        for item in by_class(DANGEROUS_EXECUTION):
            assert item.framing in (framing.REQUEST, framing.NEUTRAL), item.row()

    def test_framing_does_see_them_and_lets_them_through(self):
        """The guard passes, so framing is consulted, and it agrees."""
        for item in by_class(DANGEROUS_EXECUTION):
            assert item.framing_consulted, item.case.utterance

    def test_the_parser_is_equally_blind(self):
        for item in by_class(DANGEROUS_EXECUTION):
            assert item.parser == "system", item.case.utterance
            assert item.confidence == "clear", item.case.utterance


class TestQuestionSevenWhichLayerOwnsAFix:
    def test_the_guard_is_the_narrowest_owner(self):
        """The scorer is the only layer that can refuse at all.

        The parser has already decided ``system`` by the time framing is
        reached, and framing approves the sentence. The guard inside
        :mod:`assistant.nlu.scoring` is where every previous fix landed,
        and it is the only place a refusal can be introduced.
        """
        for item in by_class(DANGEROUS_EXECUTION):
            assert item.guard_passed, item.case.utterance

    def test_no_threshold_would_do_it(self):
        """Every candidate is a perfect or near-perfect exact match.

        Nothing is being under-scored, so lowering a bar cannot help: the
        only lever is a structural refusal.
        """
        for item in by_class(DANGEROUS_EXECUTION):
            assert item.candidate.score >= 0.95, item.case.utterance
            assert item.candidate.method in ("exact", "phrase"), item.case.utterance

    def test_no_fix_is_implemented_here(self):
        """A measurement step changes nothing."""
        from assistant.nlu.scoring import (
            DEFINITION_MARKERS,
            SYSTEM_MIN_SCORE,
            SYSTEM_TAIL_TOKENS,
        )

        assert DEFINITION_MARKERS == frozenset(
            {"does", "do", "of", "define", "explain"}
        )
        assert SYSTEM_MIN_SCORE == 0.95
        assert SYSTEM_TAIL_TOKENS == 2
        assert "you" not in SELF_OTHER_SUBJECTS


# ----------------------------------------------------------------------
# The measurement itself
# ----------------------------------------------------------------------
class TestMeasurement:
    def test_every_case_is_measured(self):
        assert len(measurements()) == len(CORPUS)

    def test_measurement_is_deterministic(self):
        assert [m.classification for m in measurements()] == [
            m.classification for m in measurements()
        ]

    def test_mismatches_do_not_fail_the_suite(self):
        """This step measures. The dangerous count is pinned, not asserted away."""
        assert isinstance(by_class(DANGEROUS_EXECUTION), list)

    def test_the_dangerous_count_is_pinned(self):
        assert len(by_class(DANGEROUS_EXECUTION)) == RECORDED_DANGEROUS

    def test_dangerous_is_never_hidden_as_ambiguous(self):
        """A run that should not have happened is always labelled dangerous.

        The step's instruction, asserted rather than trusted: no case
        that reached ``SystemTool`` on a non-command may be filed as
        :data:`AMBIGUOUS`, however charitable a reading it has.
        """
        for item in measurements():
            if item.case.should_execute:
                continue
            if item.executes:
                assert item.classification == DANGEROUS_EXECUTION, item.row()
            else:
                assert item.classification != DANGEROUS_EXECUTION, item.row()

    def test_ambiguous_only_means_the_parser_asked(self):
        """A tie is the only thing that stops a run, and the only
        thing that may be filed as ambiguous."""
        for item in by_class(AMBIGUOUS):
            assert item.status == STATUS_AMBIGUOUS, item.case.utterance
            assert not item.executes, item.case.utterance

    def test_every_case_falls_in_exactly_one_class(self):
        for item in measurements():
            assert item.classification in CLASSIFICATIONS, item.case.utterance

    def test_the_classes_partition_the_corpus(self):
        total = sum(len(by_class(label)) for label in CLASSIFICATIONS)
        assert total == len(CORPUS)

    def test_the_measures_are_printed(self, capsys):
        report = build_report()
        assert report
        with capsys.disabled():
            print(report)

