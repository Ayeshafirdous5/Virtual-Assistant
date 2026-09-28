"""Measurement of contextual command phrasing.

This step is **measurement only**. No production code is changed, and no
fix is proposed or implemented here.

The question
------------
Step 14 showed that no ordinary alias can carry the twelve remaining
gaps, because every candidate's wording also occurs in a sentence that
is not a command. This module asks the follow-up honestly:

    can **context or structure** tell them apart?

The answer, measured rather than assumed, is a qualified yes.

What the signal is
------------------
**Trigger position.** Across the 19 command/non-command pairs in
:mod:`tests.nlu_contextual_command_corpus`, the trigger is
sentence-initial in the command and mid-sentence in its conversational
twin in **17 of 19**. The two failures are ``"what is"`` and
``"did you know"``, where the twin is itself a question and opens
identically. So the signal is real, and it has a known shape.

Why it is still not a rule
--------------------------
A ``must be sentence-initial`` test is the obvious implementation, and
it would break ordinary speech. Every sentence in
:data:`~tests.nlu_contextual_command_corpus.MID_SENTENCE_COMMANDS` is a
genuine request with the trigger in the second slot::

    "ok amuse me"                "so define gravity"
    "today wind speed please"    "right gossip"

The signal is a **weight**, not a **condition**, and nothing in the
current architecture has anywhere to put a weight.

The finding this step actually produced
---------------------------------------
A **live, dangerous bug that predates it**::

    "I should quit smoking"   -> system / clear / neutral   -> the app quits

The ``system`` guard requires the trigger in the closing words, and
``quit`` sits in the second of four tokens, so it passes. This is
recorded as an expected mismatch and is reported end to end, because a
measurement step that found it and said nothing would be worth less
than one that had not looked.

Layer attribution
-----------------
:func:`pipeline_status` runs each case through the real application so a
parser match is not confused with a command running. "he was big in the
news today" matches ``news`` and is then **rejected by framing**: the
parser layer is wrong and the pipeline is safe. "I should quit smoking"
matches and **runs**. Those two sentences are why the distinction is
measured rather than assumed.
"""

from __future__ import annotations

import functools

import pytest

from assistant.nlu import framing
from assistant.nlu.normalize import normalize
from assistant.nlu.parser import parse
from assistant.nlu.scoring import score_intents
from tests.nlu_contextual_command_corpus import (
    ALREADY_SUPPORTED,
    CATEGORIES,
    CATEGORY_FACTS_JOKES,
    CATEGORY_INFORMATION,
    CATEGORY_MID_SENTENCE,
    CATEGORY_NEWS,
    CATEGORY_SYSTEM,
    CATEGORY_WEATHER,
    CLASSIFICATIONS,
    COMMAND,
    CORPUS,
    DANGEROUS_CASES,
    MATCH,
    MID_SENTENCE_COMMANDS,
    MISS_IS_DANGEROUS,
    NEEDS_MORE_EVIDENCE,
    NON_COMMAND,
    NO_MATCH,
    SAFE_CONTEXTUAL,
    UNSAFE,
    ContextCase,
    by_category,
)

#: Size of the corpus, pinned so cases cannot be dropped.
RECORDED_CORPUS_SIZE = 60

#: Measured at the end of Phase 5 Step 14, with no production change in
#: this step. Every case matches its expectation except the one live bug
#: this corpus found, which is deliberately left mismatched.
RECORDED_AS_EXPECTED = 59
RECORDED_MISMATCHES = 1

#: The single mismatch, named. A bare count would let it be "fixed" by
#: editing the label.
RECORDED_MISMATCH = "I should quit smoking"

#: How the cases divide. Pinned because these numbers are the step's
#: whole output.
RECORDED_SAFE_CONTEXTUAL = 22
RECORDED_UNSAFE = 9
RECORDED_ALREADY_SUPPORTED = 20
RECORDED_NEEDS_MORE_EVIDENCE = 9

#: The positional signal, measured on the command/twin pairs in the
#: corpus. Nineteen of twenty-one separate; the two that do not are the
#: question-shaped families, where the twin opens the same way.
RECORDED_PAIRS = 21
RECORDED_PAIRS_SEPARATED_BY_POSITION = 19

#: Of those pairs, how many are already-supported commands whose trigger
#: is a topic word rather than the whole request. Those are excluded from
#: the "sentence-initial" claim, and the count is pinned so the
#: exclusion cannot quietly grow.
RECORDED_SUPPORTED_IN_PAIRS = 4


@functools.lru_cache(maxsize=1)
def _lexicon():
    from assistant.app import build_runtime_lexicon
    from assistant.tools import build_default_router

    return build_runtime_lexicon(build_default_router())


@pytest.fixture(scope="module")
def lexicon():
    return _lexicon()


def trigger_index(utterance: str, trigger: str) -> int:
    """Where the trigger sits, or ``-1`` when it is absent."""
    if not trigger:
        return -1
    tokens = normalize(utterance).tokens
    wanted = trigger.split()
    for start in range(len(tokens) - len(wanted) + 1):
        if list(tokens[start : start + len(wanted)]) == wanted:
            return start
    return -1


def parser_outcome(utterance: str, lexicon) -> str:
    """What the parser returns, as one of the three corpus outcomes."""
    parsed = parse(utterance, lexicon)
    if parsed is None:
        return NO_MATCH
    return "ambiguous" if parsed.is_ambiguous else MATCH


def top_candidate(utterance: str, lexicon):
    """The highest-scoring candidate the scoring layer produced."""
    found = score_intents(normalize(utterance), lexicon)
    return found[0] if found else None


def framing_verdict(utterance: str, intent: str) -> str:
    return framing.assess(normalize(utterance), intent)


def pipeline_status(utterance: str, lexicon) -> str:
    """The real application path, so a match is not read as a run."""
    from assistant.app import resolve_detail
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
    return resolve_detail(ctx, build_default_router(), utterance, lexicon).status


class Measurement:
    """One case, measured every way this step cares about."""

    def __init__(self, case: ContextCase, lexicon) -> None:
        self.case = case
        self.outcome = parser_outcome(case.utterance, lexicon)
        self.top = top_candidate(case.utterance, lexicon)
        self.index = trigger_index(case.utterance, case.trigger)
        self.framing = framing_verdict(case.utterance, case.intent)
        self.reaches_framing = self.outcome == MATCH

    @property
    def is_command(self) -> bool:
        return self.case.outcome == COMMAND

    @property
    def as_expected(self) -> bool:
        return self.outcome == self.case.expected

    @property
    def is_dangerous(self) -> bool:
        return self.case.significance == MISS_IS_DANGEROUS

    def describe(self) -> str:
        return (
            f"{self.case.utterance!r} expected {self.case.expected}, "
            f"got {self.outcome} "
            f"(top={self.top.intent if self.top else '-'} "
            f"{self.top.score:.2f} {self.top.method if self.top else ''}, "
            f"framing={self.framing or 'not reached'})"
        )


def measurements(cases=CORPUS) -> list[Measurement]:
    lex = _lexicon()
    return [Measurement(c, lex) for c in cases]


def mismatches(cases=CORPUS) -> list[Measurement]:
    return [m for m in measurements(cases) if not m.as_expected]


def pairs() -> list[tuple[Measurement, Measurement]]:
    """Every command paired with its conversational twin.

    Built from the corpus by matching on the trigger and the intent, so
    a pair really is the same phrase used two ways. The mid-sentence
    commands are excluded on purpose: they are commands with no twin, and
    pairing them with an unrelated sentence would invent evidence.
    """
    lex = _lexicon()
    measured = [
        Measurement(c, lex)
        for c in CORPUS
        if c.trigger and c.category != CATEGORY_MID_SENTENCE
    ]
    out: list[tuple[Measurement, Measurement]] = []
    for command in measured:
        if not command.is_command:
            continue
        twin = next(
            (
                m
                for m in measured
                if not m.is_command
                and m.case.trigger == command.case.trigger
                and m.case.intent == command.case.intent
            ),
            None,
        )
        if twin is not None:
            out.append((command, twin))
    return out


def build_report() -> str:
    """Render the measurement as readable text."""
    all_cases = measurements()
    bad = mismatches()
    commands = [m for m in all_cases if m.is_command]
    talking = [m for m in all_cases if not m.is_command]

    lines = [
        "contextual command phrasing (MEASUREMENT ONLY)",
        f"  total cases        : {len(CORPUS)}",
        f"  as expected        : {len(CORPUS) - len(bad)}",
        f"  mismatches         : {len(bad)}",
        f"  commands           : {len(commands)}",
        f"  non-commands       : {len(talking)}",
        "",
        "  by category:",
    ]
    for category in CATEGORIES:
        subset = by_category(category)
        if not subset:
            continue
        hits = len(subset) - len(
            [m for m in mismatches() if m.case.category == category]
        )
        lines.append(f"    {category:<24} {hits:>2}/{len(subset):<2} as expected")

    lines += ["", "  classification:"]
    for label in CLASSIFICATIONS:
        count = sum(1 for c in CORPUS if c.classification == label)
        lines.append(f"    {label:<32} {count}")

    lines += [
        "",
        "  the positional signal:",
        f"    pairs measured                 : {len(pairs())}",
        f"    separated by trigger position  : "
        f"{sum(1 for c, t in pairs() if c.index != t.index)}",
        f"    NOT separated                  : "
        f"{sum(1 for c, t in pairs() if c.index == t.index)}",
        "",
        "    Position is a weight, not a condition. Every mid-sentence",
        "    command in the corpus is a real request that a",
        "    'must be sentence-initial' rule would break.",
        "",
        f"    mid-sentence commands that would break: {len(MID_SENTENCE_COMMANDS)}",
    ]

    lines += [
        "",
        "  risk split:",
        f"    a miss here is harmless       : "
        f"{sum(1 for c in CORPUS if not c.is_dangerous)}",
        f"    a miss here is dangerous      : {len(DANGEROUS_CASES)}",
    ]

    lines += ["", "  layer attribution:"]
    lines.append(
        f"    framing reached                : "
        f"{sum(1 for m in all_cases if m.reaches_framing)}"
    )
    lines.append(
        f"    framing never reached          : "
        f"{len(CORPUS) - sum(1 for m in all_cases if m.reaches_framing)}"
    )
    lines.append(
        "    A parser match is not a command running. Framing rejects"
    )
    lines.append(
        "    some of them, and the system intent is the one that does not."
    )

    lines += ["", "  mismatches:"]
    if not bad:
        lines.append("    none")
    for item in bad:
        lines.append(f"    - {item.describe()}")

    if RECORDED_MISMATCH:
        lines += [
            "",
            "  the one live bug this step found:",
            f"    {RECORDED_MISMATCH!r}",
            "    the parser returns system/clear, framing returns neutral,",
            "    and the application terminates. Pre-existing, and it is",
            "    left mismatched on purpose.",
        ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Corpus integrity
# ----------------------------------------------------------------------
class TestCorpus:
    def test_corpus_is_large_enough_to_be_useful(self):
        assert len(CORPUS) >= 35

    def test_corpus_size_is_pinned(self):
        assert len(CORPUS) == RECORDED_CORPUS_SIZE

    def test_no_duplicate_utterances(self):
        utterances = [c.utterance for c in CORPUS]
        assert len(utterances) == len(set(utterances))

    def test_every_expected_outcome_is_legal(self):
        for case in CORPUS:
            assert case.expected in (MATCH, NO_MATCH)

    def test_every_category_is_represented(self):
        assert {c.category for c in CORPUS} == set(CATEGORIES)

    def test_every_category_is_non_trivial(self):
        for category in CATEGORIES:
            assert len(by_category(category)) >= 4, category

    def test_every_classification_is_a_documented_one(self):
        for case in CORPUS:
            assert case.classification in CLASSIFICATIONS, case.classification

    def test_every_case_carries_a_reason(self):
        for case in CORPUS:
            assert case.note, case.utterance

    def test_intents_are_real_tool_names(self):
        from assistant.tools import build_default_router

        known = {tool.name for tool in build_default_router().tools}
        assert {c.intent for c in CORPUS} <= known

    def test_commands_and_conversation_are_both_represented(self):
        outcomes = {c.outcome for c in CORPUS}
        assert COMMAND in outcomes
        assert NON_COMMAND in outcomes

    def test_every_non_command_names_its_twin(self):
        """A conversational case with no command to contrast is useless."""
        for case in CORPUS:
            if case.outcome != NON_COMMAND or not case.trigger:
                continue
            twin = any(
                other.outcome == COMMAND
                and other.trigger == case.trigger
                and other.intent == case.intent
                for other in CORPUS
            )
            assert twin, case.utterance

    def test_both_risk_levels_are_represented(self):
        risky = [c for c in CORPUS if c.significance == MISS_IS_DANGEROUS]
        assert risky and len(risky) < len(CORPUS)


# ----------------------------------------------------------------------
# The measurement
# ----------------------------------------------------------------------
class TestMeasurement:
    def test_every_case_is_measured(self):
        assert len(measurements()) == len(CORPUS)

    def test_measurement_is_deterministic(self):
        assert [m.outcome for m in measurements()] == [
            m.outcome for m in measurements()
        ]

    def test_mismatches_do_not_fail_the_suite(self):
        """This step measures. It does not enforce."""
        assert isinstance(mismatches(), list)

    def test_the_baseline_is_as_recorded(self):
        assert len(CORPUS) - len(mismatches()) == RECORDED_AS_EXPECTED
        assert len(mismatches()) == RECORDED_MISMATCHES

    def test_the_single_mismatch_is_the_live_bug(self):
        """Named, so it can never be closed by editing the label."""
        assert [m.case.utterance for m in mismatches()] == [RECORDED_MISMATCH]
        case = next(c for c in CORPUS if c.utterance == RECORDED_MISMATCH)
        assert case.expected == NO_MATCH
        assert case.significance == MISS_IS_DANGEROUS
        assert case.classification == UNSAFE

    def test_the_live_bug_really_terminates_the_assistant(self):
        """End to end. This is what makes the step worth doing."""
        from assistant.app import resolve_detail
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
        resolution = resolve_detail(
            ctx, build_default_router(), RECORDED_MISMATCH, _lexicon()
        )
        assert resolution.tool_name == "system"
        assert resolution.status == "resolved"

    def test_the_classification_counts_are_pinned(self):
        counts = {label: 0 for label in CLASSIFICATIONS}
        for case in CORPUS:
            counts[case.classification] += 1
        assert counts[SAFE_CONTEXTUAL] == RECORDED_SAFE_CONTEXTUAL
        assert counts[UNSAFE] == RECORDED_UNSAFE
        assert counts[ALREADY_SUPPORTED] == RECORDED_ALREADY_SUPPORTED
        assert counts[NEEDS_MORE_EVIDENCE] == RECORDED_NEEDS_MORE_EVIDENCE
        assert sum(counts.values()) == len(CORPUS)

    def test_the_positional_signal_is_pinned(self):
        """The headline finding, with its own count."""
        matched = pairs()
        assert len(matched) == RECORDED_PAIRS
        separated = sum(1 for c, t in matched if c.index != t.index)
        assert separated == RECORDED_PAIRS_SEPARATED_BY_POSITION

    def test_the_signal_holds_for_the_gap_families(self):
        """Scope of the signal, stated honestly.

        It holds for the phrases Step 14 was trying to add, where the
        trigger *is* the whole request. It does **not** hold for
        already-supported commands, where the trigger is a topic word
        sitting at the end ("tell me a fact"), and averaging the two
        together would overstate the finding.
        """
        gap_categories = {
            CATEGORY_INFORMATION,
            CATEGORY_FACTS_JOKES,
            CATEGORY_WEATHER,
            CATEGORY_NEWS,
            CATEGORY_SYSTEM,
        }
        checked = 0
        for command, _twin in pairs():
            if command.case.category not in gap_categories:
                continue
            if command.case.classification == ALREADY_SUPPORTED:
                continue
            assert command.index == 0, command.case.utterance
            checked += 1
        assert checked == RECORDED_PAIRS - RECORDED_SUPPORTED_IN_PAIRS

    def test_the_signal_does_not_hold_for_topic_triggers(self):
        """Why the scope above is narrower than the corpus."""
        case = next(c for c in CORPUS if c.utterance == "tell me a fact")
        assert trigger_index(case.utterance, case.trigger) != 0
        assert case.classification == ALREADY_SUPPORTED

    def test_the_two_question_families_do_not_separate(self):
        """The known limits, named rather than averaged away."""
        stuck = {c.case.trigger for c, t in pairs() if c.index == t.index}
        assert stuck == {"what is", "did you know"}

    def test_position_alone_would_break_real_commands(self):
        """Why this is a weight and not a condition."""
        for case in MID_SENTENCE_COMMANDS:
            assert trigger_index(case.utterance, case.trigger) != 0, (
                case.utterance
            )
            assert case.outcome == COMMAND, case.utterance

    def test_every_dangerous_case_is_the_conversational_side_of_a_pair(self):
        """:attr:`significance` describes the cost of being wrong here.

        For a dangerous case, being wrong means a remark reached a tool,
        so it is always the **non-command** half of a command/twin pair.
        The command half carries the opposite, harmless cost: not
        matching it is a missed request.
        """
        for case in CORPUS:
            if not case.is_dangerous:
                continue
            assert case.outcome == NON_COMMAND, case.utterance
            assert any(
                other.outcome == COMMAND
                and other.trigger == case.trigger
                and other.intent == case.intent
                for other in CORPUS
            ), case.utterance

    def test_the_system_family_is_never_recommended(self):
        """No case in the dangerous family is a safe candidate."""
        for case in by_category("system-dangerous"):
            assert case.classification in (UNSAFE, ALREADY_SUPPORTED), (
                case.utterance
            )

    def test_no_production_code_is_required_by_this_step(self):
        """The step is complete as a measurement; nothing depends on it."""
        from assistant.nlu.lexicon import COMMON_ALIASES

        assert COMMON_ALIASES["jokes"] == ("make me laugh",)
        assert COMMON_ALIASES["system"] == (
            "stop the assistant",
            "stop listening",
            "shut down",
            "that is all",
        )

    def test_the_report_states_every_section(self, ):
        report = build_report()
        for heading in (
            "the positional signal",
            "risk split",
            "layer attribution",
            "mismatches:",
            "the one live bug this step found",
        ):
            assert heading in report, heading

    def test_the_report_names_every_mismatch(self):
        report = build_report()
        for item in mismatches():
            assert item.case.utterance in report, item.case.utterance

    def test_the_report_is_printed(self, capsys):
        report = build_report()
        with capsys.disabled():
            print(report)

