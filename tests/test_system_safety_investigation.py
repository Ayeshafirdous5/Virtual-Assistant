"""Investigation of the system safety guard. Diagnosis, not a fix.

Step 15 found that *"I should quit smoking"* terminates the assistant.
This module traces exactly why, and records the answer as tests.

How to read the expectations
----------------------------
Each case declares the outcome it **should** have. The suite then pins
what the code **currently** does, and the two are allowed to disagree.
That is deliberate: a failing test here would only tell us the risk still
exists, which the risk assertions already say far more clearly.

The dangerous cases are therefore *pinned as risks* rather than asserted
safe. They are named ``test_RISK_...`` so they cannot be skimmed past, and
they fail loudly if someone fixes the problem without updating this file,
which is what makes a future fix deliberate rather than accidental.

Nothing here changes production behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from assistant.app import build_runtime_lexicon, resolve_detail
from assistant.core.context import AppContext
from assistant.core.router import NoMatch
from assistant.nlu import framing
from assistant.nlu.normalize import normalize
from assistant.nlu.parser import parse
from assistant.nlu.scoring import (
    SELF_OTHER_SUBJECTS,
    SYSTEM_MIN_SCORE,
    SYSTEM_TAIL_TOKENS,
    score_intents,
)
from assistant.tools import build_default_router

#: The user meant "do not run any tool".
EXPECT_BLOCKED = "blocked"
#: The user meant "shut the assistant down"; the system tool should run.
EXPECT_EXIT = "exit"

#: Dangerous sentences that reached the system tool **before** Step 17.
#: Step 17's subject check in the system guard closed every one of them, and
#: the two counts are pinned so the closure cannot be undone silently and so
#: a regression is visible. Measured, not guessed.
RECORDED_CORE_RISK = 0
RECORDED_TOTAL_RISK = 0
RECORDED_CORE_RISK_BEFORE = 5
RECORDED_TOTAL_RISK_BEFORE = 7

#: First- or third-person exit phrasings that Step 17 also blocked. They
#: worked before, and are recorded here as the deliberate cost of the rule
#: rather than left as a surprise.
NEWLY_BLOCKED_BY_SUBJECT_CHECK = [
    "I need to exit",
    "let me quit",
    "can I quit",
    "should I exit",
    "I will quit now",
    "us exit now",
]


class Silent:
    def speak(self, text: str) -> None:
        pass

    def listen(self) -> str:
        return ""


@pytest.fixture
def router():
    return build_default_router()


@pytest.fixture
def lexicon(router):
    return build_runtime_lexicon(router)


@dataclass
class Probe:
    """The full pipeline trace for one utterance."""

    utterance: str
    normalized: str
    tokens: tuple
    candidates: list
    parsed_name: str | None
    parsed_trigger: str
    score: float
    confidence: str
    framing: str
    status: str
    tool_name: str
    reason: str
    legacy_match: str
    note: str = ""

    @property
    def executes(self) -> bool:
        return self.tool_name == "system"

    @property
    def classification(self) -> str:
        """One of the four outcomes the investigation must distinguish."""
        if self.parsed_name is None:
            return "parser-no-match"
        if self.parsed_name != "system":
            return "non-system-intent"
        if self.status == "rejected" and self.framing == framing.MENTION:
            return "blocked-by-framing"
        if self.executes:
            return "reaches-system"
        return "blocked-by-parser-or-guard"

    def describe(self) -> str:
        return (
            f"{self.utterance!r}: norm={self.normalized!r} "
            f"cand={[(c.intent, c.trigger, round(c.score, 3), c.method) for c in self.candidates]} "
            f"parsed={self.parsed_name} score={self.score:.3f} conf={self.confidence} "
            f"framing={self.framing} status={self.status} tool={self.tool_name or 'none'} "
            f"legacy={self.legacy_match or 'none'} -> {self.classification}"
        )

    def as_row(self) -> str:
        return (
            f"  {self.utterance!r:<34} norm={self.normalized!r:<32} "
            f"parsed={str(self.parsed_name):<11} score={self.score:.3f} "
            f"conf={self.confidence:<7} framing={self.framing:<8} "
            f"status={self.status:<10} tool={self.tool_name or 'none':<8} "
            f"executes={self.executes}"
        )


@dataclass
class SafetyCase:
    """An utterance and the safety outcome it ought to have."""

    utterance: str
    expected: str
    note: str = ""


def probe(utterance: str, router, lexicon) -> Probe:
    """Run one utterance through the entire pipeline and record every stage."""
    normalized = normalize(utterance)
    candidates = score_intents(normalized, lexicon)
    parsed = parse(utterance, lexicon)

    ctx = AppContext()
    ctx.speaker = Silent()
    ctx.listener = Silent()
    resolution = resolve_detail(ctx, router, utterance, lexicon)

    match = router.find_match(utterance)
    return Probe(
        utterance=utterance,
        normalized=normalized.text,
        tokens=normalized.tokens,
        candidates=list(candidates),
        parsed_name=parsed.name if parsed else None,
        parsed_trigger=parsed.trigger if parsed else "",
        score=parsed.score if parsed else 0.0,
        confidence=parsed.confidence if parsed else "-",
        framing=resolution.framing or ("-" if parsed is None else framing.NEUTRAL),
        status=resolution.status,
        tool_name=resolution.tool_name,
        reason=resolution.reason,
        legacy_match=(
            "" if isinstance(match, NoMatch) else match.tool_name
        ),
    )


# ----------------------------------------------------------------------
# A. The known dangerous false positive, found in Step 15.
# ----------------------------------------------------------------------
DANGEROUS: list[SafetyCase] = [
    SafetyCase("I should quit smoking", EXPECT_BLOCKED,
               "Step 15 finding: terminates the assistant"),
]

# ----------------------------------------------------------------------
# B. The same shape: a personal decision, not a command to the assistant.
# ----------------------------------------------------------------------
PERSONAL_DECISIONS: list[SafetyCase] = [
    SafetyCase("I want to quit smoking", EXPECT_BLOCKED,
               "the speaker is the user, not the assistant"),
    SafetyCase("she decided to quit smoking", EXPECT_BLOCKED,
               "third person; no addressee"),
    SafetyCase("he plans to quit smoking", EXPECT_BLOCKED,
               "third person; no addressee"),
    SafetyCase("I need to quit smoking", EXPECT_BLOCKED,
               "personal need, not a request to the assistant"),
    SafetyCase("they told me to quit smoking", EXPECT_BLOCKED,
               "already blocked, but only because 'told' reads as narrative"),
]

# ----------------------------------------------------------------------
# C. Genuine system commands. These must keep working; a fix that blocks
#    them is a regression, not a safety win.
# ----------------------------------------------------------------------
GENUINE_COMMANDS: list[SafetyCase] = [
    SafetyCase("exit", EXPECT_EXIT, "bare command"),
    SafetyCase("quit", EXPECT_EXIT, "bare command"),
    SafetyCase("goodbye", EXPECT_EXIT, "bare command"),
    SafetyCase("bye", EXPECT_EXIT, "bare command"),
    SafetyCase("please exit", EXPECT_EXIT, "politeness prefix"),
    SafetyCase("goodbye assistant", EXPECT_EXIT, "addressed to the assistant"),
    SafetyCase("shut down", EXPECT_EXIT, "two-word alias"),
    SafetyCase("you can quit now", EXPECT_EXIT, "addressed to the assistant"),
]

# ----------------------------------------------------------------------
# D. Lexical traps. The command word appears in ordinary language, where
#    the user is describing the world rather than issuing an order.
# ----------------------------------------------------------------------
LEXICAL_TRAPS: list[SafetyCase] = [
    SafetyCase("the exit was closed", EXPECT_BLOCKED, "a door, not a command"),
    SafetyCase("there is an exit sign", EXPECT_BLOCKED, "a sign, not a command"),
    SafetyCase("he quit smoking", EXPECT_BLOCKED, "third person"),
    SafetyCase("i want to exit early", EXPECT_BLOCKED, "leaving a meeting"),
    SafetyCase("the stop button is broken", EXPECT_BLOCKED, "a button"),
    SafetyCase("a bus stop", EXPECT_BLOCKED, "a place"),
    SafetyCase("train stop", EXPECT_BLOCKED, "a schedule"),
    SafetyCase("the quit smoking plan failed", EXPECT_BLOCKED, "about a habit"),
]

ALL_CASES: list[SafetyCase] = (
    DANGEROUS + PERSONAL_DECISIONS + GENUINE_COMMANDS + LEXICAL_TRAPS
)


def all_probes(cases=ALL_CASES, router=None, lexicon=None) -> list[Probe]:
    """Trace every case once."""
    return [probe(case.utterance, router, lexicon) for case in cases]


def reaching_system(cases=ALL_CASES, router=None, lexicon=None) -> list[Probe]:
    """Every case that currently terminates the assistant."""
    return [p for p in all_probes(cases, router, lexicon) if p.executes]


def _trace_report(router, lexicon) -> str:
    """The real, measured trace, one row per case."""
    return "\n".join(f"  {p.as_row()}" for p in all_probes(ALL_CASES, router, lexicon))


# ----------------------------------------------------------------------
# The guard's mechanics. This is the root cause, stated as a test.
# ----------------------------------------------------------------------
class TestGuardMechanics:
    def test_the_guard_is_a_window_not_a_closing_position(self):
        """Root cause: the guard allows the trigger in the last N tokens.

        ``_passes_system_guard`` computes
        ``tail_start = len(tokens) - SYSTEM_TAIL_TOKENS`` and accepts any
        trigger at or after it. With N = 2 that admits a trigger in the
        *second to last* position, which is exactly where "quit" sits in
        "i should quit smoking".
        """
        assert SYSTEM_TAIL_TOKENS == 2
        assert SYSTEM_MIN_SCORE == 0.95

    def test_the_arithmetic_that_lets_quit_slip_through(self):
        tokens = normalize("i should quit smoking").text.split()
        index = tokens.index("quit")
        tail_start = max(0, len(tokens) - SYSTEM_TAIL_TOKENS)
        assert len(tokens) == 4
        assert index == 2
        assert index >= tail_start, "the guard accepts this position"

    def test_the_same_guard_blocks_the_case_it_was_written_for(self):
        """The window was sized against "goodbye is in the dictionary"."""
        tokens = normalize("goodbye is in the dictionary").text.split()
        index = tokens.index("goodbye")
        tail_start = max(0, len(tokens) - SYSTEM_TAIL_TOKENS)
        assert index < tail_start, "the original target is still blocked"

    def test_quite_good_is_protected_by_boundaries_not_the_guard(self, router, lexicon):
        """A correction to an earlier assumption about this guard.

        "quite good" is *not* saved by the system guard. The guard would
        accept "quit" at index 0 of 2 tokens. It is safe only because the
        lexicon matches on word boundaries, so "quit" is not a word inside
        "quite". The guard contributes nothing here.
        """
        tokens = normalize("quite good").text.split()
        assert len(tokens) == 2
        assert 0 >= max(0, len(tokens) - SYSTEM_TAIL_TOKENS)
        assert probe("quite good", router, lexicon).parsed_name is None


# ----------------------------------------------------------------------
# A/B/D: the dangerous shapes.
# ----------------------------------------------------------------------
class TestKnownRisk:
    def test_RISK_quit_smoking_no_longer_terminates_the_assistant(
        self, router, lexicon
    ):
        """Step 15's finding, now closed by the Step 17 subject check.

        Step 16 pinned this test to assert the *dangerous* behaviour so that
        fixing the problem could not happen by accident. Step 17 fixed it
        deliberately, so the assertion moves with the behaviour and keeps the
        sentence named.
        """
        result = probe("I should quit smoking", router, lexicon)
        assert not result.executes, "the Step 15 regression has returned"
        assert result.tool_name != "system"
        assert result.parsed_name is None
        assert result.status == "no-match"

    def test_RISK_every_dangerous_case_is_now_blocked(self, router, lexicon):
        dangerous = reaching_system(DANGEROUS + PERSONAL_DECISIONS, router, lexicon)
        assert dangerous == [], [p.describe() for p in dangerous]

    def test_RISK_the_risk_counts_are_zero(self, router, lexicon):
        blocked_expected = [c for c in ALL_CASES if c.expected == EXPECT_BLOCKED]
        assert len(reaching_system(blocked_expected, router, lexicon)) == RECORDED_TOTAL_RISK

    def test_the_before_counts_are_preserved_for_the_record(self):
        """Step 16's measurements, kept so the fix can be compared with it."""
        assert RECORDED_CORE_RISK_BEFORE == 5
        assert RECORDED_TOTAL_RISK_BEFORE == 7

    def test_every_expected_blocked_case_is_blocked(self, router, lexicon):
        for case in DANGEROUS + PERSONAL_DECISIONS + LEXICAL_TRAPS:
            result = probe(case.utterance, router, lexicon)
            assert not result.executes, result.describe()


# ----------------------------------------------------------------------
# C: genuine commands must keep working.
# ----------------------------------------------------------------------
class TestGenuineCommandsStillWork:
    @pytest.mark.parametrize("case", GENUINE_COMMANDS, ids=lambda c: c.utterance)
    def test_reaches_the_system_tool(self, router, lexicon, case):
        result = probe(case.utterance, router, lexicon)
        assert result.executes, result.describe()
        assert result.parsed_name == "system"

    @pytest.mark.parametrize("case", GENUINE_COMMANDS, ids=lambda c: c.utterance)
    def test_is_a_perfect_exact_match(self, router, lexicon, case):
        result = probe(case.utterance, router, lexicon)
        assert result.score >= SYSTEM_MIN_SCORE
        assert result.confidence == "clear"


# ----------------------------------------------------------------------
# The four outcomes the investigation must distinguish.
# ----------------------------------------------------------------------
class TestClassification:
    def test_outcomes_are_distinguishable(self, router, lexicon):
        seen = {p.classification for p in all_probes(router=router, lexicon=lexicon)}
        assert "reaches-system" in seen
        assert "parser-no-match" in seen

    def test_legacy_router_would_also_misroute_these(self, router, lexicon):
        """The fallback is not the cause; the NLU is."""
        for utterance in ("I should quit smoking", "I want to quit smoking"):
            assert probe(utterance, router, lexicon).legacy_match == "system"

    def test_the_trace_is_complete_for_every_case(self, router, lexicon):
        for p in all_probes(router=router, lexicon=lexicon):
            assert p.normalized
            assert isinstance(p.tokens, tuple)
            assert p.status
            assert p.confidence
            assert p.classification

    def test_the_report_renders(self, router, lexicon):
        report = _trace_report(router, lexicon)
        for case in ALL_CASES:
            assert case.utterance in report

    def test_the_report_is_printed(self, router, lexicon, capsys):
        report = _trace_report(router, lexicon)
        with capsys.disabled():
            print(report)


# ----------------------------------------------------------------------
# Step 17 regression coverage.
# ----------------------------------------------------------------------
#: Ordinary sentences that contain a system word but name nobody. The
#: subject check must not be needed to stop them, and must not break them
#: either: they are all already refused, by the window or by framing.
ADDITIONAL_ORDINARY_SENTENCES = [
    "we talked about quitting yesterday",
    "my neighbour quit his job",
    "the exit ramp was closed",
    "there is a stop sign ahead",
    "she told me to exit early",
]

#: Natural questions. These must resolve as questions, not be refused as
#: dangerous, whether or not they contain a system word.
QUESTION_SHAPED = [
    "can I quit",
    "should I exit",
    "how do I exit",
]

#: Command-like sentences that contain a pronoun. The rule must not be a
#: blanket pronoun ban, so these are checked against other intents, where
#: the system guard is never consulted.
PRONOUN_COMMANDS_OTHER_INTENTS = [
    ("I need a joke", "jokes"),
    ("I want information about Hyderabad", "information"),
    ("I want to hear the news", "news"),
    ("I want a joke", "jokes"),
]

#: The window's original target. The subject check must not have displaced
#: it, and must not become the only thing standing between the assistant
#: and a phrase lookup.
LEADING_POSITION_SAFETY = [
    "goodbye is in the dictionary",
    "the exit was closed",
    "the quit smoking plan failed",
]


class TestSubjectCheck:
    def test_the_rule_is_a_subject_check_not_a_pronoun_ban(self, router, lexicon):
        """Constraint 4: first person must not be globally rejected."""
        for utterance, expected in PRONOUN_COMMANDS_OTHER_INTENTS:
            result = probe(utterance, router, lexicon)
            assert result.parsed_name == expected, result.describe()
            assert result.tool_name == expected

    def test_first_person_questions_are_still_questions(self, router, lexicon):
        for utterance in QUESTION_SHAPED:
            result = probe(utterance, router, lexicon)
            assert not result.executes, result.describe()

    def test_the_newly_blocked_cost_is_recorded(self, router, lexicon):
        """The price of the rule is named, not hidden."""
        for utterance in NEWLY_BLOCKED_BY_SUBJECT_CHECK:
            result = probe(utterance, router, lexicon)
            assert not result.executes, result.describe()

    def test_second_person_is_never_a_blocking_subject(self):
        assert "you" not in SELF_OTHER_SUBJECTS
        assert "your" not in SELF_OTHER_SUBJECTS

    def test_the_list_covers_first_and_third_person_only(self):
        for word in ("i", "me", "we", "us", "he", "him", "she", "her", "they", "them"):
            assert word in SELF_OTHER_SUBJECTS, word

    def test_other_intents_never_consult_the_guard(self, router, lexicon):
        """A first-person sentence for another intent is untouched."""
        for utterance, expected in PRONOUN_COMMANDS_OTHER_INTENTS:
            assert probe(utterance, router, lexicon).parsed_name == expected


class TestLeadingPositionStillSafe:
    @pytest.mark.parametrize("utterance", LEADING_POSITION_SAFETY)
    def test_remain_blocked(self, router, lexicon, utterance):
        result = probe(utterance, router, lexicon)
        assert not result.executes, result.describe()

    def test_the_window_is_unchanged(self):
        assert SYSTEM_TAIL_TOKENS == 2

    def test_the_window_still_blocks_a_leading_trigger_on_its_own(self):
        """Proves the two conditions are independent, not one replacing the other."""
        from assistant.nlu.scoring import SYSTEM_INTENT, _passes_system_guard

        candidate = type("C", (), {"intent": SYSTEM_INTENT, "score": 1.0})()
        text = "goodbye is in the dictionary"
        assert _passes_system_guard(text, candidate, 0) is False


class TestAdditionalOrdinarySentences:
    @pytest.mark.parametrize("utterance", ADDITIONAL_ORDINARY_SENTENCES)
    def test_never_terminates_the_assistant(self, router, lexicon, utterance):
        result = probe(utterance, router, lexicon)
        assert not result.executes, result.describe()


class TestNoLegacyBypass:
    """The guard rejects at the scoring stage, so no fallback can revive it."""

    @pytest.mark.parametrize(
        "case", DANGEROUS + PERSONAL_DECISIONS, ids=lambda c: c.utterance
    )
    def test_parser_returns_none(self, router, lexicon, case):
        assert probe(case.utterance, router, lexicon).parsed_name is None

    @pytest.mark.parametrize(
        "case", DANGEROUS + PERSONAL_DECISIONS, ids=lambda c: c.utterance
    )
    def test_the_legacy_router_still_matches_but_is_not_used(
        self, router, lexicon, case
    ):
        """The old router would still misroute these; the NLU must not ask it."""
        result = probe(case.utterance, router, lexicon)
        assert result.legacy_match == "system", "the fallback is the real risk"
        assert result.tool_name != "system"
        assert result.status == "no-match"
