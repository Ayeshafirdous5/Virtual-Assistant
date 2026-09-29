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
    SYSTEM_MIN_SCORE,
    SYSTEM_TAIL_TOKENS,
    score_intents,
)
from assistant.tools import build_default_router

#: The user meant "do not run any tool".
EXPECT_BLOCKED = "blocked"
#: The user meant "shut the assistant down"; the system tool should run.
EXPECT_EXIT = "exit"

#: How many dangerous sentences are known to reach the system tool today.
#: Pinned so the risk cannot shrink or grow unnoticed. Measured, not
#: guessed: 5 of the 6 personal-decision sentences, and 7 once the subtler
#: lexical traps are included.
RECORDED_CORE_RISK = 5
RECORDED_TOTAL_RISK = 7


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
    def test_RISK_quit_smoking_terminates_the_assistant(self, router, lexicon):
        """Step 15's finding, pinned so it cannot be forgotten.

        This asserts the *current, dangerous* behaviour on purpose. It is
        written so that fixing the problem makes this test fail, forcing the
        fix to be deliberate and this file to be updated with it.
        """
        result = probe("I should quit smoking", router, lexicon)
        assert result.executes, (
            "the risk no longer reproduces; update this file deliberately"
        )
        assert result.tool_name == "system"
        assert result.parsed_name == "system"
        assert result.score == 1.0
        assert result.confidence == "clear"

    def test_RISK_the_guard_is_what_lets_it_through(self, router, lexicon):
        result = probe("I should quit smoking", router, lexicon)
        assert result.classification == "reaches-system"
        assert result.framing == framing.NEUTRAL
        assert result.status == "resolved"

    def test_known_risk_count_is_pinned(self, router, lexicon):
        dangerous = reaching_system(DANGEROUS + PERSONAL_DECISIONS, router, lexicon)
        assert len(dangerous) == RECORDED_CORE_RISK, [p.describe() for p in dangerous]

    def test_risk_includes_the_step_15_sentence(self, router, lexicon):
        spoken = {p.utterance for p in reaching_system(router=router, lexicon=lexicon)}
        assert "I should quit smoking" in spoken

    def test_every_expected_blocked_case_is_blocked_or_a_pinned_risk(self, router, lexicon):
        """Safety is asserted, but known gaps are named rather than hidden."""
        for case in DANGEROUS + PERSONAL_DECISIONS + LEXICAL_TRAPS:
            result = probe(case.utterance, router, lexicon)
            if result.executes:
                continue  # pinned above as a counted risk
            assert result.classification in (
                "parser-no-match",
                "blocked-by-framing",
                "blocked-by-parser-or-guard",
            ), result.describe()

    def test_lexical_trap_risk_count(self, router, lexicon):
        """Pinned, not asserted away. Several reach the system tool."""
        risky = reaching_system(LEXICAL_TRAPS, router, lexicon)
        assert len(risky) >= 2, [p.describe() for p in risky]

    def test_total_dangerous_count_across_all_groups(self, router, lexicon):
        blocked_expected = [
            c for c in ALL_CASES if c.expected == EXPECT_BLOCKED
        ]
        risky = reaching_system(blocked_expected, router, lexicon)
        assert len(risky) == RECORDED_TOTAL_RISK, [p.describe() for p in risky]


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
