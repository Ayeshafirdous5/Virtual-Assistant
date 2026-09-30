"""End-to-end safety of the definition/explanation boundary (Step 20).

``tests/test_nlu_scoring.py`` covers the rule against an invented
lexicon, because that is the layer's independence promise. This file
covers the same boundary through the **real** application: the real
tools, the real lexicon, the resolver, and a check on whether
:class:`~assistant.tools.system.SystemTool` would actually execute.

Every case is checked on all five axes the step asked for:

* the parser result,
* the resolver status,
* the framing verdict,
* the final tool name,
* whether the system tool would execute.

The last one is the point. A guard that is correct in the scorer but
leaks through the resolver has not fixed anything, and ``SystemTool`` is
the only tool that can end the process, so it gets its own assertion
rather than an inference from the tool name.
"""

from __future__ import annotations

import functools

import pytest

from assistant.app import (
    STATUS_NO_MATCH,
    STATUS_RESOLVED,
    build_runtime_lexicon,
    resolve_detail,
)
from assistant.core.context import AppContext
from assistant.nlu import framing
from assistant.nlu.normalize import normalize
from assistant.nlu.parser import parse
from assistant.tools import build_default_router
from assistant.tools.system import SystemTool

#: The six sentences that used to terminate the application.
DEFINITION_FPS = (
    "what does quit mean",
    "what does exit mean",
    "what does goodbye mean",
    "what is the meaning of quit",
    "define exit",
    "explain quit",
)

#: Genuine orders, with the real tools' own patterns.
GENUINE_SYSTEM = (
    "exit",
    "quit",
    "goodbye",
    "bye",
    "please exit",
    "goodbye assistant",
    "shut down",
    "you can quit now",
    "can you exit",
    "would you quit now",
)

#: The seven Step 17 dangers.
STEP_17_FPS = (
    "I should quit smoking",
    "I want to quit smoking",
    "she decided to quit smoking",
    "he plans to quit smoking",
    "I need to quit smoking",
    "he quit smoking",
    "I want to exit early",
)

#: Ordinary sentences containing system vocabulary.
ORDINARY_SENTENCES = (
    "the exit sign is red",
    "quit your browser",
    "goodbye is in the dictionary",
    "define my own words",
    "explain the rule to me",
    "the meaning of life",
)

#: Deliberately still open after this step.
STILL_OPEN = "do you want to quit"

#: Unrelated intents, to prove nothing else moved.
UNRELATED = (
    ("tell me a joke", "jokes"),
    ("what is the weather", "weather"),
    ("top headlines", "news"),
    ("look up python", "information"),
    ("play a song", "youtube"),
    ("note buy milk", "notes"),
    ("what did I say", "history"),
    ("random fact", "facts"),
)


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


def report(utterance: str, quiet, router) -> dict:
    """All five axes for one utterance."""
    parsed = parse(utterance, _lexicon())
    resolution = resolve_detail(quiet, router, utterance, _lexicon())
    return {
        "parser": parsed.name if parsed else "no-match",
        "confidence": parsed.confidence if parsed else "-",
        "framing": framing.assess(normalize(utterance), parsed.name) if parsed else "-",
        "status": resolution.status,
        "tool": resolution.tool_name or "",
        "executes_system": isinstance(resolution.tool, SystemTool),
    }


class TestDefinitionFalsePositivesAreGone:
    @pytest.mark.parametrize("utterance", DEFINITION_FPS)
    def test_the_parser_no_longer_returns_system(self, utterance):
        parsed = parse(utterance, _lexicon())
        assert parsed is None or parsed.name != "system", utterance

    @pytest.mark.parametrize("utterance", DEFINITION_FPS)
    def test_the_resolver_reports_no_match(self, utterance, quiet, router):
        assert resolve_detail(
            quiet, router, utterance, _lexicon()
        ).status == STATUS_NO_MATCH, utterance

    @pytest.mark.parametrize("utterance", DEFINITION_FPS)
    def test_system_tool_would_not_execute(self, utterance, quiet, router):
        """The assertion that matters: no process is ended."""
        result = report(utterance, quiet, router)
        assert result["executes_system"] is False, (utterance, result)
        assert result["tool"] == "", (utterance, result)

    def test_all_six_are_covered(self):
        assert len(DEFINITION_FPS) == 6


class TestGenuineCommandsStillWork:
    @pytest.mark.parametrize("utterance", GENUINE_SYSTEM)
    def test_the_parser_returns_system(self, utterance):
        parsed = parse(utterance, _lexicon())
        assert parsed is not None, utterance
        assert parsed.name == "system", utterance
        assert parsed.confidence == "clear", utterance

    @pytest.mark.parametrize("utterance", GENUINE_SYSTEM)
    def test_the_resolver_still_executes(self, utterance, quiet, router):
        result = report(utterance, quiet, router)
        assert result["status"] == STATUS_RESOLVED, (utterance, result)
        assert result["tool"] == "system", (utterance, result)
        assert result["executes_system"] is True, (utterance, result)

    def test_all_ten_are_covered(self):
        assert len(GENUINE_SYSTEM) == 10


class TestStepSeventeenDangersStayBlocked:
    @pytest.mark.parametrize("utterance", STEP_17_FPS)
    def test_they_still_do_not_execute(self, utterance, quiet, router):
        result = report(utterance, quiet, router)
        assert result["executes_system"] is False, (utterance, result)

    @pytest.mark.parametrize("utterance", STEP_17_FPS)
    def test_the_parser_returns_nothing(self, utterance):
        assert parse(utterance, _lexicon()) is None, utterance

    def test_all_seven_are_covered(self):
        assert len(STEP_17_FPS) == 7


class TestOrdinarySentencesAreUnaffected:
    @pytest.mark.parametrize("utterance", ORDINARY_SENTENCES)
    def test_none_of_them_executes(self, utterance, quiet, router):
        result = report(utterance, quiet, router)
        assert result["executes_system"] is False, (utterance, result)

    def test_there_are_at_least_five(self):
        assert len(ORDINARY_SENTENCES) >= 5


class TestMultiWordTriggerStillWorks:
    def test_please_shut_down_executes(self, quiet, router):
        result = report("please shut down", quiet, router)
        assert result["tool"] == "system", result
        assert result["executes_system"] is True, result

    def test_shut_down_alone_executes(self, quiet, router):
        assert report("shut down", quiet, router)["tool"] == "system"

    def test_a_definition_before_shut_down_does_not_refuse_it(self, quiet, router):
        """The multi-word exemption, proved through the real matcher."""
        result = report("define shut down", quiet, router)
        assert result["tool"] == "system", result

    def test_what_does_shut_down_mean_does_not_execute(self, quiet, router):
        result = report("what does shut down mean", quiet, router)
        assert result["executes_system"] is False, result

    def test_the_marker_set_is_exactly_the_five_agreed(self):
        from assistant.nlu.scoring import DEFINITION_MARKERS

        assert DEFINITION_MARKERS == frozenset(
            {"does", "do", "of", "define", "explain"}
        )



class TestStillOpenDeliberately:
    """Recorded by Step 20, and **closed by Step 22**.

    The want-frame condition in :mod:`assistant.nlu.scoring` now refuses
    this sentence. The structural claim that made the fix possible is
    unchanged and still asserted below: the token before the trigger is
    "to", which is not a definition marker, so no amount of reading
    :data:`DEFINITION_MARKERS` would ever have caught it.
    """

    def test_it_no_longer_executes(self, quiet, router):
        result = report(STILL_OPEN, quiet, router)
        assert result["executes_system"] is False, result
        assert result["tool"] != "system", result

    def test_the_reason_it_was_structural(self):
        """The token before the trigger is "to", which is not a marker."""
        from assistant.nlu.scoring import DEFINITION_MARKERS

        tokens = normalize(STILL_OPEN).tokens
        assert tokens[tokens.index("quit") - 1] == "to"
        assert "to" not in DEFINITION_MARKERS

    def test_no_general_question_rule_was_added(self):
        """Constraint of this step, asserted."""
        from assistant.nlu.scoring import SYSTEM_MIN_SCORE, SYSTEM_TAIL_TOKENS

        assert SYSTEM_TAIL_TOKENS == 2
        assert SYSTEM_MIN_SCORE == 0.95


class TestNothingElseMoved:
    @pytest.mark.parametrize("utterance,intent", UNRELATED)
    def test_no_unrelated_intent_regressed(self, utterance, intent, quiet, router):
        result = report(utterance, quiet, router)
        assert result["parser"] == intent, (utterance, result)
        assert result["status"] == STATUS_RESOLVED, (utterance, result)

    def test_no_legacy_router_bypass(self, quiet, router):
        """The resolver must refuse, not fall through to a substring match."""
        for utterance in DEFINITION_FPS + STEP_17_FPS:
            resolution = resolve_detail(quiet, router, utterance, _lexicon())
            assert resolution.status != "fallback", utterance
            assert resolution.tool is None, utterance

    def test_only_scoring_changed(self):
        """The parser, lexicon and framing modules are untouched here."""
        from assistant.nlu.lexicon import COMMON_ALIASES
        from assistant.nlu.parser import AMBIGUITY_MARGIN, MIN_CONFIDENCE
        from assistant.nlu.scoring import FUZZY_CUTOFF

        assert MIN_CONFIDENCE == 0.65
        assert AMBIGUITY_MARGIN == 0.05
        assert FUZZY_CUTOFF == 0.80
        assert COMMON_ALIASES["system"] == (
            "stop the assistant",
            "stop listening",
            "shut down",
            "that is all",
        )

