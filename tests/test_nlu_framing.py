"""Tests for the contextual framing layer.

The framing layer answers one question that scoring cannot: **is the user
asking, or just talking about the topic?** It is deliberately small, so
these tests focus on the two directions that matter.

* A genuine request must never be blocked. That is the expensive mistake,
  because it silently breaks a command that used to work.
* A conversational mention must not reach a tool.

The two known false positives that motivated the layer are asserted
directly, and the whole existing suite is the regression net for
everything else.
"""

from __future__ import annotations

import pytest

from assistant.app import (
    STATUS_REJECTED,
    build_runtime_lexicon,
    resolve_detail,
)
from assistant.core.context import AppContext
from assistant.nlu import framing
from assistant.nlu.normalize import normalize
from assistant.tools import build_default_router


class Silent:
    """A speaker and listener that do nothing, so tests stay hermetic."""

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


@pytest.fixture
def quiet():
    ctx = AppContext()
    ctx.speaker = Silent()
    ctx.listener = Silent()
    return ctx


def verdict(utterance, intent):
    return framing.assess(normalize(utterance), intent)


# ----------------------------------------------------------------------
# The two known false positives
# ----------------------------------------------------------------------
class TestKnownFalsePositives:
    @pytest.mark.parametrize(
        "utterance,intent",
        [
            ("I want to log this information", "information"),
            ("the weather is nice today I guess", "weather"),
        ],
    )
    def test_judged_a_mention(self, utterance, intent):
        assert verdict(utterance, intent) == framing.MENTION

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            ("I want to log this information", "information"),
            ("the weather is nice today I guess", "weather"),
        ],
    )
    def test_no_tool_runs(self, quiet, router, lexicon, utterance, intent):
        resolution = resolve_detail(quiet, router, utterance, lexicon)
        assert resolution.tool is None
        assert resolution.status == STATUS_REJECTED
        assert resolution.framing == framing.MENTION

    def test_rejection_never_falls_back_to_the_legacy_router(self, quiet, router, lexicon):
        """The old router would have matched both; framing must not use it."""
        assert router.find_match("I want to log this information").tool_name
        resolution = resolve_detail(
            quiet, router, "I want to log this information", lexicon
        )
        assert resolution.tool is None
        assert resolution.tool_name == ""


# ----------------------------------------------------------------------
# Genuine requests must survive
# ----------------------------------------------------------------------
class TestGenuineRequestsSurvive:
    @pytest.mark.parametrize(
        "utterance",
        [
            "what is the weather",
            "tell me the weather",
            "check the weather",
            "show me the weather",
            "weather please",
            "show me the logs",
        ],
    )
    def test_named_commands_are_requests(self, utterance):
        assert verdict(utterance, "weather") != framing.MENTION
        assert verdict(utterance, "history") != framing.MENTION

    @pytest.mark.parametrize(
        "utterance",
        [
            "what is the weather",
            "tell me the weather",
            "check the weather",
            "show me the weather",
            "weather please",
        ],
    )
    def test_named_commands_still_resolve(self, quiet, router, lexicon, utterance):
        tool = resolve_detail(quiet, router, utterance, lexicon).tool
        assert tool is not None
        assert tool.name == "weather"

    @pytest.mark.parametrize(
        "utterance,expected",
        [
            ("weather", "weather"),
            ("what is the weather", "weather"),
            ("wether", "weather"),
            ("tell me the news", "news"),
            ("top headlines", "news"),
            ("joke", "jokes"),
            ("tell me a joke", "jokes"),
            ("I need a joke", "jokes"),
            ("make me laugh", "jokes"),
            ("information about python", "information"),
            ("I want information about Hyderabad", "information"),
            ("tell me about Python", "information"),
            ("play lofi beats", "youtube"),
            ("play a song", "youtube"),
            ("history", "history"),
            ("what did I ask", "history"),
            ("can you read my history", "history"),
            ("note buy milk", "notes"),
            ("notes", "notes"),
            ("list notes", "notes"),
            ("show me my notes", "notes"),
            ("save this as a note", "notes"),
            ("exit", "system"),
            ("quit", "system"),
            ("goodbye", "system"),
            ("please exit", "system"),
        ],
    )
    def test_important_commands_are_unaffected(
        self, quiet, router, lexicon, utterance, expected
    ):
        resolution = resolve_detail(quiet, router, utterance, lexicon)
        assert resolution.tool is not None, utterance
        assert resolution.tool_name == expected, utterance


# ----------------------------------------------------------------------
# Conversational mentions
# ----------------------------------------------------------------------
class TestConversationalMentions:
    @pytest.mark.parametrize(
        "utterance,intent",
        [
            ("the weather is nice today", "weather"),
            ("my notes are on the desk", "notes"),
            ("this is a good joke", "jokes"),
            ("the news is bad today", "news"),
            ("that was a strange fact", "facts"),
            ("the history of the place is long", "history"),
        ],
    )
    def test_declarative_sentences_are_mentions(self, utterance, intent):
        assert verdict(utterance, intent) == framing.MENTION

    @pytest.mark.parametrize(
        "utterance",
        [
            "the weather is nice today",
            "my notes are on the desk",
            "the news is bad today",
        ],
    )
    def test_mentions_run_no_tool(self, quiet, router, lexicon, utterance):
        assert resolve_detail(quiet, router, utterance, lexicon).tool is None

    def test_a_leading_copula_is_still_a_question(self):
        # "is the weather good" asks. Only position tells the two apart.
        assert verdict("is the weather good", "weather") != framing.MENTION

    def test_hedging_alone_marks_a_remark(self):
        assert verdict("I suppose the weather is fine", "weather") == framing.MENTION


# ----------------------------------------------------------------------
# The layer's own contract
# ----------------------------------------------------------------------
class TestFramingContract:
    def test_only_three_verdicts_exist(self):
        assert {framing.REQUEST, framing.MENTION, framing.NEUTRAL} == {
            "request",
            "mention",
            "neutral",
        }

    def test_bare_commands_are_neutral_not_blocked(self):
        # No cue at all must mean "no opinion", never "do not proceed".
        for utterance in ("weather", "exit", "note", "news", "history"):
            assert verdict(utterance, "weather") == framing.NEUTRAL

    def test_want_and_need_are_not_request_cues(self):
        # They introduce competing actions, so treating them as request
        # cues would make the competing-action rule unreachable.
        assert "want" not in framing.REQUEST_CUES
        assert "need" not in framing.REQUEST_CUES

    def test_empty_utterance_is_neutral(self):
        assert framing.assess(normalize(""), "weather") == framing.NEUTRAL

    def test_deterministic(self):
        for utterance in (
            "what is the weather",
            "the weather is nice",
            "log this information",
        ):
            first = verdict(utterance, "information")
            assert first == verdict(utterance, "information")

    def test_is_independent_of_the_application(self):
        import ast
        import pathlib

        tree = ast.parse(pathlib.Path(framing.__file__).read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        # It may use the rest of the NLU, but nothing from the application.
        forbidden = (
            "assistant.tools",
            "assistant.core",
            "assistant.app",
            "assistant.speech",
            "assistant.storage",
        )
        assert all(not name.startswith(forbidden) for name in imported), imported

    def test_imports_nothing_heavy(self):
        import sys

        forbidden = {"requests", "selenium", "pyttsx3", "randfacts", "nltk", "spacy"}
        assert not ({m.split(".")[0] for m in sys.modules} & forbidden)


# ----------------------------------------------------------------------
# Preserved behaviour
# ----------------------------------------------------------------------
class TestExistingBehaviourPreserved:
    def test_negation_still_blocks(self, quiet, router, lexicon):
        for utterance in (
            "do not play music",
            "never play music",
            "I don't want to exit",
        ):
            assert resolve_detail(quiet, router, utterance, lexicon).tool is None

    def test_ambiguity_still_asks(self, quiet, router, lexicon):
        resolution = resolve_detail(
            quiet, router, "information about the news", lexicon
        )
        assert resolution.status == "ambiguous"

    def test_fuzzy_recovery_still_works(self, quiet, router, lexicon):
        for utterance, expected in (("wether", "weather"), ("joks", "jokes")):
            tool = resolve_detail(quiet, router, utterance, lexicon).tool
            assert tool.name == expected

    def test_confidence_scoring_is_untouched(self, lexicon):
        """Framing must not change what the parser reports."""
        from assistant.nlu.parser import parse

        for utterance in ("the weather is nice today", "what is the weather"):
            parsed = parse(utterance, lexicon)
            assert parsed is not None
            assert parsed.name == "weather"
            assert parsed.score == 1.0
            assert parsed.confidence == "clear"

    def test_framing_is_separate_from_confidence(self, quiet, router, lexicon):
        resolution = resolve_detail(
            quiet, router, "the weather is nice today", lexicon
        )
        # The parser was confident; framing is a separate, later veto.
        assert resolution.parsed.confidence == "clear"
        assert resolution.framing == framing.MENTION
        assert resolution.tool is None

    def test_constants_are_unchanged(self):
        from assistant.nlu.parser import AMBIGUITY_MARGIN, MIN_CONFIDENCE
        from assistant.nlu.scoring import FUZZY_CUTOFF

        assert MIN_CONFIDENCE == 0.65
        assert AMBIGUITY_MARGIN == 0.05
        assert FUZZY_CUTOFF == 0.80


# ----------------------------------------------------------------------
# The narrative guard (Step 3)
# ----------------------------------------------------------------------
class TestNarrativeGuard:
    """Past-tense reporting is a remark; only explicit past verbs count."""

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            ("I mentioned the logs earlier", "history"),
            ("I heard the news this morning", "news"),
            ("my friend told me a joke", "jokes"),
            ("we discussed the news yesterday", "news"),
        ],
    )
    def test_the_four_known_gaps_are_now_blocked(self, utterance, intent):
        assert verdict(utterance, intent) == framing.MENTION

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            ("the four known gaps are now blocked", "history"),
            ("we talked about the weather", "weather"),
            ("she reported the news to me", "news"),
            ("he explained the joke to me", "jokes"),
            ("they discussed the weather at home", "weather"),
            ("my boss announced the news", "news"),
        ],
    )
    def test_additional_narrative_mentions_are_blocked(self, utterance, intent):
        assert verdict(utterance, intent) == framing.MENTION

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            ("can you read my history", "history"),
            ("what did I say", "history"),
            ("did you hear the news", "news"),
            ("tell me the news", "news"),
            ("tell me a joke", "jokes"),
            ("read my saved notes", "notes"),
        ],
    )
    def test_near_neighbours_are_not_blocked(self, utterance, intent):
        """"read", "say", "hear" and "tell" must stay usable."""
        assert verdict(utterance, intent) != framing.MENTION

    @pytest.mark.parametrize(
        "utterance",
        ["weather today", "today's weather", "the news", "some music", "a joke"],
    )
    def test_fragments_are_still_neutral(self, utterance):
        assert verdict(utterance, "weather") == framing.NEUTRAL

    @pytest.mark.parametrize(
        "utterance",
        [
            "what is the weather",
            "tell me the weather",
            "check the weather",
            "show me the logs",
            "tell me a joke",
            "play some music",
            "get today's news",
            "please exit",
        ],
    )
    def test_clear_requests_are_still_requests(self, utterance):
        assert verdict(utterance, "weather") != framing.MENTION
        assert verdict(utterance, "history") != framing.MENTION

    @pytest.mark.parametrize(
        "utterance",
        [
            "is the weather good",
            "is today's weather okay",
            "what do you think about the weather",
            "did you hear the news",
        ],
    )
    def test_questions_still_win_over_the_guard(self, utterance):
        """A question opener is checked earlier, so it must still win."""
        assert verdict(utterance, "weather") == framing.REQUEST
        assert verdict(utterance, "news") == framing.REQUEST

    def test_a_request_cue_beats_a_narrative_verb(self):
        """Rule ordering: a real request must not be talked out of running."""
        assert verdict("tell me the news I mentioned", "news") == framing.REQUEST

    def test_time_expressions_alone_do_not_block(self):
        """A bare past time word must not turn a fragment into a mention."""
        for utterance in ("weather yesterday", "the news earlier", "jokes today"):
            assert verdict(utterance, "weather") != framing.MENTION

    def test_read_and_say_are_excluded_from_the_verb_list(self):
        # Documented exclusions: each would break a real command.
        assert "read" not in framing.NARRATIVE_VERBS
        assert "say" not in framing.NARRATIVE_VERBS
        assert "tell" not in framing.NARRATIVE_VERBS

    def test_the_verb_list_holds_only_past_forms(self):
        """No base form may slip in, or a command would be blocked."""
        base_forms = {
            "mention", "hear", "tell", "discuss", "chat", "talk", "report",
            "announce", "explain", "confirm", "comment",
        }
        assert not (base_forms & framing.NARRATIVE_VERBS)

    def test_the_guard_does_not_override_a_confident_parse(self, quiet, router, lexicon):
        """Blocking is a routing veto; the parser still reports its intent."""
        from assistant.nlu.parser import parse

        parsed = parse("my friend told me a joke", lexicon)
        assert parsed is not None
        assert parsed.name == "jokes"
        assert parsed.confidence == "clear"
        assert resolve_detail(quiet, router, "my friend told me a joke", lexicon).tool is None
