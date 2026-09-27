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


# ----------------------------------------------------------------------
# Trailing questions (Step 5)
# ----------------------------------------------------------------------
class TestTrailingQuestions:
    """A question behind a narrative clause must still be a request.

    ``_opens_a_question`` only ever inspects the start of an utterance, so
    "we discussed the weather, what is it now" opened with a narrative
    clause, the trailing question was never seen, and the remark rules
    below blocked a genuine request. The question is now recognised by the
    inversion inside its clause, and it is checked before every remark
    rule, which is the ordering principle the layer already promised:
    an explicit ask is never talked out of running.
    """

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            ("we discussed the weather, what's it like now", "weather"),
            ("I mentioned the weather, what is it now", "weather"),
        ],
    )
    def test_the_two_known_false_blocks_are_requests(self, utterance, intent):
        assert verdict(utterance, intent) == framing.REQUEST

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            ("we discussed the weather, what's it like now", "weather"),
            ("I mentioned the weather, what is it now", "weather"),
        ],
    )
    def test_the_two_known_false_blocks_run_their_tool(
        self, quiet, router, lexicon, utterance, intent
    ):
        """The whole pipeline, because a block is a command that stopped."""
        resolution = resolve_detail(quiet, router, utterance, lexicon)
        assert resolution.framing == framing.REQUEST
        assert resolution.tool is not None, utterance
        assert resolution.tool.name == intent

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            ("we talked about the news, what did I miss", "news"),
            ("they discussed the logs, what did I ask then", "history"),
            ("she reported the weather, is it still cold", "weather"),
            ("my friend mentioned the joke, did you see it", "jokes"),
            ("he commented on the weather, how is it looking", "weather"),
        ],
    )
    def test_more_narrative_clauses_carrying_a_trailing_question(
        self, utterance, intent
    ):
        assert verdict(utterance, intent) == framing.REQUEST

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            # A narrative clause and nothing after it: still a remark.
            ("we discussed the news yesterday", "news"),
            ("I mentioned the logs earlier", "history"),
            ("my friend told me a joke", "jokes"),
            ("she reported the weather to me", "weather"),
            ("he confirmed the joke was good", "jokes"),
            # The trap. These carry a question word and invert nothing, so
            # they must not be mistaken for the cases above. This is the
            # difference between reading for structure and searching for
            # "what", "is" or "did".
            ("I mentioned what I heard about the news", "news"),
            ("we discussed what was funny", "jokes"),
            ("he explained what I had told him", "news"),
        ],
    )
    def test_narrative_statements_are_still_mentions(self, utterance, intent):
        assert verdict(utterance, intent) == framing.MENTION

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            ("weather", "weather"),
            ("today's weather", "weather"),
            ("the news", "news"),
            ("some music", "youtube"),
            ("a joke", "jokes"),
            ("the logs", "history"),
        ],
    )
    def test_fragments_are_still_neutral(self, utterance, intent):
        assert verdict(utterance, intent) == framing.NEUTRAL

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            ("what is the weather", "weather"),
            ("is the weather good", "weather"),
            ("did you hear the news", "news"),
            ("what did I say", "history"),
            ("how was your day", "news"),
        ],
    )
    def test_leading_questions_are_untouched(self, utterance, intent):
        """The leading check still runs first and must not have moved."""
        assert verdict(utterance, intent) == framing.REQUEST

    @pytest.mark.parametrize(
        "utterance",
        [
            "tell me the weather",
            "check the weather",
            "show me the logs",
            "play some music",
            "get today's news",
            "please exit",
        ],
    )
    def test_request_cues_are_untouched(self, utterance):
        assert verdict(utterance, "weather") != framing.MENTION

    def test_the_two_question_sets_partition_the_interrogatives(self):
        """Leading and trailing must agree on what interrogative means."""
        assert (
            framing.WH_QUESTIONS | framing.INVERTING_AUXILIARIES
            == framing.INTERROGATIVE
        )
        assert not (framing.WH_QUESTIONS & framing.INVERTING_AUXILIARIES)

    def test_inverted_subjects_are_pronouns_not_determiners(self):
        """"is the" is not evidence: remarks open with it as well."""
        assert "the" not in framing.INVERTED_SUBJECTS
        assert "a" not in framing.INVERTED_SUBJECTS
        assert "i" in framing.INVERTED_SUBJECTS

    def test_only_later_positions_count_as_trailing(self):
        """Position, not vocabulary, is what separates the two checks."""
        assert framing._has_trailing_question(
            normalize("what is the weather").tokens
        ) is False
        assert framing._has_trailing_question(
            normalize("we discussed the weather, what is it now").tokens
        ) is True


# ----------------------------------------------------------------------
# The reported-"tell" guard (Step 7)
# ----------------------------------------------------------------------
class TestReportedTellGuard:
    """``tell`` is a request cue *and* an ordinary reporting verb.

    The cue is what carries "tell me a joke", and it is also what made
    eight reporting sentences look like requests. This guard vetoes the
    cue only where the structure says the sentence is describing an
    event, and it is the reason the tell corpus sits at 55 of 57.
    """

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            # Pattern 1: a perception verb, its subject, then "tell".
            ("I heard you tell a joke", "jokes"),
            ("I heard him tell a joke", "jokes"),
            ("I heard her tell the news", "news"),
            ("I saw you tell a joke", "jokes"),
            ("I saw him tell the news", "news"),
            # Pattern 3: a first-position subject, "tell", and a frequency.
            ("I tell you a joke every morning", "jokes"),
            ("you tell me that every day", "jokes"),
            # Pattern 2: a reported command, "to tell".
            ("he told me to tell you a joke", "jokes"),
        ],
    )
    def test_the_eight_cue_leaks_are_mentions(self, utterance, intent):
        assert verdict(utterance, intent) == framing.MENTION

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            ("I heard you tell a joke", "jokes"),
            ("I heard him tell a joke", "jokes"),
            ("I saw you tell a joke", "jokes"),
            ("he told me to tell you a joke", "jokes"),
        ],
    )
    def test_the_closed_gap_runs_no_tool(self, quiet, router, lexicon, utterance, intent):
        resolution = resolve_detail(quiet, router, utterance, lexicon)
        assert resolution.tool is None, utterance
        assert resolution.framing == framing.MENTION

    def test_a_habitual_with_no_trigger_word_never_matches_at_all(
        self, quiet, router, lexicon
    ):
        """"you tell me that every day" names no trigger, so it stops earlier.

        Worth its own test because it would be easy to assume the guard
        did the work. It did not: the parser finds nothing to look up, so
        framing is never consulted and the answer is ``no-match``. The
        guard still judges the sentence correctly, which the framing test
        above covers; this one is about the pipeline as a whole.
        """
        resolution = resolve_detail(
            quiet, router, "you tell me that every day", lexicon
        )
        assert resolution.parsed is None
        assert resolution.framing == ""
        assert resolution.tool is None
        assert resolution.status == "no-match"

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            ("tell me a joke", "jokes"),
            ("tell me the news", "news"),
            ("tell me about the weather", "weather"),
            ("can you tell me a joke?", "jokes"),
            ("please tell me the news", "news"),
            ("tell me what you heard", "news"),
            ("tell me about Python", "information"),
            ("tell me a joke please", "jokes"),
            ("tell me what's on the news", "news"),
            ("tell me again", "jokes"),
        ],
    )
    def test_genuine_tell_requests_are_untouched(self, utterance, intent):
        """Requirement 1: the cue is the whole point of these sentences."""
        assert verdict(utterance, intent) == framing.REQUEST

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            ("what did I tell you earlier?", "history"),
            ("what did she tell you?", "news"),
            ("did he tell you the news?", "news"),
            ("did you hear me tell the news?", "news"),
            ("when did I tell you that?", "history"),
            ("did I tell you about the weather?", "weather"),
            ("who told you the news?", "news"),
            ("have I told you about this?", "history"),
            ("can you tell me a joke?", "jokes"),
            ("would you tell me the news?", "news"),
            ("could you tell me a joke, please?", "jokes"),
        ],
    )
    def test_questions_about_past_events_still_work(self, utterance, intent):
        """Requirement 2: a question outranks a reporting verb."""
        assert verdict(utterance, intent) == framing.REQUEST

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            ("I heard you tell a joke, tell me another one", "jokes"),
            ("she told me the news, now tell me today's news", "news"),
            ("I heard you tell that story, can you tell it again?", "jokes"),
            ("you told me a joke earlier, tell me another", "jokes"),
            ("I heard him tell a joke, do you have another one?", "jokes"),
            ("my friend told me a joke, tell me a better one", "jokes"),
        ],
    )
    def test_a_request_after_narrative_context_survives(self, utterance, intent):
        """Requirement 4: only the reporting "tell" is vetoed.

        "I heard him tell a joke, do you have another one?" is the sharp
        one: its only "tell" is the reporting one, and it is still a
        request because the second half is a question.
        """
        assert verdict(utterance, intent) == framing.REQUEST


    @pytest.mark.parametrize(
        "utterance,intent",
        [
            # Past "told" and the copula rules were already correct, and
            # must stay correct while the "tell" guard is in place.
            ("he told me a joke", "jokes"),
            ("my friend told me a joke", "jokes"),
            ("I saw the joke you told me", "jokes"),
            ("we talked about what he told me", "jokes"),
            ("I heard the news you told me about", "news"),
            ("she told me the weather was cold", "weather"),
        ],
    )
    def test_nearby_narrative_cases_are_still_mentions(self, utterance, intent):
        assert verdict(utterance, intent) == framing.MENTION

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            # "telling" is in neither verb list, and the Step 9 guard only
            # ever read "tells". Step 11 added a separate recall rule, so
            # this sentence is no longer a fall-through.
            ("I remember you telling me a joke", "jokes"),
        ],
    )
    def test_tells_is_no_longer_a_fallthrough(self, utterance, intent):
        """Step 9 closed the "tells" half and Step 11 the recall half."""
        assert verdict(utterance, intent) == framing.MENTION

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            # A gerund that is not a recall, and is not caught by anything.
            # Step 9 could not reach it and Step 11 deliberately did not
            # either: it is a request the user is making.
            ("keep telling me more", "jokes"),
        ],
    )
    def test_the_gerund_still_falls_through_when_it_is_an_order(
        self, utterance, intent
    ):
        assert verdict(utterance, intent) == framing.NEUTRAL

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            ("tell me", "jokes"),
            ("I want to tell you a joke", "jokes"),
            ("I need to tell you something", "news"),
            ("tell me about the history", "history"),
            ("tell me the weather", "weather"),
            ("tell me what time it is", "system"),
        ],
    )
    def test_nearby_request_cases_are_unchanged(self, utterance, intent):
        """The guard is narrow: a "tell" that is not reporting is a cue."""
        assert verdict(utterance, intent) != framing.MENTION

    def test_a_pronoun_before_tell_is_not_enough(self):
        """Requirement 6, asserted directly.

        Every sentence below has a pronoun immediately before "tell" and
        none of them is reported speech. A rule of the shape "pronoun
        before tell -> MENTION" would break all of them, which is why
        every branch of the guard needs a second, independent mark.
        """
        for utterance in (
            "can you tell me a joke?",
            "what did she tell you?",
            "did he tell you the news?",
            "tell me a joke",
        ):
            tokens = normalize(utterance).tokens
            index = tokens.index("tell")
            assert index == 0 or tokens[index - 1] in framing.INVERTED_SUBJECTS
            assert framing._reports_speech(tokens, index) is False, utterance

    def test_the_guard_only_ever_looks_at_tell(self):
        """No other request cue can be vetoed by this rule."""
        for token in sorted(framing.REQUEST_CUES - {"tell"}):
            assert not framing._reports_speech((token,), 0), token

    def test_tell_is_still_a_request_cue(self):
        """Requirement 5: the cue was narrowed per occurrence, not removed."""
        assert "tell" in framing.REQUEST_CUES
        assert verdict("tell me a joke", "jokes") == framing.REQUEST

    def test_the_narrative_verb_list_is_unchanged(self):
        """This step vetoes the cue; it does not extend the verb list."""
        assert framing.NARRATIVE_VERBS == frozenset(
            {
                "mentioned", "heard", "told", "discussed", "chatted", "talked",
                "reported", "announced", "explained", "confirmed", "commented",
            }
        )
        for word in ("tells", "telling", "saw", "remember", "ask"):
            assert word not in framing.NARRATIVE_VERBS

    def test_a_reported_infinitive_needs_a_reporting_verb(self):
        """"I want to tell you a joke" shares the shape and is spared."""
        tokens = normalize("I want to tell you a joke").tokens
        index = tokens.index("tell")
        assert tokens[index - 1] == "to"
        assert framing._reports_speech(tokens, index) is False

    def test_a_habitual_needs_a_first_position_subject(self):
        """"tell me a joke every day" is a request, not a habit statement."""
        tokens = normalize("tell me a joke every day").tokens
        assert tokens[0] == "tell"
        assert framing._reports_speech(tokens, 0) is False
        assert verdict("tell me a joke every day", "jokes") == framing.REQUEST


# ----------------------------------------------------------------------
# The declarative-"tells" guard (Step 9)
# ----------------------------------------------------------------------
class TestDeclaresTellsGuard:
    """``tells`` is in neither verb list, and used to fall through.

    It is the present third person of a verb the layer already reads as
    a remark in the past, so "she tells me the news" is a statement about
    speech. The guard keys on **a subject in front of the verb**, which
    an English order to the assistant never has.
    """

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            # The five cases Step 8 measured as genuine framing leaks.
            ("my brother tells me jokes", "jokes"),
            ("she tells me the news", "news"),
            ("my friend tells me about the weather", "weather"),
            ("my dad tells me the same joke daily", "jokes"),
            ("the teacher tells us the weather", "weather"),
        ],
    )
    def test_the_five_target_cases_are_mentions(self, utterance, intent):
        assert verdict(utterance, intent) == framing.MENTION

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            # More of the same shape, from the verb-form corpus.
            ("he tells everyone the story", "jokes"),
            ("she tells me what she had for lunch", "jokes"),
            ("he tells me the weather is fine", "weather"),
            ("it tells you nothing", "news"),
            ("my cat tells me about the weather", "weather"),
            ("the sign tells you the weather", "weather"),
        ],
    )
    def test_more_third_person_tells_are_mentions(self, utterance, intent):
        assert verdict(utterance, intent) == framing.MENTION

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            # A request cue outranks the declarative subject.
            ("my brother tells me jokes, tell me one too", "jokes"),
            ("he tells me the weather, check it for me", "weather"),
            ("she tells me the news, show me the headlines", "news"),
            # A question outranks it too.
            ("who tells you the weather?", "weather"),
            ("does he tell me the news?", "news"),
        ],
    )
    def test_a_request_or_question_beats_the_subject(self, utterance, intent):
        assert verdict(utterance, intent) == framing.REQUEST

    def test_the_copula_also_catches_these(self):
        """A declarative about the weather is a remark twice over.

        "he tells me the weather is fine" has both the subject that this
        step added and the copula that was always there, so it is worth
        pinning that the older rule still fires on its own.
        """
        tokens = normalize("he tells me the weather is fine").tokens
        assert framing._declares_habit(tokens) is True
        assert any(token in framing.COPULA_CUES for token in tokens)

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            ("tell me a joke", "jokes"),
            ("tell me the news", "news"),
            ("can you tell me a joke?", "jokes"),
            ("please tell me the news", "news"),
            ("tell me what you heard", "news"),
            ("what did she tell you?", "news"),
            ("did he tell you the news?", "news"),
            ("who told you the news?", "news"),
        ],
    )
    def test_genuine_requests_are_never_blocked(self, utterance, intent):
        """The rule reads "tells" only, so the bare cue is untouched."""
        assert verdict(utterance, intent) == framing.REQUEST

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            ("tells", "jokes"),
            ("the news", "news"),
            ("weather", "weather"),
            ("a joke", "jokes"),
            ("some music", "youtube"),
        ],
    )
    def test_fragments_are_still_neutral(self, utterance, intent):
        """A bare "tells" has nothing in front of it, so it cannot match."""
        assert verdict(utterance, intent) == framing.NEUTRAL


    def test_tells_alone_is_not_enough(self):
        """Requirement: the word must not be sufficient on its own.

        Each of these has "tells" and no subject in front of it, so
        nothing about the rule fires and the sentence is left exactly as
        it was before this step.
        """
        for utterance in (
            "tells me a joke",
            "tells the news",
            "i tells me jokes",
            "you tells me jokes",
        ):
            tokens = normalize(utterance).tokens
            assert framing._declares_habit(tokens) is False, utterance
            assert verdict(utterance, "jokes") == framing.NEUTRAL

    def test_only_the_literal_form_is_matched(self):
        """No stemming: nothing ending in "-s" is inferred."""
        for utterance in ("shout", "yells", "sells", "calls", "brings"):
            tokens = normalize(utterance).tokens
            assert framing._declares_habit(tokens) is False, utterance

    def test_the_search_does_not_cross_a_phrase_break(self):
        """A determiner in an earlier phrase is not a subject."""
        for utterance in (
            "a joke about my brother tells",
            "read the news about my dad tells",
        ):
            tokens = normalize(utterance).tokens
            assert framing._declares_habit(tokens) is False, utterance

    def test_interrogatives_are_not_treated_as_subjects(self):
        """"who tells you the weather?" must stay a question."""
        for word in ("who", "what", "when", "where", "why", "how"):
            assert word not in framing.TELLS_SUBJECT_DETERMINERS
            assert word not in framing.TELLS_SUBJECT_PRONOUNS
        assert framing._declares_habit(
            normalize("who tells you the weather").tokens
        ) is False

    def test_the_verb_lists_are_unchanged(self):
        """The form is handled by a rule, not by editing the lists."""
        assert "tells" not in framing.REQUEST_CUES
        assert "tells" not in framing.NARRATIVE_VERBS
        assert "telling" not in framing.NARRATIVE_VERBS
        assert framing.NARRATIVE_VERBS == frozenset(
            {
                "mentioned", "heard", "told", "discussed", "chatted", "talked",
                "reported", "announced", "explained", "confirmed", "commented",
            }
        )

    def test_telling_was_left_out_of_this_step(self):
        """Step 9 read "tells" only, and still does.

        The gerund is handled by a different rule added in Step 11, so
        nothing here was widened to reach it.
        """
        assert verdict("keep telling me more", "jokes") == framing.NEUTRAL
        assert framing._declares_habit(
            normalize("I remember you telling me a joke").tokens
        ) is False


# ----------------------------------------------------------------------
# The recall + "telling" guard (Step 11)
# ----------------------------------------------------------------------
class TestRecallTellGuard:
    """``telling`` is a remark in the past and an order in the present.

    The gerund sits in both directions at once, so it cannot be judged on
    its own. This guard matches one whole clause instead: a recall verb in
    front of a subject and the gerund. That is the only shape Step 10
    showed to be a leak, and the request side of the gerund is not in it.
    """

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            # The four gaps Step 10 measured as reachable framing leaks.
            ("I remember you telling me a joke", "jokes"),
            ("I remember him telling me the news", "news"),
            ("I recall him telling me the news", "news"),
            ("I remember you telling me the weather", "weather"),
        ],
    )
    def test_the_four_recall_targets_are_mentions(self, utterance, intent):
        assert verdict(utterance, intent) == framing.MENTION

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            # More of the same clause shape, so the rule is not pinned to
            # four literal sentences.
            ("I recall you telling me that yesterday", "news"),
            ("I remember her telling that story", "jokes"),
            ("I remember him telling that story", "jokes"),
            ("I recall you telling me a joke", "jokes"),
            ("I remember them telling the news", "news"),
            ("we remember you telling me the weather", "weather"),
        ],
    )
    def test_more_recall_constructions_are_mentions(self, utterance, intent):
        assert verdict(utterance, intent) == framing.MENTION

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            # The same opening, continued into a real request. The recall
            # is the setup and the request is the point, so the guard has
            # to stand down.
            ("I remember you telling me that, explain it again", "facts"),
            ("I remember you telling that joke, tell me another one", "jokes"),
            ("I remember you telling me the weather, check it", "weather"),
            ("I recall you telling me the news, show me the headlines", "news"),
            ("I remember him telling the story, read it again", "history"),
        ],
    )
    def test_a_later_request_beats_the_recollection(self, utterance, intent):
        assert verdict(utterance, intent) != framing.MENTION

    @pytest.mark.parametrize(
        "utterance,intent",
        [
            ("tell me a joke", "jokes"),
            ("tell me the news", "news"),
            ("can you tell me a joke?", "jokes"),
            ("please tell me the news", "news"),
            ("keep telling me more jokes", "jokes"),
            ("keep telling me more", "jokes"),
            ("what are you telling me?", "news"),
            ("why were you telling me that?", "news"),
            ("did you keep telling him the news?", "news"),
        ],
    )
    def test_genuine_requests_and_questions_are_never_blocked(
        self, utterance, intent
    ):
        """The request side of the gerund, and every bare "tell" cue."""
        assert verdict(utterance, intent) != framing.MENTION

    def test_the_one_named_exception_still_runs(self):
        """"I remember you telling me that, explain it again".

        "explain" is **not** a request cue, so rule 1 does not catch it
        and the stand-down check in the guard is the only thing between
        this sentence and a false block. It is neutral, which is not
        REQUEST, but it is not blocked either, and nothing regressed.
        """
        tokens = normalize("I remember you telling me that, explain it again").tokens
        assert framing._recalls_speech(tokens) is False
        assert "explain" not in framing.REQUEST_CUES
        assert verdict(
            "I remember you telling me that, explain it again", "facts"
        ) == framing.NEUTRAL


    @pytest.mark.parametrize(
        "utterance",
        [
            # The recall verb is missing.
            "you telling me a joke",
            "him telling me the news",
            # The subject is missing.
            "I remember telling me a joke",
            "I recall telling the news",
            # The gerund is missing or different.
            "I remember you said a joke",
            "I remember you tell me a joke",
        ],
    )
    def test_the_clause_must_be_complete(self, utterance):
        """Not a generic gerund rule: the whole clause has to be there."""
        assert framing._recalls_speech(normalize(utterance).tokens) is False

    def test_the_speaker_has_to_be_first_person(self):
        """A third-person recall is not the measured shape."""
        for utterance in (
            "she remembers you telling me a joke",
            "they recall him telling the news",
            "the news reminds me of you telling jokes",
        ):
            assert framing._recalls_speech(normalize(utterance).tokens) is False, (
                utterance
            )

    def test_interrogatives_are_not_recall_subjects(self):
        """So a question keeps its own path through the rule order."""
        for word in ("what", "who", "when", "why", "how"):
            assert word not in framing.RECALL_SUBJECTS
            assert word not in framing.RECALL_SPEAKERS
        assert verdict("what do you remember telling me", "news") != framing.MENTION

    def test_telling_was_not_added_to_a_verb_list(self):
        """The gerund is handled by a rule, not by editing the lists."""
        assert "telling" not in framing.REQUEST_CUES
        assert "telling" not in framing.NARRATIVE_VERBS
        assert framing.NARRATIVE_VERBS == frozenset(
            {
                "mentioned", "heard", "told", "discussed", "chatted", "talked",
                "reported", "announced", "explained", "confirmed", "commented",
            }
        )

    def test_the_recall_verbs_are_a_closed_pair(self):
        assert framing.RECALL_VERBS == frozenset({"remember", "recall"})

    def test_the_stand_down_list_can_only_make_framing_permissive(self):
        """Every entry must be a word no current rule treats as a request.

        The list exists to stop the guard from firing, so an entry that
        already reaches rule 1 would be dead weight, and one that reaches
        a remark rule would be actively misleading.
        """
        for word in framing.FOLLOW_UP_VERBS:
            assert word not in framing.REQUEST_CUES, word
            assert word not in framing.NARRATIVE_VERBS, word

    def test_a_fragment_cannot_match(self):
        """The clause needs five tokens, so a fragment is unaffected."""
        for utterance in ("telling", "remember", "i remember", "telling me jokes"):
            assert framing._recalls_speech(normalize(utterance).tokens) is False
            assert verdict(utterance, "jokes") == framing.NEUTRAL

