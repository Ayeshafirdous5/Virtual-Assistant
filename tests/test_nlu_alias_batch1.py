"""Focused tests for the Step 13 alias batch.

Written **before** the production change, so they fail first and cannot
be shaped to fit whatever the change happened to do.

The batch
----------
Three aliases, chosen from the fifteen candidates Step 12 measured as
:data:`~tests.nlu_parser_gap_corpus.CAUSE_MISSING_ALIAS`, and the
smallest set that survived a false-positive sweep:

==============  =========  ==================================================
alias           intent     why it was safe to take
==============  =========  ==================================================
"put on a song" youtube    carries its own verb and sits beside the
                           existing "play a song" family; "put on" on
                           its own never fires
"will it rain"  weather    a weather question; no other tool could ever
                           claim the word "rain"
"is it going     weather    the commonest spoken form of the same
to rain"                    question
==============  =========  ==================================================

Candidates deliberately left for a later batch
----------------------------------------------
Every one of these was tested and dropped, and the reason is recorded
here rather than left in a commit message:

``"amuse me"``
    **Measured false positive:** *"he will amuse me later"* became a joke
    command. The third person is not an imperative, but phrase matching
    cannot see the difference, and *"make me laugh"* above has the same
    weakness already.
``"define"``
    *"the problem is hard to define"* and *"this joke defines my day"*.
    A bare English verb used non-referentially.
``"crack me up"``, ``"look that up"``, ``"in the news today"``
    Each matched a narrative sentence: *"he will crack me up"*, *"I will
    look that up later"*, *"he was big in the news today"*.
``"celsius"``, ``"gossip"``, ``"trivia"``
    Bare nouns. The lexicon already rejected "song", "video" and "stop"
    for this reason, one of which can close the application.
``"power off"``
    The ``system`` intent can terminate the application, and scoring
    holds it to a higher bar on purpose. Not a low-risk first choice.
``"do I need an umbrella"``
    Named no tool at all, so supporting it means guessing.
"""

from __future__ import annotations

import pytest

from assistant.nlu.lexicon import COMMON_ALIASES
from assistant.nlu.parser import AMBIGUOUS, CLEAR, parse
from assistant.nlu.scoring import PHRASE_SCORE, score_intents
from assistant.nlu.normalize import normalize


def verdict(utterance: str, lexicon) -> str:
    """The intent, the confidence, and the score, for one utterance."""
    parsed = parse(utterance, lexicon)
    if parsed is None:
        return "no-match"
    return f"{parsed.name}/{parsed.confidence}/{parsed.score:.2f}"


def matched_intents(utterance: str, lexicon) -> set[str]:
    """Every intent that reached the action boundary, at any stage."""
    return {
        candidate.intent
        for candidate in score_intents(normalize(utterance), lexicon)
        if candidate.score >= 0.65 and candidate.method in ("exact", "phrase")
    }


@pytest.fixture(scope="module")
def lexicon():
    """The real runtime lexicon, built once from the real tools."""
    from assistant.app import build_runtime_lexicon
    from assistant.tools import build_default_router

    return build_runtime_lexicon(build_default_router())


#: The batch, as data. One entry per alias, each naming its intent.
BATCH: tuple[tuple[str, str], ...] = (
    ("put on a song", "youtube"),
    ("will it rain", "weather"),
    ("is it going to rain", "weather"),
)

#: Every alias in the batch, with the exact intended utterance, at least
#: three natural variants, and the intent each must resolve to.
ALIAS_CASES: dict[str, dict] = {
    "put on a song": {
        "intent": "youtube",
        "exact": "put on a song",
        "variants": (
            "put on a song for me",
            "could you put on a song",
            "put on a song please",
        ),
    },
    "will it rain": {
        "intent": "weather",
        "exact": "will it rain",
        "variants": ("will it rain tomorrow", "will it rain today", "will it rain soon"),
    },
    "is it going to rain": {
        "intent": "weather",
        "exact": "is it going to rain",
        "variants": (
            "is it going to rain tomorrow",
            "is it going to rain this week",
            "is it going to rain right now",
        ),
    },
}

#: Sentences near each alias that must **not** become commands. Every one
#: of these was measured against a trial lexicon before the change, and
#: every one of them also fell through **before** the change, so a match
#: now would be a regression rather than a pre-existing behaviour.
#:
#: Sentences that *look* similar but are genuine requests are deliberately
#: absent: "will you put on a song later" and "will it rain on the parade"
#: both resolve, correctly, because both contain the phrase in a question.
NON_COMMANDS: tuple[tuple[str, str], ...] = (
    # "put on a song" in a narrative sentence, and "put on" alone.
    ("he put on a brave face", "youtube"),
    ("put on your coat", "youtube"),
    ("they put on a show last night", "youtube"),
    # The rain phrases stated rather than asked.
    ("it will rain in the story", "weather"),
    ("the rain in africa", "weather"),
    ("she will never admit it", "weather"),
    ("it will rain tomorrow in the novel", "weather"),
    # The alias that was dropped for exactly this reason, still absent.
    ("he will amuse me later", "jokes"),
)


# ----------------------------------------------------------------------
# The batch itself
# ----------------------------------------------------------------------
class TestTheBatchExists:
    def test_every_alias_is_registered(self):
        for alias, intent in BATCH:
            assert alias in COMMON_ALIASES.get(intent, ()), (alias, intent)

    def test_the_batch_is_small(self):
        """A first batch is meant to be small enough to review by eye."""
        assert 3 <= len(BATCH) <= 5

    def test_every_alias_carries_its_own_action_or_question(self):
        """The lexicon's own rule, checked rather than trusted.

        An entry may be a word that is unambiguous on its own, or a
        phrase that names the action. "forecast" is the first kind and
        "make me laugh" is the second. A bare generic verb or noun is
        neither, which is why "define" and "celsius" were rejected.
        """
        for alias, _intent in BATCH:
            assert " " in alias, alias
            assert alias.split()[0] in {
                "amuse", "put", "will", "is", "look", "search", "jot",
            }, alias

    def test_no_rejected_candidate_was_taken(self, ):
        """The candidates tested and dropped must stay dropped."""
        every = {a for values in COMMON_ALIASES.values() for a in values}
        for rejected in (
            "define", "celsius", "gossip", "trivia", "crack me up",
            "look that up", "in the news today", "power off",
        ):
            assert rejected not in every, rejected

    def test_the_existing_aliases_were_not_removed(self):
        for intent, aliases in (
            ("weather", ("forecast", "how hot", "how cold")),
            ("jokes", ("make me laugh",)),
            ("information", ("look up", "search for", "tell me about")),
            ("youtube", ("play a song", "on youtube")),
        ):
            for alias in aliases:
                assert alias in COMMON_ALIASES[intent], (intent, alias)


# ----------------------------------------------------------------------
# Every alias reaches its tool
# ----------------------------------------------------------------------
class TestAliasesResolve:
    @pytest.mark.parametrize("alias", [a for a, _ in BATCH])
    def test_the_exact_utterance_resolves(self, alias, lexicon):
        intent = ALIAS_CASES[alias]["intent"]
        parsed = parse(ALIAS_CASES[alias]["exact"], lexicon)
        assert parsed is not None, alias
        assert parsed.name == intent, alias
        assert parsed.confidence == CLEAR, alias

    @pytest.mark.parametrize("alias", [a for a, _ in BATCH])
    def test_the_natural_variants_resolve(self, alias, lexicon):
        intent = ALIAS_CASES[alias]["intent"]
        for variant in ALIAS_CASES[alias]["variants"]:
            parsed = parse(variant, lexicon)
            assert parsed is not None, variant
            assert parsed.name == intent, variant
            assert parsed.confidence == CLEAR, variant

    @pytest.mark.parametrize("alias", [a for a, _ in BATCH])
    def test_an_alias_is_never_ambiguous_on_its_own(self, alias, lexicon):
        """A single trigger cannot tie, so this is a cheap invariant."""
        parsed = parse(ALIAS_CASES[alias]["exact"], lexicon)
        assert parsed is not None
        assert parsed.confidence != AMBIGUOUS, alias



# ----------------------------------------------------------------------
# False positives: the cost of the batch
# ----------------------------------------------------------------------
class TestNoNewFalsePositives:
    @pytest.mark.parametrize("utterance,intent", NON_COMMANDS)
    def test_a_nearby_sentence_still_falls_through(self, utterance, intent, lexicon):
        """Narrative sentences measured before the change must not match.

        Every entry in :data:`NON_COMMANDS` was checked against a trial
        lexicon first. None of them reached the action boundary then, and
        none may now.
        """
        parsed = parse(utterance, lexicon)
        assert parsed is None or parsed.name != intent, utterance

    def test_the_generic_noun_rule_still_holds(self):
        """The documented failure mode must not have come back."""
        bare = {pattern for values in COMMON_ALIASES.values() for pattern in values}
        for noun in ("song", "video", "music", "stop", "log", "celsius", "gossip"):
            assert noun not in bare, noun

    def test_no_alias_is_a_bare_generic_word(self):
        """Every entry is a phrase, or a word that names its own action."""
        for values in COMMON_ALIASES.values():
            for alias in values:
                assert len(alias.split()) >= 1
                if len(alias.split()) == 1:
                    assert alias in {"forecast"}, alias

    def test_a_weather_mention_matched_before_the_batch_and_still_does(
        self, lexicon
    ):
        """"the weather will rain tomorrow" reaches weather, and did so already.

        Recorded because it is easy to misread as something this batch
        caused. It did not: the match comes from the weather tool's own
        single-word pattern, not from either new alias, and the alias
        sweep below proves the trigger is unchanged.
        """
        parsed = parse("the weather will rain tomorrow", lexicon)
        assert parsed is not None
        assert parsed.name == "weather"
        # The trigger is the tool's own word, not "will it rain".
        assert parsed.trigger == "weather", parsed.trigger

    def test_a_rain_mention_without_a_trigger_still_falls_through(self, lexicon):
        """The case the aliases really could have caused, and did not.

        Neither phrase appears in a statement about the rain, so the
        batch adds nothing here.
        """
        for utterance in ("it will rain in the story", "the rain in africa"):
            assert parse(utterance, lexicon) is None, utterance


# ----------------------------------------------------------------------
# Nothing else moved
# ----------------------------------------------------------------------
class TestExistingBehaviourPreserved:
    @pytest.mark.parametrize(
        "utterance,expected",
        [
            # One working command per intent this batch touched, plus one
            # for an intent it did not.
            ("tell me a joke", "jokes"),
            ("make me laugh", "jokes"),
            ("play a song", "youtube"),
            ("play lofi beats", "youtube"),
            ("what is the weather", "weather"),
            ("how cold is it", "weather"),
            ("tell me about Python", "information"),
            ("top headlines", "news"),
            ("what did I say", "history"),
            ("note buy milk", "notes"),
            ("quit", "system"),
        ],
    )
    def test_existing_commands_do_not_regress(self, utterance, expected, lexicon):
        parsed = parse(utterance, lexicon)
        assert parsed is not None, utterance
        assert parsed.name == expected, utterance

    def test_the_system_intent_was_not_touched(self):
        """The batch must not have added anything to a terminating tool."""
        assert COMMON_ALIASES["system"] == (
            "stop the assistant",
            "stop listening",
            "shut down",
            "that is all",
        )

    def test_the_thresholds_did_not_move(self):
        from assistant.nlu.parser import AMBIGUITY_MARGIN, MIN_CONFIDENCE
        from assistant.nlu.scoring import FUZZY_CUTOFF

        assert MIN_CONFIDENCE == 0.65
        assert AMBIGUITY_MARGIN == 0.05
        assert FUZZY_CUTOFF == 0.80

    def test_framing_was_not_touched(self):
        """The batch is a vocabulary change and nothing more."""
        from assistant.nlu import framing

        assert "tell" in framing.REQUEST_CUES
        assert framing._recalls_speech(
            ("i", "remember", "you", "telling", "me", "a", "joke")
        )
        assert framing._declares_habit(("my", "brother", "tells", "me", "jokes"))

